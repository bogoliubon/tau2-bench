"""
Online policy refinement from agent rollouts.

Closed-loop approach: load an initial policy from batch_refine output,
then iteratively refine it by running the agent on held-out tasks in batches
(concurrently) and asking an LLM to update the policy after each batch.

Usage:
    python3 online_refine.py \
      --refinement_path evaluations/new_prompt/tudor/policy_refinement_batch_tudor_74traj_gpt-5-mini_new_prompt.json \
      --agent_llm gpt-4o \
      --user_llm gpt-4.1 \
      --refine_llm gpt-5-mini \
      --batch_size 5 \
      --max_concurrency 5 \
      --start_batch 5 \
      --seed 42
"""

import json
import argparse
import random
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Dict, Any

# Suppress noisy pydantic serialization warnings from litellm
warnings.filterwarnings("ignore", message=".*serialized value may not be as expected.*")

# Ensure experiments/ is on the path regardless of where the script is invoked from
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

from loguru import logger
logger.remove()
logger.add(sys.stderr, level="ERROR")

from tau2.run import get_tasks
from tau2.agent.llm_agent import LLMAgent
from tau2.data_model.tasks import Task
from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
from tau2.orchestrator.orchestrator import Orchestrator
from tau2.registry import registry
from tau2.data_model.message import (
    AssistantMessage,
    UserMessage,
    ToolMessage,
    SystemMessage,
    MultiToolMessage,
)
from utils import call_llm


ONLINE_REFINE_PROMPT = """You are maintaining an operational policy for a customer service agent.

Current policy:
{current_policy}

The agent just completed {num_traj} conversation(s) using this policy:

{trajectories_text}

Identify genuine agent errors — where the agent's behavior was incorrect, not merely where
the user was unhappy. A user requesting something against policy and being refused is NOT
a policy gap.

Signals of a genuine error:
- Agent took the wrong action when the correct one was unambiguous
- Agent failed to act when action was clearly warranted
- Agent asked unnecessary questions when the answer was retrievable
- Agent was inconsistent with its own stated rules

Output the COMPLETE updated policy:
- Correct rules that caused observed errors
- Add rules only for genuinely uncovered situations
- Merge overlapping or redundant rules into a single more general rule
- Preserve all other rules exactly as written
"""


def load_policy_at_batch(refinement_data: dict, start_batch: int) -> tuple[str, List[str]]:
    """Load policy from a specific batch iteration index (0-based).

    Args:
        refinement_data: The loaded batch_refine JSON data.
        start_batch: 0-based index into iterations. -1 means the last iteration.

    Returns:
        (policy_str, training_task_ids) where training_task_ids are all task IDs
        used in iterations 0..start_batch (i.e. the ones to exclude from held-out).
    """
    iterations = refinement_data.get("iterations", [])
    if not iterations:
        raise ValueError("No iterations found in refinement data")

    if start_batch == -1:
        start_batch = len(iterations) - 1

    if start_batch >= len(iterations):
        raise ValueError(
            f"start_batch={start_batch} exceeds number of iterations ({len(iterations)}). "
            f"Valid range: 0 to {len(iterations) - 1}."
        )

    policy = iterations[start_batch]["policy"]

    # Collect deduplicated training task IDs from iterations 0..start_batch
    seen = set()
    training_task_ids = []
    for it in iterations[: start_batch + 1]:
        for tid in it.get("task_ids", []):
            tid_str = str(tid)
            if tid_str not in seen:
                seen.add(tid_str)
                training_task_ids.append(tid_str)

    return policy, training_task_ids


def format_simulation_messages(messages) -> str:
    """Convert SimulationRun.messages to formatted text."""
    lines = []

    # Build tool_call_id -> (func_name, compact_args) mapping for tool messages
    id2call = {}
    for msg in messages:
        if isinstance(msg, AssistantMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                args_str = json.dumps(tc.arguments, ensure_ascii=False)
                if len(args_str) > 200:
                    args_str = args_str[:199] + "…"
                id2call[tc.id] = (tc.name, args_str)

    # Find first user message
    start_idx = 0
    for i, msg in enumerate(messages):
        if isinstance(msg, UserMessage):
            start_idx = i
            break

    for msg in messages[start_idx:]:
        if isinstance(msg, AssistantMessage):
            if msg.content is None and msg.tool_calls:
                parts = []
                for tc in msg.tool_calls:
                    args_str = json.dumps(tc.arguments, ensure_ascii=False)
                    if len(args_str) > 200:
                        args_str = args_str[:199] + "…"
                    parts.append(f"{tc.name}({args_str})")
                lines.append("[assistant]")
                lines.append("→ tool call(s): " + "; ".join(parts))
            else:
                lines.append("[assistant]")
                if msg.content:
                    lines.append(msg.content)

        elif isinstance(msg, UserMessage):
            lines.append("[user]")
            if msg.content:
                lines.append(msg.content)

        elif isinstance(msg, ToolMessage):
            fname, args_str = id2call.get(msg.id, ("", ""))
            display = fname or ""
            if display and args_str:
                role_tag = f"[tool:{display}({args_str})]"
            elif display:
                role_tag = f"[tool:{display}]"
            else:
                role_tag = "[tool]"
            lines.append(role_tag)
            if msg.content:
                lines.append(msg.content)

        elif isinstance(msg, MultiToolMessage):
            for tm in msg.tool_messages:
                fname, args_str = id2call.get(tm.id, ("", ""))
                display = fname or ""
                if display and args_str:
                    role_tag = f"[tool:{display}({args_str})]"
                elif display:
                    role_tag = f"[tool:{display}]"
                else:
                    role_tag = "[tool]"
                lines.append(role_tag)
                if tm.content:
                    lines.append(tm.content)

        elif isinstance(msg, SystemMessage):
            pass  # skip system messages

    return "\n".join(lines)


def get_held_out_task_ids(
    split_tasks_path: str,
    training_task_ids: List[str],
    seed: int = 42,
) -> tuple[List[str], List[str]]:
    """Compute held-out task IDs: unused train (shuffled) then test (shuffled).

    Returns:
        Tuple of (ordered_ids, test_ids) where ordered_ids has train-held-out first, then test.
    """
    with open(split_tasks_path, "r") as f:
        splits = json.load(f)

    train_ids = set(splits["train"])
    test_ids = set(splits["test"])
    used_ids = set(training_task_ids)

    train_held_out = sorted(train_ids - used_ids, key=lambda x: int(x))
    test_list = sorted(test_ids, key=lambda x: int(x))

    rng = random.Random(seed)
    rng.shuffle(train_held_out)
    rng.shuffle(test_list)

    return train_held_out + test_list, list(test_ids)


def _run_single_task(
    domain: str,
    task: Task,
    current_policy: str,
    agent_llm: str,
    user_llm: str,
    seed: int,
) -> Dict[str, Any]:
    """Run one task with the current policy. Returns a result dict."""
    try:
        environment_constructor = registry.get_env_constructor(domain)
        environment = environment_constructor()

        agent_obj = LLMAgent(
            tools=environment.get_tools(),
            domain_policy=current_policy,
            llm=agent_llm,
            llm_args={"temperature": 0.0},
        )

        try:
            user_tools = environment.get_user_tools()
        except Exception:
            user_tools = None

        UserConstructor = registry.get_user_constructor("user_simulator")
        user_obj = UserConstructor(
            tools=user_tools,
            instructions=str(task.user_scenario),
            llm=user_llm,
            llm_args={"temperature": 0.0},
        )

        orchestrator = Orchestrator(
            domain=domain,
            agent=agent_obj,
            user=user_obj,
            environment=environment,
            task=task,
            max_steps=30,
            max_errors=5,
            seed=seed,
        )
        sim = orchestrator.run()

        reward_info = evaluate_simulation(
            domain=domain,
            task=task,
            simulation=sim,
            evaluation_type=EvaluationType.ALL,
            solo_mode=False,
        )
        sim.reward_info = reward_info
        trajectory_text = format_simulation_messages(sim.messages)

        return {
            "sim": sim,
            "trajectory_text": trajectory_text,
            "reward": reward_info.reward,
            "error": None,
        }
    except Exception as e:
        return {"sim": None, "trajectory_text": None, "reward": 0.0, "error": str(e)}


def online_refine(
    refinement_path: str,
    domain: str = "retail",
    agent_llm: str = "gpt-4o",
    user_llm: str = "gpt-4.1",
    refine_llm: str = "gpt-5-mini",
    seed: int = 42,
    output_path: Optional[str] = None,
    dry_run: bool = False,
    max_tasks: Optional[int] = None,
    max_policy_chars: Optional[int] = None,
    batch_size: int = 5,
    max_concurrency: int = 5,
    start_batch: int = -1,
):
    """Run online policy refinement with concurrent rollouts per batch."""

    # 1. Load initial policy and training task IDs
    print(f"Loading initial policy from: {refinement_path} (start_batch={start_batch})")
    with open(refinement_path, "r") as f:
        refinement_data = json.load(f)

    current_policy, training_task_ids = load_policy_at_batch(refinement_data, start_batch)

    resolved_batch = start_batch if start_batch >= 0 else len(refinement_data.get("iterations", [])) - 1
    print(f"Loaded policy from iteration {resolved_batch} ({resolved_batch + 1} batch(es) of training)")
    print(f"Initial policy length: {len(current_policy)} chars")
    print(f"Policy was trained on {len(training_task_ids)} task IDs")

    # 2. Determine held-out tasks
    split_tasks_path = Path(__file__).parent.parent / "data" / "tau2" / "domains" / domain / "split_tasks.json"
    held_out_ids, test_ids = get_held_out_task_ids(str(split_tasks_path), training_task_ids, seed)
    test_id_set = set(test_ids)
    n_train_held = sum(1 for tid in held_out_ids if tid not in test_id_set)
    n_test = sum(1 for tid in held_out_ids if tid in test_id_set)
    print(f"Held-out task IDs: {len(held_out_ids)} tasks ({n_train_held} train-held-out, then {n_test} test)")

    if max_tasks is not None:
        held_out_ids = held_out_ids[:max_tasks]
        print(f"Limiting to {max_tasks} tasks")

    # Build batches — train and test separately so no batch mixes splits.
    # This guarantees a clean checkpoint after all held-out train tasks are done,
    # at which point the policy has been refined on all 74 training tasks.
    train_held_out = [tid for tid in held_out_ids if tid not in test_id_set]
    test_held_out  = [tid for tid in held_out_ids if tid in test_id_set]
    train_batches = [train_held_out[i:i + batch_size] for i in range(0, len(train_held_out), batch_size)]
    test_batches  = [test_held_out[i:i + batch_size]  for i in range(0, len(test_held_out),  batch_size)]
    batches = train_batches + test_batches
    print(f"Batches: {len(train_batches)} train + {len(test_batches)} test = {len(batches)} total "
          f"(batch_size={batch_size}, concurrency={max_concurrency})")

    if dry_run:
        print(f"\n[DRY RUN] Would run {len(batches)} batches on {len(held_out_ids)} tasks:")
        for b_idx, b_ids in enumerate(batches):
            labels = [("test" if tid in test_id_set else "train") for tid in b_ids]
            print(f"  Batch {b_idx}: {list(zip(b_ids, labels))}")
        return

    # Load task objects
    tasks_by_id = {t.id: t for t in get_tasks(task_set_name=domain)}

    # 3. Initialize output structure
    result = {
        "config": {
            "initial_policy_path": refinement_path,
            "start_batch": start_batch,
            "batch_size": batch_size,
            "max_concurrency": max_concurrency,
            "agent_llm": agent_llm,
            "user_llm": user_llm,
            "refine_llm": refine_llm,
            "seed": seed,
            "training_task_ids": training_task_ids,
            "test_task_ids": test_ids,
            "eval_task_ids_order": held_out_ids,
        },
        "steps": [],
        "final_policy": None,
        "summary": None,
    }

    # Set up output path
    if output_path is None:
        source_stem = Path(refinement_path).stem
        output_path = (
            f"evaluations/online_refine_{source_stem}"
            f"_{domain}"
            f"_startb{resolved_batch}"
            f"_bs{batch_size}"
            f"_agent-{agent_llm}"
            f"_seed{seed}.json"
        )
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resume: find last completed batch (any step with policy_updated=True)
    start_batch_idx = 0
    rewards = []
    if Path(output_path).exists():
        with open(output_path, "r") as f:
            existing = json.load(f)
        if existing.get("steps"):
            result["steps"] = existing["steps"]
            rewards = [s["reward"] for s in result["steps"]]

            # Find the last batch that completed a policy update
            completed_batches = set()
            for s in result["steps"]:
                if s.get("policy_updated"):
                    completed_batches.add(s.get("batch", -1))

            if completed_batches:
                last_done = max(completed_batches)
                start_batch_idx = last_done + 1

                # Restore current policy from the last update
                for s in reversed(result["steps"]):
                    if s.get("policy_updated"):
                        current_policy = s["policy"]
                        break

                print(f"Resuming from batch {start_batch_idx} (completed: {sorted(completed_batches)})")

    length_constraint = ""
    if max_policy_chars is not None:
        length_constraint = (
            f"\n\nThe policy must not exceed {max_policy_chars} characters. "
            "Be concise — use short bullet points, merge overlapping rules, "
            "and drop details that can be inferred from general rules."
        )

    # 4. Main loop over batches
    for batch_idx in range(start_batch_idx, len(batches)):
        batch_task_ids = batches[batch_idx]
        batch_num = batch_idx + 1
        print(f"\n[Batch {batch_num}/{len(batches)}] Running {len(batch_task_ids)} tasks concurrently "
              f"(policy length: {len(current_policy)} chars)...")

        # a. Run all tasks in this batch concurrently
        task_results: Dict[str, Dict] = {}
        workers = min(max_concurrency, len(batch_task_ids))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_tid = {
                executor.submit(
                    _run_single_task,
                    domain,
                    tasks_by_id[tid],
                    current_policy,
                    agent_llm,
                    user_llm,
                    seed,
                ): tid
                for tid in batch_task_ids
                if tasks_by_id.get(tid)
            }
            for future in as_completed(future_to_tid):
                tid = future_to_tid[future]
                r = future.result()
                task_results[tid] = r
                split_label = "test" if tid in test_id_set else "train"
                if r["error"]:
                    print(f"  Task {tid} ({split_label}): ERROR — {r['error']}")
                else:
                    print(f"  Task {tid} ({split_label}): reward={r['reward']}")

        # b. Build trajectories text from successful rollouts (in original batch order)
        trajectories_text = ""
        n_valid = 0
        for i, tid in enumerate(batch_task_ids, 1):
            r = task_results.get(tid, {"trajectory_text": None, "reward": 0.0, "error": "missing"})
            if r.get("trajectory_text"):
                trajectories_text += (
                    f"=== Conversation {i} (task_id={tid}) ===\n"
                    f"{r['trajectory_text']}\n\n"
                )
                n_valid += 1

        # c. Refine policy if we have any trajectories
        policy_updated = False
        if trajectories_text:
            prompt = ONLINE_REFINE_PROMPT.format(
                current_policy=current_policy,
                num_traj=n_valid,
                trajectories_text=trajectories_text,
            ) + length_constraint

            print(f"  Calling {refine_llm} for refinement ({n_valid}/{len(batch_task_ids)} valid trajectories)...")
            new_policy = call_llm(prompt, refine_llm)

            if max_policy_chars is not None and len(new_policy) > max_policy_chars:
                print(f"  Policy too long ({len(new_policy)} chars), compressing...")
                compress_prompt = (
                    f"The following policy is {len(new_policy)} characters but must not exceed "
                    f"{max_policy_chars} characters. Compress it by merging overlapping rules, "
                    "using shorter bullet points, and dropping details that can be inferred. "
                    f"Provide ONLY the compressed policy.\n\n{new_policy}"
                )
                new_policy = call_llm(compress_prompt, refine_llm)
                print(f"  Compressed to {len(new_policy)} chars")

            print(f"  Policy updated ({len(current_policy)} -> {len(new_policy)} chars)")
            current_policy = new_policy
            policy_updated = True
        else:
            print(f"  No valid trajectories in batch — skipping refinement")

        # d. Save one step record per task in the batch
        batch_rewards = []
        for i, tid in enumerate(batch_task_ids):
            r = task_results.get(tid, {"reward": 0.0, "sim": None, "trajectory_text": None, "error": "missing"})
            split_label = "test" if tid in test_id_set else "train"
            is_last_in_batch = (i == len(batch_task_ids) - 1)

            step_record = {
                "step": len(result["steps"]),
                "batch": batch_idx,
                "task_id": tid,
                "split": split_label,
                "reward": r["reward"],
                # Only mark the last step in the batch so eval_online_snapshots gets one snapshot per batch
                "policy_updated": is_last_in_batch and policy_updated,
                "policy_length": len(current_policy),
                "policy": current_policy,
                "trajectory_text": r.get("trajectory_text"),
                "trajectory": r["sim"].model_dump() if r.get("sim") else None,
                "error": r.get("error"),
            }
            result["steps"].append(step_record)
            rewards.append(r["reward"])
            batch_rewards.append(r["reward"])

        running_mean = sum(rewards) / len(rewards)
        batch_mean = sum(batch_rewards) / len(batch_rewards) if batch_rewards else 0.0
        print(f"  Batch mean reward: {batch_mean:.3f} | Running mean: {running_mean:.4f} "
              f"({sum(r > 0 for r in rewards)}/{len(rewards)} successes)")

        _save_incremental(result, rewards, output_path)

    # Final save
    result["final_policy"] = current_policy
    train_rewards = [s["reward"] for s in result["steps"] if s.get("split") == "train"]
    test_rewards = [s["reward"] for s in result["steps"] if s.get("split") == "test"]
    result["summary"] = {
        "total_tasks": len(rewards),
        "total_updates": sum(1 for s in result["steps"] if s.get("policy_updated")),
        "rewards": rewards,
        "mean_reward": sum(rewards) / len(rewards) if rewards else 0.0,
        "train_mean_reward": sum(train_rewards) / len(train_rewards) if train_rewards else 0.0,
        "test_mean_reward": sum(test_rewards) / len(test_rewards) if test_rewards else 0.0,
        "train_count": len(train_rewards),
        "test_count": len(test_rewards),
    }

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n{'='*70}")
    print(f"ONLINE REFINEMENT COMPLETE")
    print(f"{'='*70}")
    print(f"Total tasks: {len(rewards)}")
    print(f"Total policy updates: {result['summary']['total_updates']}")
    print(f"Mean reward (all):   {result['summary']['mean_reward']:.4f}")
    print(f"Mean reward (train): {result['summary']['train_mean_reward']:.4f} ({len(train_rewards)} tasks)")
    print(f"Mean reward (test):  {result['summary']['test_mean_reward']:.4f} ({len(test_rewards)} tasks)")
    print(f"Final policy length: {len(current_policy)} chars")
    print(f"Results saved to: {output_path}")

    return result


def _save_incremental(result, rewards, output_path):
    """Save current state after each batch."""
    result["final_policy"] = result["steps"][-1]["policy"] if result["steps"] else None
    train_rewards = [s["reward"] for s in result["steps"] if s.get("split") == "train"]
    test_rewards = [s["reward"] for s in result["steps"] if s.get("split") == "test"]
    result["summary"] = {
        "total_tasks": len(rewards),
        "total_updates": sum(1 for s in result["steps"] if s.get("policy_updated")),
        "rewards": rewards,
        "mean_reward": sum(rewards) / len(rewards) if rewards else 0.0,
        "train_mean_reward": sum(train_rewards) / len(train_rewards) if train_rewards else 0.0,
        "test_mean_reward": sum(test_rewards) / len(test_rewards) if test_rewards else 0.0,
        "train_count": len(train_rewards),
        "test_count": len(test_rewards),
    }
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Online policy refinement from concurrent agent rollouts"
    )
    parser.add_argument(
        "--refinement_path",
        type=str,
        required=True,
        help="Path to batch_refine output JSON (initial policy source)",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default="retail",
        help="Domain name (default: retail)",
    )
    parser.add_argument(
        "--agent_llm",
        type=str,
        default="gpt-4o",
        help="Model for the agent (default: gpt-4o)",
    )
    parser.add_argument(
        "--user_llm",
        type=str,
        default="gpt-4.1",
        help="Model for the user simulator (default: gpt-4.1)",
    )
    parser.add_argument(
        "--refine_llm",
        type=str,
        default="gpt-5-mini",
        help="Model for policy refinement (default: gpt-5-mini)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to save output JSON (auto-generated if not specified)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Just print batch/task order, don't execute",
    )
    parser.add_argument(
        "--max_tasks",
        type=int,
        default=None,
        help="Limit number of held-out tasks to process (for testing)",
    )
    parser.add_argument(
        "--max_policy_chars",
        type=int,
        default=None,
        help="Max policy length in characters",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=5,
        help="Number of rollouts per batch before calling the refine LLM (default: 5)",
    )
    parser.add_argument(
        "--max_concurrency",
        type=int,
        default=5,
        help="Max concurrent rollouts within a batch (default: 5, i.e. full batch parallelism)",
    )
    parser.add_argument(
        "--start_batch",
        type=int,
        default=-1,
        help=(
            "0-based index into the batch_refine iterations to use as the starting policy. "
            "-1 (default) loads the last (final) iteration."
        ),
    )

    args = parser.parse_args()

    online_refine(
        refinement_path=args.refinement_path,
        domain=args.domain,
        agent_llm=args.agent_llm,
        user_llm=args.user_llm,
        refine_llm=args.refine_llm,
        seed=args.seed,
        output_path=args.output_path,
        dry_run=args.dry_run,
        max_tasks=args.max_tasks,
        max_policy_chars=args.max_policy_chars,
        batch_size=args.batch_size,
        max_concurrency=args.max_concurrency,
        start_batch=args.start_batch,
    )


if __name__ == "__main__":
    main()
