"""
Online policy refinement from agent rollouts.

Closed-loop approach: load an initial policy from batch_refine output,
then iteratively refine it by running the agent on held-out tasks in batches
and asking an LLM to update the policy after each batch of rollouts.

Usage:
    python3 online_refine.py \
      --refinement_path evaluations/shortened_prompt/policy_refinement_tudor_74traj_gpt-5-mini_seed42.json \
      --agent_llm gpt-4o \
      --user_llm gpt-4.1 \
      --refine_llm gpt-5-mini \
      --batch_size 5 \
      --start_batch 5 \
      --seed 42 \
      --output_path evaluations/shortened_prompt/online_refine_tudor_gpt5mini_seed42.json
"""

import json
import argparse
import random
import sys
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
load_dotenv()

from loguru import logger
logger.remove()
logger.add(sys.stderr, level="ERROR")

from tau2.run import run_task, get_tasks
from tau2.data_model.message import (
    AssistantMessage,
    UserMessage,
    ToolMessage,
    SystemMessage,
    MultiToolMessage,
)
from utils import call_llm


EXTRACTION_INSTRUCTIONS = """Your task is to write an actionable operational policy for a customer service agent, based on the provided trajectories. This policy will be injected into the agent's system prompt as its operating instructions.

Infer the concrete decision procedure that maps:
  (conversation state, retrieved information, system state) → next action

Only extract rules supported by trajectory evidence.
Do not invent best practices or assume unstated policies.

If a behavior appears only once and there is no evidence it is mandatory, mark it as:
  Observed Behavior (not proven mandatory).

For each rule, express in conditional form where possible:
  IF [conditions] → THEN [actions]

Each rule must be:
- Specific and testable
- Based only on observable evidence
- Written as concrete instructions
- Free of abstraction

Do not introduce domain-specific assumptions beyond the trajectory."""


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
    """Convert SimulationRun.messages to formatted text (same format as extract_conversation_text)."""
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
        test_ids is returned separately so we can identify which tasks are test.
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
    batch_size: int = 1,
    start_batch: int = -1,
):
    """Run online policy refinement loop."""

    # 1. Load initial policy and training task IDs from the specified batch iteration
    print(f"Loading initial policy from: {refinement_path} (start_batch={start_batch})")
    with open(refinement_path, "r") as f:
        refinement_data = json.load(f)

    current_policy, training_task_ids = load_policy_at_batch(refinement_data, start_batch)

    resolved_batch = start_batch if start_batch >= 0 else len(refinement_data.get("iterations", [])) - 1
    print(f"Loaded policy from iteration {resolved_batch} ({resolved_batch + 1} batch(es) of training)")
    print(f"Initial policy length: {len(current_policy)} chars")
    print(f"Policy was trained on {len(training_task_ids)} task IDs")

    # 2. Determine held-out tasks: train-held-out (shuffled) then test (shuffled)
    split_tasks_path = Path(__file__).parent.parent / "data" / "tau2" / "domains" / domain / "split_tasks.json"
    held_out_ids, test_ids = get_held_out_task_ids(str(split_tasks_path), training_task_ids, seed)
    test_id_set = set(test_ids)
    n_train_held = sum(1 for tid in held_out_ids if tid not in test_id_set)
    n_test = sum(1 for tid in held_out_ids if tid in test_id_set)
    print(f"Held-out task IDs: {len(held_out_ids)} tasks ({n_train_held} train-held-out, then {n_test} test)")

    if max_tasks is not None:
        held_out_ids = held_out_ids[:max_tasks]
        print(f"Limiting to {max_tasks} tasks")

    # Dry run: just print task order
    if dry_run:
        print(f"\n[DRY RUN] Would run on {len(held_out_ids)} tasks in this order:")
        for i, tid in enumerate(held_out_ids):
            split_label = "test" if tid in test_id_set else "train"
            print(f"  Step {i}: task_id={tid} ({split_label})")
        return

    # Load task objects
    tasks_by_id = {t.id: t for t in get_tasks(task_set_name=domain)}

    # 3. Initialize output structure
    result = {
        "config": {
            "initial_policy_path": refinement_path,
            "start_batch": start_batch,
            "batch_size": batch_size,
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

    # Check for existing progress to resume
    start_step = 0
    batch_buffer = []  # list of {"task_id": ..., "trajectory_text": ...}
    if Path(output_path).exists():
        with open(output_path, "r") as f:
            existing = json.load(f)
        if existing.get("steps"):
            start_step = len(existing["steps"])
            result["steps"] = existing["steps"]

            # Current policy = policy from the last step that triggered an update
            for s in reversed(existing["steps"]):
                if s.get("policy_updated"):
                    current_policy = s["policy"]
                    break

            # Reconstruct batch buffer: steps since the last policy update
            last_update_step = -1
            for s in existing["steps"]:
                if s.get("policy_updated"):
                    last_update_step = s["step"]

            for s in existing["steps"]:
                if s["step"] > last_update_step and s.get("trajectory_text"):
                    batch_buffer.append({
                        "task_id": s["task_id"],
                        "trajectory_text": s["trajectory_text"],
                    })

            print(
                f"Resuming from step {start_step} "
                f"(batch buffer has {len(batch_buffer)} pending trajectories)"
            )

    rewards = [s["reward"] for s in result["steps"]]

    # Build length constraint suffix (reused in every refinement call)
    length_constraint = ""
    if max_policy_chars is not None:
        length_constraint = (
            f"\n\nThe policy must not exceed {max_policy_chars} characters. "
            "Be concise — use short bullet points, merge overlapping rules, "
            "and drop details that can be inferred from general rules."
        )

    # 4. Main loop
    for step_idx in range(start_step, len(held_out_ids)):
        task_id = held_out_ids[step_idx]
        task = tasks_by_id.get(task_id)
        if task is None:
            print(f"[Step {step_idx}] Task {task_id} not found, skipping")
            continue

        split_label = "test" if task_id in test_id_set else "train"
        print(f"\n[Step {step_idx}/{len(held_out_ids)}] Running task {task_id} ({split_label})...")

        # a. Run agent with current policy
        try:
            sim = run_task(
                domain=domain,
                task=task,
                agent="llm_agent",
                user="user_simulator",
                llm_agent=agent_llm,
                llm_args_agent={"temperature": 0.0},
                llm_user=user_llm,
                llm_args_user={"temperature": 0.0},
                seed=seed,
                policy_override=current_policy,
            )
            reward = sim.reward_info.reward
        except Exception as e:
            print(f"  Error running task {task_id}: {e}")
            step_record = {
                "step": step_idx,
                "task_id": task_id,
                "split": split_label,
                "reward": 0.0,
                "policy_updated": False,
                "policy_length": len(current_policy),
                "policy": current_policy,
                "trajectory_text": None,
                "trajectory": None,
                "error": str(e),
            }
            result["steps"].append(step_record)
            rewards.append(0.0)
            _save_incremental(result, rewards, output_path)
            continue

        rewards.append(reward)
        running_mean = sum(rewards) / len(rewards)
        print(f"  Reward: {reward} | Running mean: {running_mean:.4f} ({sum(r > 0 for r in rewards)}/{len(rewards)})")

        # b. Format rollout trajectory (reward not shown to LLM)
        trajectory_text = format_simulation_messages(sim.messages)

        # c. Add to batch buffer
        batch_buffer.append({"task_id": task_id, "trajectory_text": trajectory_text})
        is_last_task = step_idx == len(held_out_ids) - 1

        # d. Update policy when batch is full or on the last task
        policy_updated = False
        if len(batch_buffer) >= batch_size or is_last_task:
            trajectories_text = ""
            for i, entry in enumerate(batch_buffer, 1):
                trajectories_text += (
                    f"=== Conversation {i} (task_id={entry['task_id']}) ===\n"
                    f"{entry['trajectory_text']}\n\n"
                )

            prompt = ONLINE_REFINE_PROMPT.format(
                current_policy=current_policy,
                num_traj=len(batch_buffer),
                trajectories_text=trajectories_text,
            ) + length_constraint

            print(f"  Calling {refine_llm} for refinement ({len(batch_buffer)} trajectories in batch)...")
            response = call_llm(prompt, refine_llm)

            # Enforce length limit — if over, ask LLM to compress
            if max_policy_chars is not None and len(response) > max_policy_chars:
                print(f"  Policy too long ({len(response)} chars), compressing to {max_policy_chars}...")
                compress_prompt = (
                    f"The following policy is {len(response)} characters but must not exceed "
                    f"{max_policy_chars} characters. Compress it by merging overlapping rules, "
                    "using shorter bullet points, and dropping details that can be inferred. "
                    f"Provide ONLY the compressed policy.\n\n{response}"
                )
                response = call_llm(compress_prompt, refine_llm)
                print(f"  Compressed to {len(response)} chars")

            old_len = len(current_policy)
            current_policy = response
            policy_updated = True
            batch_buffer = []
            print(f"  Policy updated ({old_len} -> {len(current_policy)} chars)")

        # e. Log step (include trajectory_text for batch-buffer reconstruction on resume)
        step_record = {
            "step": step_idx,
            "task_id": task_id,
            "split": split_label,
            "reward": reward,
            "policy_updated": policy_updated,
            "policy_length": len(current_policy),
            "policy": current_policy,
            "trajectory_text": trajectory_text,
            "trajectory": sim.model_dump(),
        }
        result["steps"].append(step_record)

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
    print(f"Total updates: {result['summary']['total_updates']}")
    print(f"Mean reward (all):   {result['summary']['mean_reward']:.4f}")
    print(f"Mean reward (train): {result['summary']['train_mean_reward']:.4f} ({len(train_rewards)} tasks)")
    print(f"Mean reward (test):  {result['summary']['test_mean_reward']:.4f} ({len(test_rewards)} tasks)")
    print(f"Final policy length: {len(current_policy)} chars")
    print(f"Results saved to: {output_path}")

    return result


def _save_incremental(result, rewards, output_path):
    """Save current state after each step."""
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
        description="Online policy refinement from agent rollouts"
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
        help="Just print task order, don't execute",
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
        help="Max policy length in characters. Adds a length constraint to the refinement prompt.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Number of rollouts to accumulate before calling the refine LLM (default: 1)",
    )
    parser.add_argument(
        "--start_batch",
        type=int,
        default=-1,
        help=(
            "0-based index into the batch_refine iterations to use as the starting policy. "
            "-1 (default) loads the last (final) iteration. "
            "E.g. --start_batch 2 uses the policy after the 3rd batch and treats only "
            "those task IDs as already trained."
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
        start_batch=args.start_batch,
    )


if __name__ == "__main__":
    main()
