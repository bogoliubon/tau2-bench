"""
Online delta learning pipeline.

The agent runs tasks live with the current policy, then the reflector
proposes deltas based on the resulting trajectories. Deltas are applied
after each batch, so the policy evolves as the agent gains experience.

Pipeline per batch:
  1. Run agent on batch of tasks with current policy → collect trajectories
  2. Reflector proposes deltas based on trajectories vs current playbook
  3. Curator applies deltas → playbook updates
  4. Dedup at the end of the epoch
"""

import json
import argparse
import random
import warnings
from typing import Optional, List, Dict, Any
from datetime import datetime

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic.*")

from dotenv import load_dotenv
load_dotenv()

import sys
from loguru import logger
logger.remove()
logger.add(sys.stderr, level="ERROR")

from tau2.agent.llm_agent import LLMAgent
from tau2.data_model.tasks import Task
from tau2.environment.environment import Environment
from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
from tau2.orchestrator.orchestrator import Orchestrator
from tau2.registry import registry
from tau2.run import get_tasks

from utils import call_llm
from playbook import (
    SECTIONS,
    Playbook,
    DeltaOp,
    OpType,
    parse_json_response,
    curator_apply,
    deduplicate_playbook,
)
from delta_batch_refine import (
    BOOTSTRAP_PROMPT,
    REFLECTOR_PROMPT,
    bootstrap,
    _run_reflector_batch,
)


# ---------------------------------------------------------------------------
# Run agent on a single task
# ---------------------------------------------------------------------------

def run_task(
    domain: str,
    task: Task,
    policy: str,
    llm_agent: str,
    llm_user: str,
    max_steps: int = 30,
    max_errors: int = 5,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Run agent with given policy on a task. Returns result dict with trajectory."""
    try:
        environment_constructor = registry.get_env_constructor(domain)
        environment = environment_constructor()

        agent = LLMAgent(
            tools=environment.get_tools(),
            domain_policy=policy,
            llm=llm_agent,
            llm_args={"temperature": 0.0},
        )

        try:
            user_tools = environment.get_user_tools()
        except Exception:
            user_tools = None

        UserConstructor = registry.get_user_constructor("user_simulator")
        user = UserConstructor(
            tools=user_tools,
            instructions=str(task.user_scenario),
            llm=llm_user,
            llm_args={"temperature": 0.0},
        )

        orchestrator = Orchestrator(
            domain=domain,
            agent=agent,
            user=user,
            environment=environment,
            task=task,
            max_steps=max_steps,
            max_errors=max_errors,
            seed=seed,
        )
        simulation = orchestrator.run()

        reward_info = evaluate_simulation(
            domain=domain,
            task=task,
            simulation=simulation,
            evaluation_type=EvaluationType.ALL,
            solo_mode=False,
        )
        simulation.reward_info = reward_info

        return {
            "task_id": task.id,
            "success": reward_info.reward > 0,
            "reward": reward_info.reward,
            "n_messages": len(simulation.messages),
            "termination_reason": simulation.termination_reason.value,
            "error": None,
            "_simulation": simulation,
        }

    except Exception as e:
        print(f"  Error on task {task.id}: {e}")
        return {
            "task_id": task.id,
            "success": False,
            "reward": 0.0,
            "n_messages": 0,
            "termination_reason": "error",
            "error": str(e),
            "_simulation": None,
        }


# ---------------------------------------------------------------------------
# Extract trajectory text from simulation messages
# ---------------------------------------------------------------------------

def extract_trajectory_from_simulation(simulation) -> str:
    """Convert simulation messages to text format matching extract_conversation_text."""
    if simulation is None:
        return ""

    lines = []
    # Build tool_call_id -> (name, args) mapping
    id2call = {}
    for msg in simulation.messages:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                args_str = json.dumps(tc.arguments, ensure_ascii=False)
                # Compact args
                if len(args_str) > 200:
                    args_str = args_str[:199] + "…"
                id2call[tc.id] = (tc.name, args_str)

    for msg in simulation.messages:
        role = msg.role

        if role == "system":
            continue

        elif role == "assistant":
            if msg.content is not None and msg.content.strip():
                lines.append("[assistant]")
                lines.append(msg.content)
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                parts = []
                for tc in msg.tool_calls:
                    args_str = json.dumps(tc.arguments, ensure_ascii=False)
                    if len(args_str) > 200:
                        args_str = args_str[:199] + "…"
                    parts.append(f"{tc.name}({args_str})")
                lines.append("[assistant]")
                lines.append("→ tool call(s): " + "; ".join(parts))

        elif role == "user":
            content = msg.content if hasattr(msg, "content") else ""
            if content:
                lines.append("[user]")
                lines.append(content)

        elif role == "tool":
            tool_id = msg.id if hasattr(msg, "id") else ""
            fname, _ = id2call.get(tool_id, ("", ""))
            display = fname or "unknown"
            lines.append(f"[tool:{display}]")
            content = msg.content if hasattr(msg, "content") else ""
            if content:
                lines.append(content)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Build trajectory text from live simulation results
# ---------------------------------------------------------------------------

def build_trajectories_from_results(results: List[Dict], include_label: bool = False) -> str:
    """Build trajectory text from a list of run_task results."""
    parts = []
    for idx, r in enumerate(results, 1):
        sim = r.get("_simulation")
        if sim is None:
            continue
        traj_text = extract_trajectory_from_simulation(sim)
        if not traj_text:
            continue
        if include_label:
            status = "SUCCESS" if r["success"] else "FAIL"
            header = f"=== Conversation {idx} ({status}) ==="
        else:
            header = f"=== Conversation {idx} ==="
        parts.append(f"{header}\n{traj_text}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Bootstrap from seed trajectory file (reuse from delta_batch_refine)
# ---------------------------------------------------------------------------

def bootstrap_from_seed(seed_path: str, model_name: str) -> tuple:
    """Bootstrap playbook from a seed trajectory file."""
    from delta_batch_refine import bootstrap, build_trajectories_text

    with open(seed_path) as f:
        seed_data = json.load(f)
    bootstrap_ids = [r["task_id"] for r in seed_data]
    pb, log = bootstrap(seed_path, bootstrap_ids, model_name)
    return pb, log, bootstrap_ids


# ---------------------------------------------------------------------------
# Main online learning pipeline
# ---------------------------------------------------------------------------

def online_learning(
    domain: str = "retail",
    task_split_name: str = "train",
    task_ids: Optional[List[str]] = None,
    seed_path: Optional[str] = None,
    init_playbook_path: Optional[str] = None,
    llm_agent: str = "gpt-4o",
    llm_user: str = "gpt-4.1",
    reflector_model: str = "gpt-5-mini",
    batch_size: int = 5,
    max_steps: int = 30,
    max_errors: int = 5,
    seed: int = 42,
    dedup_threshold: float = 0.85,
    skip_dedup: bool = False,
    include_label: bool = False,
    output_path: Optional[str] = None,
) -> Dict:

    random.seed(seed)

    # --- Load tasks ---
    tasks = get_tasks(
        task_set_name=domain,
        task_split_name=task_split_name,
        task_ids=task_ids,
    )
    print(f"Loaded {len(tasks)} tasks (split: {task_split_name})")

    # Shuffle tasks
    task_list = list(tasks)
    random.shuffle(task_list)

    # --- Bootstrap ---
    diag_log: List[Dict] = []

    if init_playbook_path is not None:
        with open(init_playbook_path) as f:
            init_data = json.load(f)
        playbook = Playbook.from_dict(init_data["final_playbook"])
        bootstrap_ids = init_data["config"].get("bootstrap_task_ids", [])
        print(f"Resuming from playbook: {init_playbook_path} ({len(playbook.bullets)} bullets)")
        diag_log.append({"step": "init_from_file", "path": init_playbook_path,
                         "num_bullets": len(playbook.bullets)})
    elif seed_path is not None:
        playbook, bootstrap_log, bootstrap_ids = bootstrap_from_seed(seed_path, reflector_model)
        diag_log.append(bootstrap_log)
        print(f"Bootstrap from seed: {len(playbook.bullets)} bullets")
    else:
        raise ValueError("Must provide either --seed_path or --init_playbook_path for bootstrap")

    # --- Batch tasks ---
    batches = [
        task_list[i:i + batch_size]
        for i in range(0, len(task_list), batch_size)
    ]
    print(f"Split into {len(batches)} batches of up to {batch_size} tasks")

    # --- Online loop ---
    all_results: List[Dict] = []
    all_deltas: List[Dict] = []
    batch_summaries: List[Dict] = []

    for batch_idx, batch_tasks in enumerate(batches):
        batch_num = batch_idx + 1
        print(f"\n{'='*60}")
        print(f"BATCH {batch_num}/{len(batches)} ({len(batch_tasks)} tasks)")
        print(f"{'='*60}")
        print(f"Playbook: {len(playbook.bullets)} bullets")

        # --- Step 1: Run tasks with current policy ---
        policy_text = playbook.to_prompt_text()
        batch_results = []

        for task in batch_tasks:
            print(f"  Running task {task.id}...", end=" ", flush=True)
            result = run_task(
                domain=domain,
                task=task,
                policy=policy_text,
                llm_agent=llm_agent,
                llm_user=llm_user,
                max_steps=max_steps,
                max_errors=max_errors,
                seed=seed + batch_idx,
            )
            status = "SUCCESS" if result["success"] else "FAIL"
            error_msg = f" ({result['error']})" if result["error"] else ""
            print(f"{status}{error_msg}")
            batch_results.append(result)

        batch_successes = sum(1 for r in batch_results if r["success"])
        print(f"\n  Batch results: {batch_successes}/{len(batch_results)} successful")

        # --- Step 2: Reflector ---
        traj_text = build_trajectories_from_results(batch_results, include_label=include_label)

        if traj_text.strip():
            prompt = REFLECTOR_PROMPT.format(
                playbook=playbook.to_prompt_text(),
                batch_num=batch_num,
                trajectories=traj_text,
            )

            raw_response = call_llm(prompt, reflector_model)

            batch_log = {
                "step": "reflector",
                "batch_num": batch_num,
                "task_ids": [t.id for t in batch_tasks],
                "raw_llm_response": raw_response,
                "parse_error": None,
                "malformed_deltas": [],
                "num_deltas": 0,
            }

            try:
                parsed = parse_json_response(raw_response, reflector_model)
            except Exception as e:
                batch_log["parse_error"] = str(e)
                diag_log.append(batch_log)
                print(f"  [warning] JSON parse failed: {e}. Skipping reflector.")
                parsed = []

            if not isinstance(parsed, list):
                batch_log["parse_error"] = f"Expected list, got {type(parsed).__name__}"
                diag_log.append(batch_log)
                print(f"  [warning] Expected list. Skipping reflector.")
                parsed = []

            deltas = []
            for item in parsed:
                try:
                    op = OpType(item["op"])
                    delta = DeltaOp(
                        op=op,
                        section=item.get("section", "action_execution"),
                        target_id=item.get("target_id"),
                        content=item.get("content", ""),
                        rationale=item.get("rationale", ""),
                        source_task_ids=item.get("source_task_ids", []),
                        batch_num=batch_num,
                    )
                    deltas.append(delta)
                except (KeyError, ValueError) as e:
                    batch_log["malformed_deltas"].append({"raw": item, "error": str(e)})

            batch_log["num_deltas"] = len(deltas)
            diag_log.append(batch_log)
            print(f"  Reflector: {len(deltas)} deltas proposed")

            # --- Step 3: Curator ---
            if deltas:
                playbook, curator_log = curator_apply(playbook, deltas)
                diag_log.append({"step": "curator", "batch_num": batch_num, "actions": curator_log})
                print(f"  Applied → {len(playbook.bullets)} bullets")

            all_deltas.extend([
                {
                    "op": d.op.value,
                    "section": d.section,
                    "target_id": d.target_id,
                    "content": d.content,
                    "rationale": d.rationale,
                    "source_task_ids": d.source_task_ids,
                    "batch_num": d.batch_num,
                }
                for d in deltas
            ])
        else:
            print("  No trajectories to reflect on")

        # Track batch summary
        serializable_results = []
        for r in batch_results:
            sr = {k: v for k, v in r.items() if k != "_simulation"}
            serializable_results.append(sr)

        batch_summaries.append({
            "batch_num": batch_num,
            "task_ids": [t.id for t in batch_tasks],
            "successes": batch_successes,
            "total": len(batch_results),
            "success_rate": batch_successes / len(batch_results) if batch_results else 0,
            "num_bullets": len(playbook.bullets),
            "results": serializable_results,
        })

        all_results.extend(serializable_results)

    # --- Dedup at the end ---
    if not skip_dedup and len(playbook.bullets) >= 2:
        print(f"\n[Dedup] Deduplicating (threshold={dedup_threshold})...")
        playbook, dedup_log = deduplicate_playbook(playbook, reflector_model, dedup_threshold)
        diag_log.append({"step": "dedup", "events": dedup_log})
        print(f"  Final playbook: {len(playbook.bullets)} bullets")

    final_policy = playbook.to_prompt_text()

    # --- Summary ---
    total_successes = sum(r["success"] for r in all_results)
    total_tasks = len(all_results)
    print(f"\n{'='*60}")
    print(f"ONLINE LEARNING COMPLETE")
    print(f"{'='*60}")
    print(f"Tasks run: {total_tasks}")
    print(f"Overall success rate: {total_successes}/{total_tasks} = {total_successes/total_tasks:.2%}" if total_tasks else "No tasks run")
    print(f"Final playbook: {len(playbook.bullets)} bullets, {len(final_policy)} chars")
    print(f"Total deltas applied: {len(all_deltas)}")

    # --- Build output ---
    result = {
        "config": {
            "pipeline": "delta_online_learning",
            "domain": domain,
            "task_split_name": task_split_name,
            "llm_agent": llm_agent,
            "llm_user": llm_user,
            "reflector_model": reflector_model,
            "batch_size": batch_size,
            "max_steps": max_steps,
            "max_errors": max_errors,
            "seed": seed,
            "dedup_threshold": dedup_threshold,
            "seed_path": seed_path,
            "init_playbook_path": init_playbook_path,
            "bootstrap_task_ids": bootstrap_ids if seed_path else [],
        },
        "summary": {
            "total_tasks": total_tasks,
            "total_successes": total_successes,
            "success_rate": total_successes / total_tasks if total_tasks else 0,
            "num_batches": len(batches),
            "num_deltas": len(all_deltas),
            "final_num_bullets": len(playbook.bullets),
            "final_policy_chars": len(final_policy),
        },
        "batch_summaries": batch_summaries,
        "all_deltas": all_deltas,
        "final_playbook": playbook.to_dict(),
        "final_policy": final_policy,
        # backward compat with evaluate_prompt.py
        "iterations": [{
            "batch_num": len(batches),
            "task_ids": [],
            "step": "online_final",
            "playbook": playbook.to_dict(),
            "policy": final_policy,
        }],
        "diagnostic_log": diag_log,
        "timestamp": datetime.now().isoformat(),
    }

    # Save
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"online_learning_{domain}_{llm_agent}_batch{batch_size}_{timestamp}.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"\n[Done] Saved to {output_path}")
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Online delta learning — agent runs tasks live and policy evolves"
    )
    parser.add_argument("--domain", type=str, default="retail",
                        help="Domain (default: retail)")
    parser.add_argument("--task_split_name", type=str, default="train",
                        help="Task split to train on (default: train)")
    parser.add_argument("--task_ids", type=str, nargs="+", default=None,
                        help="Specific task IDs to run (overrides split)")
    parser.add_argument("--seed_path", type=str, default=None,
                        help="Seed trajectory file for bootstrap")
    parser.add_argument("--init_playbook_path", type=str, default=None,
                        help="Resume from a previous run's output JSON")
    parser.add_argument("--agent_llm", type=str, default="gpt-4o",
                        help="Agent model (default: gpt-4o)")
    parser.add_argument("--user_llm", type=str, default="gpt-4.1",
                        help="User simulator model (default: gpt-4.1)")
    parser.add_argument("--reflector_model", type=str, default="gpt-5-mini",
                        help="Reflector/extraction model (default: gpt-5-mini)")
    parser.add_argument("--batch_size", type=int, default=5,
                        help="Tasks per batch (default: 5)")
    parser.add_argument("--max_steps", type=int, default=30,
                        help="Max conversation steps per task (default: 30)")
    parser.add_argument("--max_errors", type=int, default=5,
                        help="Max tool errors per task (default: 5)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--dedup_threshold", type=float, default=0.85,
                        help="Cosine similarity threshold for dedup (default: 0.85)")
    parser.add_argument("--skip_dedup", action="store_true",
                        help="Skip deduplication step")
    parser.add_argument("--include_label", action="store_true",
                        help="Include SUCCESS/FAIL labels in trajectories shown to reflector")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Output JSON path (auto-generated if omitted)")

    args = parser.parse_args()

    online_learning(
        domain=args.domain,
        task_split_name=args.task_split_name,
        task_ids=args.task_ids,
        seed_path=args.seed_path,
        init_playbook_path=args.init_playbook_path,
        llm_agent=args.agent_llm,
        llm_user=args.user_llm,
        reflector_model=args.reflector_model,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        max_errors=args.max_errors,
        seed=args.seed,
        dedup_threshold=args.dedup_threshold,
        skip_dedup=args.skip_dedup,
        include_label=args.include_label,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()
