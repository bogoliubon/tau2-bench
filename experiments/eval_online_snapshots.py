"""
Evaluate selected policy snapshots from an online_refine output.

Loads an online_refine JSON, extracts the policy at each batch update checkpoint
(plus the initial policy as snapshot 0), and evaluates each chosen snapshot on
all test tasks using evaluate_prompt.evaluate_policy().

Snapshots are numbered 0..N where:
  0   = initial policy (before any online update)
  1..N = policy after the 1st..Nth batch update (policy_updated=True steps)

Usage:
    # Evaluate 4 evenly-spaced snapshots (default) on the test split
    python eval_online_snapshots.py \
        --online_refine_path evaluations/online_refine_gpt-4o_seed42.json \
        --n_snapshots 4

    # Evaluate specific snapshots
    python eval_online_snapshots.py \
        --online_refine_path evaluations/online_refine_gpt-4o_seed42.json \
        --snapshots 0 3 5 8

    # Evaluate on all test tasks with concurrency
    python eval_online_snapshots.py \
        --online_refine_path evaluations/online_refine_gpt-4o_seed42.json \
        --n_snapshots 4 \
        --max_concurrency 8
"""

import json
import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
load_dotenv()

from loguru import logger
logger.remove()
logger.add(sys.stderr, level="ERROR")

from tau2.run import get_tasks
from evaluate_prompt import evaluate_policy


def extract_snapshots(online_refine_data: dict) -> List[dict]:
    """Extract all policy snapshots from an online_refine output.

    Returns a list of dicts:
        {"snapshot_idx": int, "step": int or None, "policy": str, "label": str}
    where snapshot_idx=0 is the initial policy.
    """
    steps = online_refine_data.get("steps", [])

    # Snapshot 0: initial policy — first step's policy before any update
    # (batch_size > 1 means early steps have policy_updated=False with the initial policy)
    initial_policy = None
    for s in steps:
        if not s.get("policy_updated"):
            initial_policy = s["policy"]
            break
    # Fallback: if every step has policy_updated=True (batch_size=1), use the
    # initial policy stored in the refinement source
    if initial_policy is None:
        initial_path = online_refine_data.get("config", {}).get("initial_policy_path")
        if initial_path and Path(initial_path).exists():
            with open(initial_path) as f:
                src = json.load(f)
            # Grab the first iteration policy (the starting point of online refine)
            for it in src.get("iterations", []):
                if it.get("policy"):
                    initial_policy = it["policy"]
                    break
    if initial_policy is None:
        raise ValueError("Could not determine initial policy from online_refine output")

    snapshots = [{"snapshot_idx": 0, "step": None, "policy": initial_policy, "label": "initial"}]

    # Snapshots 1..N: steps where policy was updated
    update_steps = [s for s in steps if s.get("policy_updated")]
    update_steps.sort(key=lambda s: s["step"])
    for i, s in enumerate(update_steps, 1):
        snapshots.append({
            "snapshot_idx": i,
            "step": s["step"],
            "policy": s["policy"],
            "label": f"after_batch_{i}",
        })

    return snapshots


def select_snapshots(snapshots: List[dict], indices: Optional[List[int]]) -> List[dict]:
    """Select snapshots by index. If indices is None, return all."""
    if indices is None:
        return snapshots
    max_idx = len(snapshots) - 1
    selected = []
    for idx in indices:
        if idx < 0 or idx > max_idx:
            print(f"[warning] Snapshot index {idx} out of range (0..{max_idx}), skipping")
            continue
        selected.append(snapshots[idx])
    return selected


def evenly_spaced_indices(n_total: int, n_select: int) -> List[int]:
    """Pick n_select evenly spaced indices from 0..n_total-1, always including 0 and n_total-1."""
    if n_select >= n_total:
        return list(range(n_total))
    if n_select == 1:
        return [n_total - 1]
    step = (n_total - 1) / (n_select - 1)
    return sorted(set(round(i * step) for i in range(n_select)))


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate policy snapshots from an online_refine run"
    )
    parser.add_argument(
        "--online_refine_path",
        type=str,
        required=True,
        help="Path to online_refine output JSON",
    )
    parser.add_argument(
        "--snapshots",
        type=int,
        nargs="+",
        default=None,
        help="Snapshot indices to evaluate (0=initial, 1..N=after each batch update). "
             "Mutually exclusive with --n_snapshots.",
    )
    parser.add_argument(
        "--n_snapshots",
        type=int,
        default=4,
        help="Number of evenly-spaced snapshots to evaluate (default: 4). "
             "Always includes snapshot 0 (initial) and the final snapshot.",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default="retail",
        help="Domain name (default: retail)",
    )
    parser.add_argument(
        "--task_split_name",
        type=str,
        default="test",
        help="Task split to evaluate on (default: test)",
    )
    parser.add_argument(
        "--task_ids",
        type=str,
        nargs="+",
        default=None,
        help="Specific task IDs to evaluate (overrides --task_split_name)",
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
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--max_concurrency",
        type=int,
        default=1,
        help="Max concurrent evaluations per snapshot (default: 1)",
    )
    parser.add_argument(
        "--num_trials",
        type=int,
        default=3,
        help="Number of trials per task (default: 3)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to save summary JSON (auto-generated if not specified)",
    )

    args = parser.parse_args()

    if args.snapshots is not None and args.n_snapshots != 4:
        parser.error("--snapshots and --n_snapshots are mutually exclusive")

    # Load online_refine output
    print(f"Loading online_refine output from: {args.online_refine_path}")
    with open(args.online_refine_path) as f:
        online_data = json.load(f)

    # Extract all snapshots
    all_snapshots = extract_snapshots(online_data)
    n_total = len(all_snapshots)
    print(f"Found {n_total} policy snapshots (0=initial, 1..{n_total-1}=after each batch update)")

    # Select which to evaluate
    if args.snapshots is not None:
        indices = args.snapshots
    else:
        indices = evenly_spaced_indices(n_total, args.n_snapshots)

    chosen = select_snapshots(all_snapshots, indices)
    print(f"Evaluating {len(chosen)} snapshots: {[s['snapshot_idx'] for s in chosen]}")

    # Output path
    if args.output_path is None:
        stem = Path(args.online_refine_path).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_path = f"evaluations/snapshot_eval_{stem}_{timestamp}.json"
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)

    # Evaluate each chosen snapshot
    summary = {
        "config": {
            "online_refine_path": args.online_refine_path,
            "snapshot_indices": [s["snapshot_idx"] for s in chosen],
            "domain": args.domain,
            "task_split_name": args.task_split_name,
            "task_ids": args.task_ids,
            "agent_llm": args.agent_llm,
            "user_llm": args.user_llm,
            "seed": args.seed,
        },
        "snapshots": [],
        "timestamp": datetime.now().isoformat(),
    }

    print(f"\n{'='*70}")
    for snap in chosen:
        idx = snap["snapshot_idx"]
        label = snap["label"]
        policy = snap["policy"]
        step_info = f"step {snap['step']}" if snap["step"] is not None else "initial"
        print(f"\n[Snapshot {idx} ({label}, {step_info})] policy length: {len(policy)} chars")
        print(f"  Evaluating on {args.task_split_name} split...")

        eval_results = evaluate_policy(
            policy=policy,
            domain=args.domain,
            task_split_name=args.task_split_name,
            task_ids=args.task_ids,
            llm_agent=args.agent_llm,
            llm_args_agent={"temperature": 0.0},
            llm_user=args.user_llm,
            llm_args_user={"temperature": 0.0},
            seed=args.seed,
            max_concurrency=args.max_concurrency,
            num_trials=args.num_trials,
        )

        print(f"  -> success_rate={eval_results['success_rate']:.2%}  mean_reward={eval_results['mean_reward']:.4f}")

        # Save trajectories in the same format as the tau2 repo (SimulationRun.model_dump())
        task_results = []
        for r in eval_results["task_results"]:
            r_out = {k: v for k, v in r.items() if k != "_simulation"}
            if r.get("_simulation") is not None:
                r_out["trajectory"] = r["_simulation"].model_dump()
            task_results.append(r_out)

        summary["snapshots"].append({
            "snapshot_idx": idx,
            "label": label,
            "step": snap["step"],
            "policy_length": len(policy),
            "success_rate": eval_results["success_rate"],
            "mean_reward": eval_results["mean_reward"],
            "n_tasks": eval_results["n_tasks"],
            "n_successes": eval_results["n_successes"],
            "task_results": task_results,
            "policy": policy,
        })

        # Save incrementally
        with open(args.output_path, "w") as f:
            json.dump(summary, f, indent=2)

    # Final summary table
    print(f"\n{'='*70}")
    print("SNAPSHOT EVALUATION SUMMARY")
    print(f"{'='*70}")
    print(f"{'Snapshot':<12} {'Label':<22} {'Step':<8} {'Success Rate':<15} {'Mean Reward'}")
    print("-" * 70)
    for s in summary["snapshots"]:
        step_str = str(s["step"]) if s["step"] is not None else "—"
        print(f"{s['snapshot_idx']:<12} {s['label']:<22} {step_str:<8} "
              f"{s['success_rate']:<15.2%} {s['mean_reward']:.4f}")

    print(f"\nResults saved to: {args.output_path}")


if __name__ == "__main__":
    main()
