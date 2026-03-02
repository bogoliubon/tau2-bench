"""
Evaluate selected policy snapshots from an online_refine output.

Loads an online_refine JSON, extracts the policy at each batch update checkpoint
(plus the initial policy as batch 0), and evaluates each chosen batch on all
test tasks. Each batch evaluation is saved to its own file in a subdirectory
named after the online_refine output file.

Batches are numbered 0..N where:
  0   = initial policy (before any online update)
  1..N = policy after the 1st..Nth batch update (policy_updated=True steps)

Usage:
    # Evaluate 4 evenly-spaced batches (default) on the test split
    python eval_online_snapshots.py \
        --online_refine_path evaluations/online_refine_....json \
        --n_batches 4

    # Evaluate specific batch numbers
    python eval_online_snapshots.py \
        --online_refine_path evaluations/online_refine_....json \
        --batches 0 3 5 8

    # Evaluate with concurrency
    python eval_online_snapshots.py \
        --online_refine_path evaluations/online_refine_....json \
        --n_batches 4 \
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

from collections import defaultdict
from evaluate_prompt import evaluate_policy
from tau2.metrics.agent_metrics import pass_hat_k


def extract_snapshots(online_refine_data: dict) -> List[dict]:
    """Extract all policy snapshots from an online_refine output.

    Returns a list of dicts:
        {"batch_num": int, "step": int or None, "policy": str, "label": str}
    where batch_num=0 is the initial policy.
    """
    steps = online_refine_data.get("steps", [])

    # Batch 0: initial policy — first step's policy before any update
    initial_policy = None
    for s in steps:
        if not s.get("policy_updated"):
            initial_policy = s["policy"]
            break
    # Fallback: if every step has policy_updated=True (batch_size=1), load
    # from the refinement source file
    if initial_policy is None:
        initial_path = online_refine_data.get("config", {}).get("initial_policy_path")
        if initial_path and Path(initial_path).exists():
            with open(initial_path) as f:
                src = json.load(f)
            for it in src.get("iterations", []):
                if it.get("policy"):
                    initial_policy = it["policy"]
                    break
    if initial_policy is None:
        raise ValueError("Could not determine initial policy from online_refine output")

    snapshots = [{"batch_num": 0, "step": None, "policy": initial_policy, "label": "initial"}]

    # Batches 1..N: steps where policy was updated
    update_steps = [s for s in steps if s.get("policy_updated")]
    update_steps.sort(key=lambda s: s["step"])
    for i, s in enumerate(update_steps, 1):
        snapshots.append({
            "batch_num": i,
            "step": s["step"],
            "policy": s["policy"],
            "label": f"batch_{i}",
        })

    return snapshots


def select_snapshots(snapshots: List[dict], indices: Optional[List[int]]) -> List[dict]:
    """Select snapshots by batch number. If indices is None, return all."""
    if indices is None:
        return snapshots
    max_idx = len(snapshots) - 1
    selected = []
    for idx in indices:
        if idx < 0 or idx > max_idx:
            print(f"[warning] Batch {idx} out of range (0..{max_idx}), skipping")
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
        "--batches",
        type=int,
        nargs="+",
        default=None,
        help="Batch numbers to evaluate (0=initial, 1..N=after each batch update). "
             "Mutually exclusive with --n_batches.",
    )
    parser.add_argument(
        "--n_batches",
        type=int,
        default=4,
        help="Number of evenly-spaced batches to evaluate (default: 4). "
             "Always includes batch 0 (initial) and the final batch.",
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
        help="Max concurrent evaluations per batch (default: 1)",
    )
    parser.add_argument(
        "--num_trials",
        type=int,
        default=3,
        help="Number of trials per task (default: 3)",
    )

    args = parser.parse_args()

    if args.batches is not None and args.n_batches != 4:
        parser.error("--batches and --n_batches are mutually exclusive")

    # Load online_refine output
    print(f"Loading online_refine output from: {args.online_refine_path}")
    with open(args.online_refine_path) as f:
        online_data = json.load(f)

    # Extract all snapshots
    all_snapshots = extract_snapshots(online_data)
    n_total = len(all_snapshots)
    print(f"Found {n_total} policy snapshots (0=initial, 1..{n_total-1}=after each batch update)")

    # Select which to evaluate
    if args.batches is not None:
        indices = args.batches
    else:
        indices = evenly_spaced_indices(n_total, args.n_batches)

    chosen = select_snapshots(all_snapshots, indices)
    print(f"Evaluating {len(chosen)} batches: {[s['batch_num'] for s in chosen]}")

    # Output subdirectory named after the online_refine file stem
    stem = Path(args.online_refine_path).stem
    output_dir = Path(args.online_refine_path).parent / stem
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"

    # Summary across all evaluated batches (lightweight, no trajectories)
    summary = {
        "config": {
            "online_refine_path": args.online_refine_path,
            "batch_nums": [s["batch_num"] for s in chosen],
            "domain": args.domain,
            "task_split_name": args.task_split_name,
            "task_ids": args.task_ids,
            "agent_llm": args.agent_llm,
            "user_llm": args.user_llm,
            "seed": args.seed,
            "num_trials": args.num_trials,
        },
        "batches": [],
        "timestamp": datetime.now().isoformat(),
    }

    print(f"\n{'='*70}")
    for snap in chosen:
        batch_num = snap["batch_num"]
        label = snap["label"]
        policy = snap["policy"]
        step_info = f"step {snap['step']}" if snap["step"] is not None else "initial"
        print(f"\n[Batch {batch_num} ({label}, {step_info})] policy length: {len(policy)} chars")
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

        # Compute pass^k using the tau2 formula (https://arxiv.org/pdf/2406.12045)
        task_trial_counts = defaultdict(lambda: {"successes": 0, "trials": 0})
        for r in eval_results["task_results"]:
            tid = r["task_id"]
            task_trial_counts[tid]["trials"] += 1
            if r["success"]:
                task_trial_counts[tid]["successes"] += 1
        n_tasks = len(task_trial_counts)
        pass_ks = {}
        for k in range(1, args.num_trials + 1):
            pass_ks[k] = (
                sum(
                    pass_hat_k(v["trials"], v["successes"], k)
                    for v in task_trial_counts.values()
                    if v["trials"] >= k
                ) / n_tasks
                if n_tasks else 0.0
            )
        pass_str = "  ".join(f"pass^{k}={v:.2%}" for k, v in pass_ks.items())
        print(f"  -> {pass_str}  mean_reward={eval_results['mean_reward']:.4f}")

        # Build per-task results with full trajectories (SimulationRun.model_dump())
        task_results = []
        for r in eval_results["task_results"]:
            r_out = {k: v for k, v in r.items() if k != "_simulation"}
            if r.get("_simulation") is not None:
                r_out["trajectory"] = r["_simulation"].model_dump()
            task_results.append(r_out)

        # Save this batch to its own file inside the subdirectory
        batch_filename = f"batch{batch_num}_{args.domain}_{args.task_split_name}_agent-{args.agent_llm}_seed{args.seed}.json"
        batch_path = output_dir / batch_filename
        batch_output = {
            "source": {
                "online_refine_path": args.online_refine_path,
                "batch_num": batch_num,
                "label": label,
                "step": snap["step"],
                "policy_length": len(policy),
            },
            "config": {
                "domain": args.domain,
                "task_split_name": args.task_split_name,
                "task_ids": args.task_ids,
                "agent_llm": args.agent_llm,
                "user_llm": args.user_llm,
                "num_trials": args.num_trials,
                "seed": args.seed,
            },
            "results": {
                "n_tasks": eval_results["n_tasks"],
                "n_trials": eval_results["n_trials"],
                "n_total_runs": eval_results["n_total_runs"],
                "n_successes": eval_results["n_successes"],
                "success_rate": eval_results["success_rate"],
                "mean_reward": eval_results["mean_reward"],
                **{f"pass_{k}": v for k, v in pass_ks.items()},
            },
            "task_results": task_results,
            "policy": policy,
            "timestamp": datetime.now().isoformat(),
        }
        with open(batch_path, "w") as f:
            json.dump(batch_output, f, indent=2)
        print(f"  Saved to: {batch_path}")

        # Update summary (no trajectories) and save incrementally
        summary["batches"].append({
            "batch_num": batch_num,
            "label": label,
            "step": snap["step"],
            "policy_length": len(policy),
            "success_rate": eval_results["success_rate"],
            "mean_reward": eval_results["mean_reward"],
            "n_tasks": eval_results["n_tasks"],
            "n_successes": eval_results["n_successes"],
            **{f"pass_{k}": v for k, v in pass_ks.items()},
            "output_file": str(batch_path),
        })
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

    # Final summary table
    print(f"\n{'='*70}")
    print("SNAPSHOT EVALUATION SUMMARY")
    print(f"{'='*70}")
    print(f"{'Batch':<8} {'Label':<18} {'Step':<8} {'pass^1':<10} {'pass^2':<10} {'pass^3':<10} {'Mean Reward'}")
    print("-" * 75)
    for s in summary["batches"]:
        step_str = str(s["step"]) if s["step"] is not None else "—"
        p1 = f"{s.get('pass_1', 0):.2%}"
        p2 = f"{s.get('pass_2', 0):.2%}"
        p3 = f"{s.get('pass_3', 0):.2%}"
        print(f"{s['batch_num']:<8} {s['label']:<18} {step_str:<8} "
              f"{p1:<10} {p2:<10} {p3:<10} {s['mean_reward']:.4f}")

    print(f"\nResults saved to: {output_dir}/")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
