"""
Evaluate a refined policy on held-out train tasks + all test tasks in a single run.

Reads the refinement output JSON to determine which task IDs were used for inference,
then runs evaluation on everything else (held-out train + test).

Usage:
    python eval_with_holdout.py \
        --refinement_path ../evaluations/llm_rollouts/policy_refinement_batch_llm-rollout_74traj_gpt-5-mini.json \
        --tasks_json ../data/tau2/domains/retail/tasks.json \
        --agent_llm gpt-4o \
        --user_llm gpt-4.1 \
        --num_trials 3 \
        --save_to ../evaluations/llm_rollouts/eval_llm-rollout_gpt5mini-refined_gpt-4o_holdout_3trials
"""

import json
import argparse
import subprocess
import sys


def get_all_task_ids_by_split(tasks_json_path):
    """Get all task IDs grouped by split from split_tasks.json (same directory as tasks.json)."""
    import os
    split_path = os.path.join(os.path.dirname(tasks_json_path), "split_tasks.json")
    with open(split_path) as f:
        splits = json.load(f)

    train_ids = set(str(tid) for tid in splits.get("train", []))
    test_ids = set(str(tid) for tid in splits.get("test", []))

    return train_ids, test_ids


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate refined policy on held-out train + test tasks"
    )
    parser.add_argument("--refinement_path", type=str, required=True,
                        help="Path to the policy refinement output JSON")
    parser.add_argument("--tasks_json", type=str, required=True,
                        help="Path to tasks.json")
    parser.add_argument("--agent_llm", type=str, default="gpt-4o")
    parser.add_argument("--user_llm", type=str, default="gpt-4.1")
    parser.add_argument("--num_trials", type=int, default=3)
    parser.add_argument("--save_to", type=str, required=True,
                        help="Save path (passed to tau2 CLI --save-to)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print the command without running it")
    args = parser.parse_args()

    # Load refinement output to get selected task IDs
    with open(args.refinement_path) as f:
        refinement = json.load(f)

    selected_ids = set(str(tid) for tid in refinement["config"]["selected_task_ids"])
    print(f"Policy was inferred from {len(selected_ids)} task IDs: {sorted(selected_ids, key=int)}")

    # Get all train/test IDs
    train_ids, test_ids = get_all_task_ids_by_split(args.tasks_json)
    print(f"Total train tasks: {len(train_ids)}, test tasks: {len(test_ids)}")

    # Compute held-out train IDs
    holdout_train_ids = train_ids - selected_ids
    print(f"Held-out train tasks: {len(holdout_train_ids)}")

    # Eval IDs = held-out train + all test
    eval_ids = sorted(holdout_train_ids | test_ids, key=int)
    print(f"Total eval tasks: {len(eval_ids)} ({len(holdout_train_ids)} held-out train + {len(test_ids)} test)")

    # Build command
    cmd = [
        sys.executable, "-m", "tau2.cli", "run",
        "--domain", "retail",
        "--task-split-name", "base",
        "--task-ids", *eval_ids,
        "--agent-llm", args.agent_llm,
        "--user-llm", args.user_llm,
        "--num-trials", str(args.num_trials),
        "--policy-path", args.refinement_path,
        "--save-to", args.save_to,
    ]

    print(f"\nCommand: {' '.join(cmd)}")

    if args.dry_run:
        print("\n[dry run] Not executing.")
        return

    # Run
    subprocess.run(cmd)


if __name__ == "__main__":
    main()
