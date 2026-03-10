"""
Convert Claude Code individual task trajectory files into the format
expected by batch_refine.py.

Input: Directory of task_*.json files (claude_code_trajectories/)
Output: Single JSON list in batch_refine format

Usage:
    python3 convert_claude_code_trajectories.py \
        --traj_dir ../claude_code_trajectories \
        --tasks_json ../data/tau2/domains/retail/tasks.json \
        --output_path ../data/tau2/domains/retail/llm_rollouts/agent=claude_code/rollouts.json
"""

import json
import argparse
import glob
from pathlib import Path
from convert_llm_rollouts import convert_messages_to_traj, load_ground_truth


def convert_claude_code_trajectories(traj_dir, tasks_json_path, output_path):
    """Convert individual Claude Code task files to batch_refine format."""
    files = sorted(glob.glob(str(Path(traj_dir) / "task_*.json")))
    print(f"Found {len(files)} task files in {traj_dir}")

    gt_map = load_ground_truth(tasks_json_path)

    converted = []
    for fpath in files:
        with open(fpath, "r") as f:
            data = json.load(f)

        task_id = data["task_id"]
        reward = data["reward"]
        sim_run = data["simulation_run"]
        messages = sim_run.get("messages", [])

        if not messages:
            print(f"  Skipping task {task_id}: no messages")
            continue

        traj = convert_messages_to_traj(messages)
        task_id_str = str(task_id)
        task_id_int = int(task_id)

        gt = gt_map.get(task_id_str, {"actions": [], "instruction": ""})

        entry = {
            "task_id": task_id_int,
            "trial": sim_run.get("trial") or 0,
            "reward": reward,
            "traj": traj,
            "info": {
                "task": {
                    "actions": gt["actions"],
                    "instruction": gt["instruction"],
                }
            },
        }
        converted.append(entry)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(converted, f, indent=2)

    n_success = sum(1 for e in converted if e["reward"] > 0)
    print(f"Converted {len(converted)} trajectories ({n_success} successful, {len(converted) - n_success} failed) -> {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert Claude Code trajectories to batch_refine format")
    parser.add_argument("--traj_dir", type=str, required=True, help="Directory with task_*.json files")
    parser.add_argument("--tasks_json", type=str, required=True, help="Path to tasks.json")
    parser.add_argument("--output_path", type=str, required=True, help="Output path")
    args = parser.parse_args()

    convert_claude_code_trajectories(args.traj_dir, args.tasks_json, args.output_path)


if __name__ == "__main__":
    main()
