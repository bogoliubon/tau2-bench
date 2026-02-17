"""
Convert tau2-bench human trajectories into the tau-bench results format
so they can be consumed by iterative_policy_refinement.py and batch_refine.py.
"""

import json
import glob
import os
import argparse
import re


def load_tasks(tasks_json_path):
    """Load tasks.json and build task_id -> {actions, instruction} mapping."""
    with open(tasks_json_path, "r") as f:
        tasks = json.load(f)

    task_map = {}
    for task in tasks:
        tid = task["id"]
        actions = []
        for a in (task.get("evaluation_criteria") or {}).get("actions") or []:
            actions.append({"name": a["name"], "kwargs": a.get("arguments", {})})

        scenario = (task.get("user_scenario") or {}).get("instructions") or {}
        parts = []
        if scenario.get("known_info"):
            parts.append(scenario["known_info"])
        if scenario.get("reason_for_call"):
            parts.append(scenario["reason_for_call"])
        if scenario.get("task_instructions"):
            parts.append(scenario["task_instructions"])
        if scenario.get("unknown_info"):
            parts.append(scenario["unknown_info"])
        instruction = " ".join(parts)

        task_map[tid] = {"actions": actions, "instruction": instruction}
    return task_map


def convert_messages(messages):
    """Convert tau2-bench messages to tau-bench traj format (OpenAI style)."""
    traj = []
    call_counter = 0
    pending_calls = []

    for msg in messages:
        role = msg.get("role")

        if role == "user":
            traj.append({"role": "user", "content": msg.get("content", "")})

        elif role == "assistant":
            tool_calls = msg.get("tool_calls")
            if tool_calls:
                converted_calls = []
                for tc in tool_calls:
                    call_id = f"call_{call_counter}"
                    call_counter += 1
                    args = tc.get("arguments", {})
                    if isinstance(args, dict):
                        args_str = json.dumps(args)
                    else:
                        args_str = str(args)
                    converted_calls.append({
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": args_str,
                        },
                    })
                    pending_calls.append((call_id, tc["name"]))
                traj.append({
                    "role": "assistant",
                    "content": msg.get("content"),
                    "tool_calls": converted_calls,
                })
            else:
                traj.append({"role": "assistant", "content": msg.get("content", "")})

        elif role == "tool":
            if pending_calls:
                call_id, tool_name = pending_calls.pop(0)
            else:
                call_id, tool_name = "", ""
            traj.append({
                "role": "tool",
                "tool_call_id": call_id,
                "name": tool_name,
                "content": msg.get("content", ""),
            })

        elif role == "system":
            traj.append({"role": "system", "content": msg.get("content", "")})

    return traj


def convert_all(traj_dir, tasks_json_path, output_path):
    task_map = load_tasks(tasks_json_path)

    pattern = os.path.join(traj_dir, "task_*_human.json")
    files = sorted(glob.glob(pattern), key=lambda f: int(re.search(r"task_(\d+)_human", f).group(1)))

    results = []
    for filepath in files:
        task_id_str = re.search(r"task_(\d+)_human", filepath).group(1)
        task_id = int(task_id_str)

        with open(filepath, "r") as f:
            human_traj = json.load(f)

        traj = convert_messages(human_traj.get("messages", []))
        task_info = task_map.get(task_id_str, {"actions": [], "instruction": ""})

        results.append({
            "task_id": task_id,
            "trial": 0,
            "reward": 1.0,
            "traj": traj,
            "info": {
                "task": {
                    "user_id": "",
                    "actions": task_info["actions"],
                    "instruction": task_info["instruction"],
                    "outputs": [],
                },
                "source": "human",
            },
        })

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Converted {len(results)} trajectories -> {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert tau2-bench human trajectories to tau-bench format")
    parser.add_argument("--traj_dir", type=str,
                        default=os.path.join(os.path.dirname(__file__), "data/tau2/human_trajectories/retail"),
                        help="Directory containing task_*_human.json files")
    parser.add_argument("--tasks_json", type=str,
                        default=os.path.join(os.path.dirname(__file__), "data/tau2/domains/retail/tasks.json"),
                        help="Path to tasks.json for ground truth actions")
    parser.add_argument("--output_path", type=str, default="human_trajectories_converted.json",
                        help="Output file path")
    args = parser.parse_args()

    convert_all(args.traj_dir, args.tasks_json, args.output_path)


if __name__ == "__main__":
    main()
