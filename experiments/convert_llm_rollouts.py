"""
Convert tau2 evaluation result files (LLM rollout trajectories) into the format
expected by batch_refine.py and iterative_policy_refinement.py.

Input format (tau2 eval output):
  {"simulations": [{"task_id", "trial", "reward_info": {"reward": ...}, "messages": [...], ...}]}

Output format (batch_refine.py expected):
  [{"task_id": int, "trial": int, "reward": float, "traj": [...], "info": {"task": {"actions": [...], "instruction": ...}}}]

Message conversion:
  - tau2 uses "messages" with tool calls embedded differently
  - batch_refine expects "traj" in OpenAI tool_call format
"""

import json
import argparse
from pathlib import Path


def convert_messages_to_traj(messages):
    """Convert tau2 message format to OpenAI-style traj format."""
    traj = []
    call_counter = 0

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "assistant":
            new_msg = {"role": "assistant", "content": content}
            # Handle tool calls
            if "tool_calls" in msg and msg["tool_calls"]:
                tool_calls = []
                for tc in msg["tool_calls"]:
                    if "function" in tc:
                        # Already in OpenAI format
                        call_id = tc.get("id", f"call_{call_counter}")
                        tool_calls.append({
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": tc["function"]["name"],
                                "arguments": tc["function"]["arguments"] if isinstance(tc["function"]["arguments"], str) else json.dumps(tc["function"]["arguments"])
                            }
                        })
                    elif "name" in tc:
                        # Alternative format
                        call_id = tc.get("id", f"call_{call_counter}")
                        args = tc.get("arguments", tc.get("kwargs", {}))
                        tool_calls.append({
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": args if isinstance(args, str) else json.dumps(args)
                            }
                        })
                    call_counter += 1
                if tool_calls:
                    new_msg["tool_calls"] = tool_calls
            traj.append(new_msg)

        elif role == "tool":
            new_msg = {
                "role": "tool",
                "content": content if isinstance(content, str) else json.dumps(content),
            }
            if "tool_call_id" in msg:
                new_msg["tool_call_id"] = msg["tool_call_id"]
            if "name" in msg:
                new_msg["name"] = msg["name"]
            traj.append(new_msg)

        elif role == "user":
            traj.append({"role": "user", "content": content})

        elif role == "system":
            traj.append({"role": "system", "content": content})

    return traj


def load_ground_truth(tasks_json_path):
    """Load ground truth actions from tasks.json."""
    with open(tasks_json_path, "r") as f:
        tasks = json.load(f)

    gt_map = {}
    for task in tasks:
        task_id = task["id"]
        actions = task.get("evaluation_criteria", {}).get("actions", [])
        # Convert to batch_refine expected format
        converted_actions = []
        for a in actions:
            converted_actions.append({
                "name": a["name"],
                "kwargs": a.get("arguments", {})
            })

        instruction = task.get("user_scenario", {}).get("instructions", {}).get("reason_for_call", "")
        gt_map[str(task_id)] = {
            "actions": converted_actions,
            "instruction": instruction
        }

    return gt_map


def convert_eval_to_batch_format(eval_path, tasks_json_path, output_path, trial=None):
    """
    Convert eval result file to batch_refine format.

    Args:
        eval_path: Path to tau2 eval result JSON
        tasks_json_path: Path to tasks.json for ground truth
        output_path: Output path for converted JSON
        trial: If specified, only include this trial number. If None, include all.
    """
    with open(eval_path, "r") as f:
        data = json.load(f)

    simulations = data["simulations"]
    gt_map = load_ground_truth(tasks_json_path)

    converted = []
    for sim in simulations:
        if trial is not None and sim["trial"] != trial:
            continue

        task_id = sim["task_id"]
        task_id_str = str(task_id)
        task_id_int = int(task_id)

        reward = sim["reward_info"]["reward"]
        traj = convert_messages_to_traj(sim["messages"])

        gt = gt_map.get(task_id_str, {"actions": [], "instruction": ""})

        entry = {
            "task_id": task_id_int,
            "trial": sim["trial"],
            "reward": reward,
            "traj": traj,
            "info": {
                "task": {
                    "actions": gt["actions"],
                    "instruction": gt["instruction"]
                }
            }
        }
        converted.append(entry)

    with open(output_path, "w") as f:
        json.dump(converted, f, indent=2)

    n_success = sum(1 for e in converted if e["reward"] > 0)
    print(f"Converted {len(converted)} trajectories ({n_success} successful, {len(converted) - n_success} failed) -> {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert tau2 eval results to batch_refine format")
    parser.add_argument("--eval_path", type=str, required=True, help="Path to tau2 eval result JSON")
    parser.add_argument("--tasks_json", type=str, required=True, help="Path to tasks.json")
    parser.add_argument("--output_path", type=str, required=True, help="Output path")
    parser.add_argument("--trial", type=int, default=None, help="Only include this trial number (default: all)")
    args = parser.parse_args()

    convert_eval_to_batch_format(args.eval_path, args.tasks_json, args.output_path, args.trial)


if __name__ == "__main__":
    main()
