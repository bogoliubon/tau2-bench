"""
Verify that human trajectories get reward 1.0 when evaluated against ground truth.

Supports two input formats:
  --source converted   : single JSON array file (human_trajectories_converted.json)
  --source tau2-dir    : directory of tau2-bench SimulationRun JSON files (task_*_human.json)

Usage:
  python verify_human_trajectories.py --source converted --path experiments/human_trajectories_converted.json
  python verify_human_trajectories.py --source tau2-dir --path data/tau2/human_trajectories_Chloe
"""

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from loguru import logger
logger.remove()
logger.add(sys.stderr, level="ERROR")

from tau2.data_model.message import AssistantMessage, UserMessage, ToolMessage, ToolCall
from tau2.data_model.simulation import SimulationRun, TerminationReason
from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
from tau2.run import get_tasks


def convert_trajectory(traj_messages):
    """Convert human trajectory messages (OpenAI format) to tau2-bench Message objects."""
    messages = []
    for msg in traj_messages:
        role = msg["role"]

        if role == "assistant":
            if msg.get("tool_calls"):
                tool_calls = []
                for tc in msg["tool_calls"]:
                    func = tc["function"]
                    args = func["arguments"]
                    if isinstance(args, str):
                        args = json.loads(args)
                    tool_calls.append(ToolCall(
                        id=tc["id"],
                        name=func["name"],
                        arguments=args,
                        requestor="assistant",
                    ))
                messages.append(AssistantMessage(
                    role="assistant",
                    content=msg.get("content"),
                    tool_calls=tool_calls,
                ))
            else:
                messages.append(AssistantMessage(
                    role="assistant",
                    content=msg.get("content"),
                ))

        elif role == "user":
            if msg.get("tool_calls"):
                tool_calls = []
                for tc in msg["tool_calls"]:
                    func = tc["function"]
                    args = func["arguments"]
                    if isinstance(args, str):
                        args = json.loads(args)
                    tool_calls.append(ToolCall(
                        id=tc["id"],
                        name=func["name"],
                        arguments=args,
                        requestor="user",
                    ))
                messages.append(UserMessage(
                    role="user",
                    content=msg.get("content"),
                    tool_calls=tool_calls,
                ))
            else:
                messages.append(UserMessage(
                    role="user",
                    content=msg.get("content"),
                ))

        elif role == "tool":
            messages.append(ToolMessage(
                id=msg.get("tool_call_id", ""),
                role="tool",
                content=msg.get("content"),
                requestor=msg.get("requestor", "assistant"),
            ))

    return messages


def load_converted(path):
    """Load from single JSON array (human_trajectories_converted.json format)."""
    with open(path) as f:
        trajectories = json.load(f)

    entries = []
    for entry in trajectories:
        task_id = str(entry["task_id"])
        messages = convert_trajectory(entry["traj"])
        simulation = SimulationRun(
            id=f"human_{task_id}",
            task_id=task_id,
            start_time="",
            end_time="",
            duration=0.0,
            termination_reason=TerminationReason.AGENT_STOP,
            messages=messages,
        )
        entries.append((task_id, simulation))
    return entries


def load_tau2_dir(path):
    """Load from a directory of tau2-bench SimulationRun JSON files."""
    dir_path = Path(path)
    entries = []
    for f in sorted(dir_path.glob("task_*_human.json")):
        with open(f) as fp:
            data = json.load(fp)
        simulation = SimulationRun.model_validate(data)
        task_id = str(simulation.task_id)
        entries.append((task_id, simulation))
    return entries


def evaluate_entries(entries, task_map, label):
    print(f"\n{'='*70}")
    print(f"Evaluating: {label} ({len(entries)} trajectories)")
    print(f"{'='*70}")

    results = []
    for i, (task_id, simulation) in enumerate(entries):
        task = task_map.get(task_id)
        if task is None:
            print(f"[{i+1}/{len(entries)}] Task {task_id}: SKIP (not found in train split)")
            results.append({"task_id": task_id, "status": "skip", "reward": None})
            continue

        try:
            reward_info = evaluate_simulation(
                domain="retail",
                task=task,
                simulation=simulation,
                evaluation_type=EvaluationType.ALL,
                solo_mode=False,
            )
            reward = reward_info.reward
            status = "PASS" if reward > 0 else "FAIL"

            detail = ""
            if reward_info.reward_breakdown:
                parts = [f"{k.value}={v:.1f}" for k, v in reward_info.reward_breakdown.items()]
                detail = f" ({', '.join(parts)})"

            if reward == 0 and reward_info.action_checks:
                failed_actions = [ac for ac in reward_info.action_checks if not ac.action_match]
                if failed_actions:
                    detail += f" | {len(failed_actions)} action(s) failed"

            print(f"[{i+1}/{len(entries)}] Task {task_id}: {status} reward={reward:.2f}{detail}")

            results.append({
                "task_id": task_id,
                "status": status,
                "reward": reward,
                "reward_breakdown": {k.value: v for k, v in reward_info.reward_breakdown.items()} if reward_info.reward_breakdown else None,
            })

        except Exception as e:
            print(f"[{i+1}/{len(entries)}] Task {task_id}: ERROR - {e}")
            results.append({"task_id": task_id, "status": "error", "reward": None, "error": str(e)})

    # Summary
    print(f"\n{'='*70}")
    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = sum(1 for r in results if r["status"] == "FAIL")
    errors = sum(1 for r in results if r["status"] == "error")
    skipped = sum(1 for r in results if r["status"] == "skip")
    print(f"PASS: {passed}, FAIL: {failed}, ERROR: {errors}, SKIP: {skipped}")
    print(f"Total: {len(results)}")

    if failed > 0:
        print(f"\nFailed tasks:")
        for r in results:
            if r["status"] == "FAIL":
                print(f"  Task {r['task_id']}: reward={r['reward']:.2f} breakdown={r.get('reward_breakdown')}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Verify human trajectories against ground truth")
    parser.add_argument("--source", type=str, required=True, choices=["converted", "tau2-dir"],
                        help="Input format: 'converted' for JSON array, 'tau2-dir' for directory of SimulationRun files")
    parser.add_argument("--path", type=str, required=True,
                        help="Path to input file or directory")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save results JSON (auto-generated if not specified)")
    args = parser.parse_args()

    # Load tasks
    all_tasks = get_tasks(task_set_name="retail", task_split_name="train")
    task_map = {task.id: task for task in all_tasks}
    print(f"Loaded {len(all_tasks)} train tasks")

    # Load trajectories
    if args.source == "converted":
        entries = load_converted(args.path)
    else:
        entries = load_tau2_dir(args.path)

    # Evaluate
    results = evaluate_entries(entries, task_map, args.path)

    # Save
    out_path = args.output
    if out_path is None:
        out_path = f"experiments/verify_{Path(args.path).stem}_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
