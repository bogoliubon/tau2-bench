"""
Evaluate generated tasks by running an LLM agent on them.

Loads tasks from a custom JSON file (not the registered tasks.json),
runs the agent on each, and reports success rate.

Usage:
    # Run on 10 tasks (quick test)
    python3 scripts/eval_generated_tasks.py \
        --tasks_path data/tau2/domains/retail/generated_tasks.json \
        --n_tasks 10 --agent_llm gpt-4o --user_llm gpt-4.1

    # Run on all tasks with concurrency
    python3 scripts/eval_generated_tasks.py \
        --tasks_path data/tau2/domains/retail/generated_tasks.json \
        --agent_llm gpt-4o --user_llm gpt-4.1 --max_concurrency 5
"""

import json
import argparse
import random
import sys
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
load_dotenv()

from loguru import logger
logger.remove()
logger.add(sys.stderr, level="ERROR")

from tau2.data_model.tasks import Task
from tau2.agent.llm_agent import LLMAgent
from tau2.environment.environment import Environment
from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
from tau2.orchestrator.orchestrator import Orchestrator
from tau2.registry import registry


def evaluate_task(domain, task, policy, llm_agent, llm_args_agent, llm_user, llm_args_user, max_steps, max_errors, seed):
    """Run agent on a single task and return result."""
    try:
        environment_constructor = registry.get_env_constructor(domain)
        environment = environment_constructor()

        agent = LLMAgent(
            tools=environment.get_tools(),
            domain_policy=policy,
            llm=llm_agent,
            llm_args=llm_args_agent,
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
            llm_args=llm_args_user,
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
            "trajectory": simulation.model_dump(),
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
        }


def main():
    parser = argparse.ArgumentParser(description="Evaluate generated tasks")
    parser.add_argument("--tasks_path", type=str, required=True, help="Path to generated tasks JSON")
    parser.add_argument("--policy_path", type=str, default=None,
                        help="Path to policy file (default: data/tau2/domains/retail/policy.md)")
    parser.add_argument("--domain", type=str, default="retail")
    parser.add_argument("--agent_llm", type=str, default="gpt-4o")
    parser.add_argument("--user_llm", type=str, default="gpt-4.1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_tasks", type=int, default=None, help="Limit number of tasks (for testing)")
    parser.add_argument("--max_concurrency", type=int, default=1)
    parser.add_argument("--output_path", type=str, default=None)
    args = parser.parse_args()

    # Load tasks
    with open(args.tasks_path) as f:
        raw_tasks = json.load(f)
    tasks = [Task.model_validate(t) for t in raw_tasks]

    if args.n_tasks:
        tasks = tasks[:args.n_tasks]

    print(f"Loaded {len(tasks)} tasks from {args.tasks_path}")

    # Load policy
    if args.policy_path is None:
        policy_path = Path(__file__).parent.parent / "data" / "tau2" / "domains" / args.domain / "policy.md"
    else:
        policy_path = Path(args.policy_path)

    policy = policy_path.read_text()
    print(f"Policy: {policy_path} ({len(policy)} chars)")

    # Run evaluations
    llm_args_agent = {"temperature": 0.0}
    llm_args_user = {"temperature": 0.0}

    results = []

    def run_one(idx_task):
        idx, task = idx_task
        print(f"  [{idx+1}/{len(tasks)}] Task {task.id}...", end=" ", flush=True)
        result = evaluate_task(
            domain=args.domain,
            task=task,
            policy=policy,
            llm_agent=args.agent_llm,
            llm_args_agent=llm_args_agent,
            llm_user=args.user_llm,
            llm_args_user=llm_args_user,
            max_steps=30,
            max_errors=5,
            seed=args.seed,
        )
        status = "PASS" if result["success"] else "FAIL"
        error_msg = f" ({result['error']})" if result["error"] else ""
        print(f"{status} (reward={result['reward']:.2f}){error_msg}")
        return result

    if args.max_concurrency <= 1:
        for idx, task in enumerate(tasks):
            results.append(run_one((idx, task)))
    else:
        with ThreadPoolExecutor(max_workers=args.max_concurrency) as executor:
            results = list(executor.map(run_one, enumerate(tasks)))

    # Summary
    successes = sum(1 for r in results if r["success"])
    mean_reward = sum(r["reward"] for r in results) / len(results) if results else 0
    errors = sum(1 for r in results if r["error"])

    print(f"\n{'='*60}")
    print(f"RESULTS: {successes}/{len(results)} passed ({successes/len(results)*100:.1f}%)")
    print(f"Mean reward: {mean_reward:.4f}")
    if errors:
        print(f"Errors: {errors}")

    # Per-flow breakdown
    from collections import Counter
    flow_results = {}
    for r, raw in zip(results, raw_tasks[:len(results)]):
        actions = raw.get("evaluation_criteria", {}).get("actions", []) or []
        action_names = [a["name"] for a in actions]
        if "cancel_pending_order" in action_names:
            flow = "cancel"
        elif "return_delivered_order_items" in action_names:
            flow = "return"
        elif "exchange_delivered_order_items" in action_names:
            flow = "exchange"
        elif "modify_pending_order_items" in action_names:
            flow = "modify_items"
        elif "modify_pending_order_address" in action_names:
            flow = "modify_address"
        elif "modify_user_address" in action_names:
            flow = "modify_user_address"
        else:
            flow = "other"
        if flow not in flow_results:
            flow_results[flow] = {"pass": 0, "total": 0}
        flow_results[flow]["total"] += 1
        if r["success"]:
            flow_results[flow]["pass"] += 1

    print(f"\nPer-flow breakdown:")
    for flow, counts in sorted(flow_results.items()):
        rate = counts["pass"] / counts["total"] * 100
        print(f"  {flow}: {counts['pass']}/{counts['total']} ({rate:.0f}%)")

    # Save
    if args.output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_path = f"evaluations/retail/eval_generated_{args.agent_llm}_{timestamp}.json"

    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    output = {
        "config": {
            "tasks_path": args.tasks_path,
            "policy_path": str(policy_path),
            "agent_llm": args.agent_llm,
            "user_llm": args.user_llm,
            "seed": args.seed,
            "n_tasks": len(results),
        },
        "results": {
            "n_tasks": len(results),
            "n_successes": successes,
            "success_rate": successes / len(results) if results else 0,
            "mean_reward": mean_reward,
            "flow_breakdown": flow_results,
        },
        "task_results": results,
        "timestamp": datetime.now().isoformat(),
    }
    with open(args.output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {args.output_path}")


if __name__ == "__main__":
    main()
