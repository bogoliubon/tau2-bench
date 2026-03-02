"""
Evaluate a learned policy/wiki on a task split using tau2-bench.

Adapted from tau-bench/tudor_experiments/evaluate_prompt.py for tau2-bench.

This script takes a policy (from iterative_policy_refinement, batch_refine,
a raw .md file, or any saved wiki) and evaluates it on a set of tasks.

Usage:
    # Evaluate the default policy.md on the test split
    python evaluate_prompt.py --policy-path ../data/tau2/domains/retail/policy.md --task-split-name test

    # Evaluate a learned policy from iterative_policy_refinement
    python evaluate_prompt.py --policy-path policy_refinement_all_74traj_gpt-5.json --task-split-name test

    # Evaluate on specific task IDs
    python evaluate_prompt.py --policy-path policy.md --task-ids 0 1 2 3

    # Use a different agent model
    python evaluate_prompt.py --policy-path policy.md --task-split-name test --agent-llm gpt-4o
"""

import json
import argparse
import random
import multiprocessing
import warnings
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor

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
from tau2.run import get_tasks, get_environment_info
from tau2.user.user_simulator import get_global_user_sim_guidelines


def load_policy(path: str) -> str:
    """
    Load a policy from a file. Supports:
    - .md files: read raw text
    - .json with "iterations" (iterative_policy_refinement output): extract last iteration's policy
    - .json with "final_population" (RPE-EGA output): extract best candidate's wiki
    - .json with "wiki" field: extract wiki
    """
    path = Path(path)

    if path.suffix == ".md":
        return path.read_text()

    with open(path, "r") as f:
        data = json.load(f)

    # iterative_policy_refinement / batch_refine output
    if "iterations" in data:
        # Find the last iteration that has a non-null policy
        for iteration in reversed(data["iterations"]):
            if iteration.get("policy"):
                return iteration["policy"]
        raise ValueError(f"No policy found in iterations of {path}")

    # RPE-EGA output
    if "final_population" in data:
        return data["final_population"][0]["wiki"]

    # Simple wiki JSON
    if "wiki" in data:
        return data["wiki"]

    raise ValueError(f"Could not find policy/wiki in {path}")


def evaluate_on_task(
    domain: str,
    task: Task,
    policy: str,
    llm_agent: str,
    llm_args_agent: dict,
    llm_user: str,
    llm_args_user: dict,
    max_steps: int,
    max_errors: int,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Run agent with given policy on a specific task."""

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

        # Extract actions taken
        actions_taken = []
        for msg in simulation.messages:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    actions_taken.append(tc.name)

        result = {
            "task_id": task.id,
            "success": reward_info.reward > 0,
            "reward": reward_info.reward,
            "actions": actions_taken,
            "n_messages": len(simulation.messages),
            "termination_reason": simulation.termination_reason.value,
            "error": None,
        }
        result["_simulation"] = simulation
        return result

    except Exception as e:
        print(f"  Error on task {task.id}: {e}")
        return {
            "task_id": task.id,
            "success": False,
            "reward": 0.0,
            "actions": [],
            "n_messages": 0,
            "termination_reason": "error",
            "error": str(e),
        }


def evaluate_policy(
    policy: str,
    domain: str = "retail",
    task_split_name: str = "test",
    task_ids: Optional[List[str]] = None,
    llm_agent: str = "gpt-4o",
    llm_args_agent: Optional[dict] = None,
    llm_user: str = "gpt-4.1",
    llm_args_user: Optional[dict] = None,
    max_steps: int = 30,
    max_errors: int = 5,
    seed: int = 42,
    max_concurrency: int = 1,
    num_trials: int = 1,
) -> Dict[str, Any]:
    """
    Evaluate a policy on a set of tasks.

    Args:
        policy: The policy text to evaluate
        domain: Domain name
        task_split_name: Task split to evaluate on
        task_ids: Specific task IDs (overrides task_split_name)
        llm_agent: Model for the agent
        llm_args_agent: LLM args for agent
        llm_user: Model for the user simulator
        llm_args_user: LLM args for user simulator
        max_steps: Max conversation steps
        max_errors: Max tool errors
        seed: Random seed
        max_concurrency: Max concurrent evaluations
        num_trials: Number of trials per task

    Returns:
        Dict with evaluation results
    """
    if llm_args_agent is None:
        llm_args_agent = {"temperature": 0.0}
    if llm_args_user is None:
        llm_args_user = {"temperature": 0.0}

    # Load tasks
    tasks = get_tasks(
        task_set_name=domain,
        task_split_name=task_split_name,
        task_ids=task_ids,
    )
    print(f"Evaluating on {len(tasks)} tasks (split: {task_split_name})...")

    random.seed(seed)
    seeds = [random.randint(0, 1000000) for _ in range(num_trials)]

    lock = multiprocessing.Lock()
    results = []
    successes = 0
    total_runs = len(tasks) * num_trials

    def _run(task: Task, trial: int, trial_seed: int, progress_str: str) -> Dict[str, Any]:
        print(f"  {progress_str} Task {task.id} (trial {trial+1})...", end=" ", flush=True)
        result = evaluate_on_task(
            domain=domain,
            task=task,
            policy=policy,
            llm_agent=llm_agent,
            llm_args_agent=llm_args_agent,
            llm_user=llm_user,
            llm_args_user=llm_args_user,
            max_steps=max_steps,
            max_errors=max_errors,
            seed=trial_seed,
        )
        result["trial"] = trial
        status = "SUCCESS" if result["success"] else "FAIL"
        error_msg = f" (error: {result['error']})" if result["error"] else ""
        print(f"{status}{error_msg}")
        return result

    # Build work items
    work_items = []
    for trial in range(num_trials):
        for i, task in enumerate(tasks):
            idx = trial * len(tasks) + i + 1
            progress_str = f"[{idx}/{total_runs}]"
            work_items.append((task, trial, seeds[trial], progress_str))

    if max_concurrency <= 1:
        for task, trial, trial_seed, progress_str in work_items:
            result = _run(task, trial, trial_seed, progress_str)
            results.append(result)
            if result["success"]:
                successes += 1
    else:
        print(f"Running with {max_concurrency} parallel workers...")
        with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
            futures_results = list(executor.map(
                lambda args: _run(*args),
                work_items,
            ))
            for result in futures_results:
                results.append(result)
                if result["success"]:
                    successes += 1

    # Calculate metrics
    success_rate = successes / len(results) if results else 0.0
    mean_reward = sum(r["reward"] for r in results) / len(results) if results else 0.0

    return {
        "n_tasks": len(tasks),
        "n_trials": num_trials,
        "n_total_runs": len(results),
        "n_successes": successes,
        "success_rate": success_rate,
        "mean_reward": mean_reward,
        "task_results": results,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate a policy on tau2-bench tasks")

    parser.add_argument(
        "--policy-path",
        type=str,
        required=True,
        help="Path to policy file (.md for raw text, .json for refinement output)",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default="retail",
        help="Domain to evaluate on (default: retail)",
    )
    parser.add_argument(
        "--task-split-name",
        type=str,
        default="test",
        help="Task split to evaluate on (default: test)",
    )
    parser.add_argument(
        "--task-ids",
        type=str,
        nargs="+",
        default=None,
        help="Specific task IDs to evaluate (overrides --task-split-name)",
    )
    parser.add_argument(
        "--agent-llm",
        type=str,
        default="gpt-4o",
        help="Model for the agent (default: gpt-4o)",
    )
    parser.add_argument(
        "--agent-temperature",
        type=float,
        default=0.0,
        help="Temperature for agent LLM (default: 0.0)",
    )
    parser.add_argument(
        "--user-llm",
        type=str,
        default="gpt-4.1",
        help="Model for user simulator (default: gpt-4.1)",
    )
    parser.add_argument(
        "--user-temperature",
        type=float,
        default=0.0,
        help="Temperature for user LLM (default: 0.0)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=30,
        help="Max conversation steps (default: 30)",
    )
    parser.add_argument(
        "--max-errors",
        type=int,
        default=5,
        help="Max tool errors (default: 5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=1,
        help="Max concurrent evaluations (default: 1)",
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        default=1,
        help="Number of trials per task (default: 1)",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Path to save results JSON (auto-generated if not specified)",
    )
    parser.add_argument(
        "--save-trajectories",
        action="store_true",
        default=False,
        help="Save full conversation trajectories in the output JSON",
    )

    args = parser.parse_args()

    # Load policy
    print("=" * 70)
    print("POLICY EVALUATION")
    print("=" * 70)
    print(f"\nLoading policy from: {args.policy_path}")

    policy = load_policy(args.policy_path)
    print(f"Policy length: {len(policy)} characters")
    print(f"Policy preview: {policy[:200]}...")

    # Run evaluation
    print(f"\n{'='*70}")
    eval_results = evaluate_policy(
        policy=policy,
        domain=args.domain,
        task_split_name=args.task_split_name,
        task_ids=args.task_ids,
        llm_agent=args.agent_llm,
        llm_args_agent={"temperature": args.agent_temperature},
        llm_user=args.user_llm,
        llm_args_user={"temperature": args.user_temperature},
        max_steps=args.max_steps,
        max_errors=args.max_errors,
        seed=args.seed,
        max_concurrency=args.max_concurrency,
        num_trials=args.num_trials,
    )

    # Print summary
    print(f"\n{'='*70}")
    print("EVALUATION RESULTS")
    print(f"{'='*70}")
    print(f"Tasks evaluated: {eval_results['n_tasks']}")
    print(f"Trials per task: {eval_results['n_trials']}")
    print(f"Total runs: {eval_results['n_total_runs']}")
    print(f"Successes: {eval_results['n_successes']}")
    print(f"Success rate: {eval_results['success_rate']:.2%}")
    print(f"Mean reward: {eval_results['mean_reward']:.4f}")

    # Per-task breakdown
    print(f"\nPer-task results:")
    for r in eval_results["task_results"]:
        status = "PASS" if r["success"] else "FAIL"
        print(f"  Task {r['task_id']}: {status} (reward={r['reward']:.2f}, steps={r['n_messages']}, term={r['termination_reason']})")

    # Build serializable task results (strip _simulation unless saving trajectories)
    serializable_task_results = []
    for r in eval_results["task_results"]:
        r_out = {k: v for k, v in r.items() if k != "_simulation"}
        if args.save_trajectories and r.get("_simulation") is not None:
            sim = r["_simulation"]
            r_out["trajectory"] = sim.model_dump()
        serializable_task_results.append(r_out)

    # Save results
    output = {
        "source": {
            "policy_path": args.policy_path,
            "policy_length": len(policy),
        },
        "config": {
            "domain": args.domain,
            "task_split_name": args.task_split_name,
            "task_ids": args.task_ids,
            "agent_llm": args.agent_llm,
            "agent_temperature": args.agent_temperature,
            "user_llm": args.user_llm,
            "user_temperature": args.user_temperature,
            "max_steps": args.max_steps,
            "max_errors": args.max_errors,
            "seed": args.seed,
            "num_trials": args.num_trials,
        },
        "results": {
            "n_tasks": eval_results["n_tasks"],
            "n_trials": eval_results["n_trials"],
            "n_total_runs": eval_results["n_total_runs"],
            "n_successes": eval_results["n_successes"],
            "success_rate": eval_results["success_rate"],
            "mean_reward": eval_results["mean_reward"],
        },
        "task_results": serializable_task_results,
        "policy": policy,
        "timestamp": datetime.now().isoformat(),
    }

    if args.output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        policy_name = Path(args.policy_path).stem
        args.output_path = f"evaluations/eval_{policy_name}_{args.agent_llm}_{args.task_split_name}_{timestamp}.json"

    output_dir = Path(args.output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {args.output_path}")

    return output


if __name__ == "__main__":
    main()
