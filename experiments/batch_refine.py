import json
import os
import argparse
import random
from typing import Optional, List, Dict

# Import utility functions
from utils import filter_tasks_by_tool, extract_conversation_text, call_llm


def tool_name_to_description(tool_name: str) -> str:
    """Convert tool name to human-readable description."""
    return tool_name.replace("_", " ")


def batch_policy_refinement(
    results_path: str,
    n_traj: int,
    tool_name: Optional[str] = None,
    model_name: str = "gpt-4o",
    output_path: Optional[str] = None,
    seed: Optional[int] = None,
    success_only: bool = True,
    batch_size: int = 5,
    n_shuffle: int = 1,
    max_policy_chars: Optional[int] = None,
) -> Dict:
    """
    Extract and refine agent policy using batches of trajectories.
    
    Within each batch: Extract from multiple trajectories at once
    Across batches: Iteratively update the policy
    
    Args:
        results_path: Path to results.json with trajectories
        n_traj: Number of trajectories to use
        tool_name: Tool name to filter by (None for all tools)
        model_name: Specific model name
        output_path: Path to save output JSON (if None, auto-generate)
        seed: Random seed for trajectory selection
        success_only: If True, only use successful trajectories
        batch_size: Number of trajectories to process in each batch
        n_shuffle: Number of times to shuffle the same batch 
    
    Returns:
        Dictionary with refinement history
    """
    # Set random seed if provided
    if seed is not None:
        random.seed(seed)
    
    # Filter tasks
    print(f"Filtering tasks (tool={tool_name}, success_only={success_only})...")
    # for non-modify tools, we just filter by tool_name
    if tool_name is not None and tool_name != "modify":
        task_ids_by_tool, task_ids_by_tool_success = filter_tasks_by_tool(results_path, tool_name)
        task_ids = task_ids_by_tool_success if success_only else task_ids_by_tool
    elif tool_name is not None:
        modify_task_ids = []
        modify_task_ids_success = []
        # for modify, we take all modify tasks
        modify_tools=['modify_pending_order_address',
            'modify_pending_order_items',
            'modify_pending_order_payment']
        for task in modify_tools:
            t_ids, t_ids_success = filter_tasks_by_tool(results_path, task)
            modify_task_ids.extend(t_ids)
            modify_task_ids_success.extend(t_ids_success)
        task_ids = list(set(modify_task_ids_success)) if success_only else list(set(modify_task_ids))

    elif tool_name is None and success_only:
        # use all successful tasks
        with open(results_path, "r") as f:
            data = json.load(f)
        task_ids = [result["task_id"] for result in data if result["reward"] > 0.0]

    elif tool_name is None and not success_only:
        # use all tasks (successful and failed)
        with open(results_path, "r") as f:
            data = json.load(f)
        task_ids = [result["task_id"] for result in data]

    else:
        raise NotImplementedError("Unsupported filter combination")
            
    if len(task_ids) < n_traj:
        print(f"[warning] Only {len(task_ids)} tasks available, requested {n_traj}")
        n_traj = len(task_ids)
    
    # Randomly sample tasks
    selected_task_ids = random.sample(task_ids, n_traj)
    print(f"Found {len(task_ids)} matching tasks, randomly selected {n_traj}")
    print(f"Selected task IDs: {selected_task_ids}")
    
    # Initialize results
    refinement_history = {
        "config": {
            "tool_name": tool_name,
            "n_traj": n_traj,
            "batch_size": batch_size,
            "model_name": model_name,
            "results_path": results_path,
            "seed": seed,
            "selected_task_ids": selected_task_ids,
        },
        "iterations": []
    }
    
    current_policy = None
    
    # Shared extraction instructions
    EXTRACTION_INSTRUCTIONS = """Your task is to write an actionable operational policy for a customer service agent, based on the provided trajectories. This policy will be injected into the agent's system prompt as its operating instructions.

Infer the concrete decision procedure that maps:
  (conversation state, retrieved information, system state) → next action

Only extract rules supported by trajectory evidence.
Do not invent best practices or assume unstated policies.

If a behavior appears only once and there is no evidence it is mandatory, mark it as:
  Observed Behavior (not proven mandatory).

For each rule, express in conditional form where possible:
  IF [conditions] → THEN [actions]

Each rule must be:
- Specific and testable
- Based only on observable evidence
- Written as concrete instructions
- Free of abstraction

Do not introduce domain-specific assumptions beyond the trajectory.{length_constraint}"""

    # Prompts for flow-specific extraction
    INITIAL_BATCH_PROMPT_FLOW_SPECIFIC = """You are given {num_traj} full agent trajectories for {flow_description}, each containing user messages, assistant messages, tool calls, and tool outputs.

{trajectories}

""" + EXTRACTION_INSTRUCTIONS

    UPDATE_BATCH_PROMPT_FLOW_SPECIFIC = """You previously extracted the following agent policy for {flow_description}:

{current_policy}

Here are {num_traj} additional trajectories involving {flow_description}:

{trajectories}

Based on these new trajectories, update the policy. Add new rules, refine existing ones, or correct any that are contradicted by new evidence. Provide the COMPLETE updated policy.

""" + EXTRACTION_INSTRUCTIONS

    # Prompts for general extraction (all flows)
    INITIAL_BATCH_PROMPT_GENERAL = """You are given {num_traj} full agent trajectories, each containing user messages, assistant messages, tool calls, and tool outputs.

{trajectories}

""" + EXTRACTION_INSTRUCTIONS

    UPDATE_BATCH_PROMPT_GENERAL = """You previously extracted the following agent policy:

{current_policy}

Here are {num_traj} additional trajectories:

{trajectories}

Based on these new trajectories, update the policy. Add new rules, refine existing ones, or correct any that are contradicted by new evidence. Provide the COMPLETE updated policy.

""" + EXTRACTION_INSTRUCTIONS

    # Build length constraint
    length_constraint = ""
    if max_policy_chars is not None:
        length_constraint = f"\n\nThe policy must not exceed {max_policy_chars} characters. Be concise — use short bullet points, merge overlapping rules, and drop details that can be inferred from general rules."

    INITIAL_BATCH_PROMPT_FLOW_SPECIFIC = INITIAL_BATCH_PROMPT_FLOW_SPECIFIC.replace("{length_constraint}", length_constraint)
    UPDATE_BATCH_PROMPT_FLOW_SPECIFIC = UPDATE_BATCH_PROMPT_FLOW_SPECIFIC.replace("{length_constraint}", length_constraint)
    INITIAL_BATCH_PROMPT_GENERAL = INITIAL_BATCH_PROMPT_GENERAL.replace("{length_constraint}", length_constraint)
    UPDATE_BATCH_PROMPT_GENERAL = UPDATE_BATCH_PROMPT_GENERAL.replace("{length_constraint}", length_constraint)

    # Process in batches
    num_batches = (n_traj + batch_size - 1) // batch_size
    
    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, n_traj)
        batch_task_ids = selected_task_ids[batch_start:batch_end]
        batch_num = batch_idx + 1
        
        print(f"\n[Batch {batch_num}/{num_batches}] Processing {len(batch_task_ids)} trajectories...")

        for shuffle_iter in range(n_shuffle):
            if n_shuffle > 1:
                random.shuffle(batch_task_ids)

            # Extract all trajectories in this batch
            trajectories_text = ""
            for idx, task_id in enumerate(batch_task_ids, 1):
                trial = 0
                trajectory = extract_conversation_text(
                    results_path,
                    task_id,
                    trial,
                    include_instruction=False
                )

                if not trajectory:
                    print(f"[warning] Could not extract trajectory for task_id={task_id}")
                    continue

                trajectories_text += f"=== Conversation {idx} (task_id={task_id}) ===\n{trajectory}\n\n"

            # Build prompt based on whether this is first batch or update
            if tool_name is not None:
                flow_desc = tool_name_to_description(tool_name)
                if current_policy is None:
                    prompt = INITIAL_BATCH_PROMPT_FLOW_SPECIFIC.format(
                        num_traj=len(batch_task_ids),
                        trajectories=trajectories_text,
                        flow_description=flow_desc,
                        tool_name=tool_name
                    )
                else:
                    prompt = UPDATE_BATCH_PROMPT_FLOW_SPECIFIC.format(
                        current_policy=current_policy,
                        num_traj=len(batch_task_ids),
                        trajectories=trajectories_text,
                        flow_description=flow_desc,
                        tool_name=tool_name
                    )
            else:
                # General extraction (all flows)
                if current_policy is None:
                    prompt = INITIAL_BATCH_PROMPT_GENERAL.format(
                        num_traj=len(batch_task_ids),
                        trajectories=trajectories_text
                    )
                else:
                    prompt = UPDATE_BATCH_PROMPT_GENERAL.format(
                        current_policy=current_policy,
                        num_traj=len(batch_task_ids),
                        trajectories=trajectories_text
                    )

            # Call LLM
            print(f"Calling {model_name}...")
            response = call_llm(prompt, model_name)

            # Enforce length limit — if over, ask LLM to compress
            if max_policy_chars is not None and len(response) > max_policy_chars:
                print(f"  Policy too long ({len(response)} chars), asking LLM to compress to {max_policy_chars}...")
                compress_prompt = f"The following policy is {len(response)} characters but must not exceed {max_policy_chars} characters. Compress it by merging overlapping rules, using shorter bullet points, and dropping details that can be inferred. Provide ONLY the compressed policy.\n\n{response}"
                response = call_llm(compress_prompt, model_name)
                print(f"  Compressed to {len(response)} chars")
                if len(response) > max_policy_chars:
                    print(f"  WARNING: Still over limit ({len(response)} > {max_policy_chars} chars) after compression")

            print(f"Policy extracted/updated (length: {len(response)} chars)")
            current_policy = response
            updated = True

            # Record batch
            refinement_history["iterations"].append({
                "batch_num": batch_num,
                "task_ids": batch_task_ids,
                "updated": updated,
                "policy": current_policy
            })
        
    refinement_history["final_policy"] = current_policy
    
    # Save results
    if output_path is None:
        tool_str = tool_name if tool_name else "all_tools"
        output_path = f"policy_refinement_{tool_str}_{n_traj}traj_batch{batch_size}nshuffle_{n_shuffle}_{model_name}.json"
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(refinement_history, f, indent=2)
    
    print(f"\n[Done] Saved refinement history to {output_path}")
    print(f"Final policy length: {len(current_policy) if current_policy else 0} chars")
    
    return refinement_history


def main():
    parser = argparse.ArgumentParser(
        description="Extract and refine agent policy using batches of trajectories"
    )
    parser.add_argument(
        "--results_path",
        type=str,
        required=True,
        help="Path to results.json file"
    )
    parser.add_argument(
        "--n_traj",
        type=int,
        required=True,
        help="Number of trajectories to use"
    )
    parser.add_argument(
        "--tool_name",
        type=str,
        default=None,
        help="Tool name to filter by (e.g., 'exchange_delivered_order_items'). If not specified, use all tools."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt-4o",
        help="Specific model name (e.g., 'claude-sonnet-4-5-20250929', 'gpt-4o')"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to save output JSON (auto-generated if not specified)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for trajectory selection"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=5,
        help="Number of trajectories to process in each batch"
    )

    parser.add_argument(
        "--n_shuffle",
        type=int,
        default=1,
        help="Number of times to shuffle the same batch"
    )
    parser.add_argument(
        "--max_policy_chars",
        type=int,
        default=None,
        help="Max policy length in characters. Adds a length constraint to the prompt. Off by default."
    )
    parser.add_argument(
        "--include_failed",
        action="store_true",
        help="Include failed trajectories (default: only successful ones)"
    )

    args = parser.parse_args()

    # Run refinement
    batch_policy_refinement(
        results_path=args.results_path,
        n_traj=args.n_traj,
        tool_name=args.tool_name,
        model_name=args.model_name,
        output_path=args.output_path,
        seed=args.seed,
        success_only=not args.include_failed,
        batch_size=args.batch_size,
        n_shuffle=args.n_shuffle,
        max_policy_chars=args.max_policy_chars,
    )


if __name__ == "__main__":
    main()