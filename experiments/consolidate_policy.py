"""
Consolidate a learned policy by compressing it to a target length.

Usage:
    python consolidate_policy.py --input policy_refinement_all_74traj_gpt-5.json --target-chars 7000
    python consolidate_policy.py --input policy_refinement_all_74traj_gpt-5.json --target-chars 7000 --model gpt-5
"""

import json
import argparse
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

import litellm


def load_policy(path: str) -> str:
    path = Path(path)
    if path.suffix == ".md":
        return path.read_text()
    with open(path, "r") as f:
        data = json.load(f)
    if "iterations" in data:
        for iteration in reversed(data["iterations"]):
            if iteration.get("policy"):
                return iteration["policy"]
    if "final_population" in data:
        return data["final_population"][0]["wiki"]
    if "wiki" in data:
        return data["wiki"]
    raise ValueError(f"Could not find policy in {path}")


def consolidate(policy: str, target_chars: int, model: str) -> str:
    prompt = f"""You are given a customer service agent policy that was extracted from analyzing conversation trajectories.
The policy is too verbose and detailed at {len(policy)} characters.

Consolidate it to approximately {target_chars} characters by:
- Removing redundancy and merging overlapping rules
- Dropping overly specific edge cases that can be inferred from general rules
- Keeping all genuinely distinct rules and constraints
- Using concise language — short bullet points, not paragraphs
- Preserving the structured format (numbered sections with bullet points)

Do NOT add any new rules. Only compress what's already there.

Here is the policy to consolidate:

{policy}

Provide ONLY the consolidated policy text, nothing else."""

    response = litellm.completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


def main():
    parser = argparse.ArgumentParser(description="Consolidate a learned policy")
    parser.add_argument("--input", type=str, required=True, help="Path to policy file")
    parser.add_argument("--target-chars", type=int, default=7000, help="Target character count (default: 7000)")
    parser.add_argument("--model", type=str, default="gpt-5", help="Model to use for consolidation (default: gpt-5)")
    parser.add_argument("--output", type=str, default=None, help="Output path (auto-generated if not specified)")
    args = parser.parse_args()

    print(f"Loading policy from: {args.input}")
    policy = load_policy(args.input)
    print(f"Original length: {len(policy)} characters")
    print(f"Target length: ~{args.target_chars} characters")
    print(f"Model: {args.model}")

    print(f"\nConsolidating...")
    consolidated = consolidate(policy, args.target_chars, args.model)
    print(f"Consolidated length: {len(consolidated)} characters ({len(consolidated)/len(policy):.0%} of original)")

    if args.output is None:
        input_stem = Path(args.input).stem
        args.output = f"consolidated_{input_stem}_{args.target_chars}chars.json"

    output = {
        "original_path": args.input,
        "original_length": len(policy),
        "target_chars": args.target_chars,
        "consolidated_length": len(consolidated),
        "model": args.model,
        "policy": consolidated,
        "iterations": [{"iteration": 1, "policy": consolidated}],
        "timestamp": datetime.now().isoformat(),
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved to: {args.output}")
    print(f"\nPreview:\n{consolidated[:500]}...")


if __name__ == "__main__":
    main()
