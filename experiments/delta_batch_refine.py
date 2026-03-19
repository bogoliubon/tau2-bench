"""
ACE-style delta batch refinement pipeline.

Pipeline:
  1. Bootstrap — LLM extracts structured playbook from seed trajectories
  2. Reflector — per-batch delta proposals (playbook is READ-ONLY)
  3. Curator — apply all accumulated deltas (pure Python)
  4. Dedup — embed + LLM merge/resolve near-duplicate bullets
"""

import json
import os
import argparse
import random
from typing import Optional, List, Dict

from utils import filter_tasks_by_tool, extract_conversation_text, call_llm
from playbook import (
    SECTIONS,
    Playbook,
    DeltaOp,
    OpType,
    parse_json_response,
    curator_apply,
    deduplicate_playbook,
)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

BOOTSTRAP_PROMPT = """\
You are given {num_traj} full agent trajectories for a customer service domain. \
Each contains user messages, assistant messages, tool calls, and tool outputs.

{trajectories}

Extract the agent's operational policy as **structured JSON**. \
Only extract rules supported by trajectory evidence — do not invent best practices.

For each rule, use conditional form where possible:
  IF [conditions] → THEN [actions]

Return a JSON object with exactly these four keys, each mapping to a list of rule strings:

{{
  "information_gathering": ["rule 1", "rule 2", ...],
  "action_execution": ["rule 1", ...],
  "confirmation_communication": ["rule 1", ...],
  "precondition_rules": ["rule 1", ...]
}}

Guidelines:
- "information_gathering": when/how to look things up (e.g., retrieve order details before acting)
- "action_execution": when/how to take actions, including tool constraints and sequencing
- "confirmation_communication": when to confirm with user, what to communicate
- "precondition_rules": conditions that must hold before acting (auth, order state, etc.)

Be specific and testable. Avoid vague statements like "verify information" — \
state exactly what is verified and how.

Return ONLY the JSON object, no markdown fences, no explanation."""


REFLECTOR_PROMPT = """\
You are reviewing agent trajectories against an existing policy playbook.

## Current Playbook (READ-ONLY — do not re-output unchanged rules)
{playbook}

## New Trajectories (batch {batch_num})
{trajectories}

Compare these trajectories against the playbook. Propose **only** the changes needed:
- ADD: a rule the playbook is missing (observed in these trajectories)
- MODIFY: an existing rule that needs refinement based on new evidence
- REMOVE: a rule contradicted by these trajectories

Return a JSON array of delta operations. Each element:
{{
  "op": "ADD" | "MODIFY" | "REMOVE",
  "section": "information_gathering" | "action_execution" | "confirmation_communication" | "precondition_rules",
  "target_id": "<bullet ID for MODIFY/REMOVE, null for ADD>",
  "content": "<new or updated rule text (empty string for REMOVE)>",
  "rationale": "<brief explanation>",
  "source_task_ids": [<task IDs from this batch that provide evidence>]
}}

If nothing needs to change, return an empty array: []

Return ONLY the JSON array, no markdown fences, no explanation."""


# ---------------------------------------------------------------------------
# Task selection (copied from batch_refine.py)
# ---------------------------------------------------------------------------

def select_task_ids(
    results_path: str,
    n_traj: int,
    tool_name: Optional[str],
    success_only: bool,
    seed: Optional[int],
) -> List[int]:
    if seed is not None:
        random.seed(seed)

    if tool_name is not None and tool_name != "modify":
        task_ids_all, task_ids_success = filter_tasks_by_tool(results_path, tool_name)
        task_ids = task_ids_success if success_only else task_ids_all
    elif tool_name == "modify":
        modify_tools = [
            "modify_pending_order_address",
            "modify_pending_order_items",
            "modify_pending_order_payment",
        ]
        ids_all, ids_success = [], []
        for t in modify_tools:
            a, s = filter_tasks_by_tool(results_path, t)
            ids_all.extend(a)
            ids_success.extend(s)
        task_ids = list(set(ids_success)) if success_only else list(set(ids_all))
    elif success_only:
        with open(results_path) as f:
            data = json.load(f)
        task_ids = [r["task_id"] for r in data if r["reward"] > 0.0]
    else:
        with open(results_path) as f:
            data = json.load(f)
        task_ids = [r["task_id"] for r in data]

    if len(task_ids) < n_traj:
        print(f"[warning] Only {len(task_ids)} tasks available, requested {n_traj}")
        n_traj = len(task_ids)

    selected = random.sample(task_ids, n_traj)
    print(f"Found {len(task_ids)} matching tasks, selected {n_traj}")
    return selected


# ---------------------------------------------------------------------------
# Build trajectory text for a batch
# ---------------------------------------------------------------------------

def build_trajectories_text(results_path: str, task_ids: List[int]) -> str:
    parts = []
    for idx, tid in enumerate(task_ids, 1):
        traj = extract_conversation_text(results_path, tid, trial=0, include_instruction=False)
        if not traj:
            print(f"[warning] Could not extract trajectory for task_id={tid}")
            continue
        parts.append(f"=== Conversation {idx} (task_id={tid}) ===\n{traj}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Step 1: Bootstrap
# ---------------------------------------------------------------------------

def bootstrap(results_path: str, task_ids: List[int], model_name: str) -> tuple:
    """Returns (playbook, log_entry)."""
    print(f"\n[Bootstrap] Extracting structured playbook from {len(task_ids)} trajectories...")
    traj_text = build_trajectories_text(results_path, task_ids)
    prompt = BOOTSTRAP_PROMPT.format(num_traj=len(task_ids), trajectories=traj_text)

    raw_response = call_llm(prompt, model_name)
    parsed = parse_json_response(raw_response, model_name)

    pb = Playbook()
    for section in SECTIONS:
        rules = parsed.get(section, [])
        if not isinstance(rules, list):
            continue
        for rule in rules:
            if isinstance(rule, str) and rule.strip():
                pb.add_bullet(section, rule.strip(), source_batch=0,
                              source_task_ids=list(task_ids))

    print(f"  Bootstrap playbook: {len(pb.bullets)} bullets")
    log_entry = {
        "step": "bootstrap",
        "raw_llm_response": raw_response,
        "num_bullets": len(pb.bullets),
    }
    return pb, log_entry


# ---------------------------------------------------------------------------
# Step 2: Reflector loop
# ---------------------------------------------------------------------------

def _run_reflector_batch(
    results_path: str,
    batch_ids: List[int],
    batch_num: int,
    playbook: Playbook,
    model_name: str,
) -> tuple:
    """Run reflector on a single batch. Returns (deltas, log_entry)."""
    print(f"\n[Reflector batch {batch_num}] {len(batch_ids)} trajectories...")

    traj_text = build_trajectories_text(results_path, batch_ids)
    prompt = REFLECTOR_PROMPT.format(
        playbook=playbook.to_prompt_text(),
        batch_num=batch_num,
        trajectories=traj_text,
    )

    raw_response = call_llm(prompt, model_name)

    batch_log = {
        "step": "reflector",
        "batch_num": batch_num,
        "task_ids": batch_ids,
        "raw_llm_response": raw_response,
        "parse_error": None,
        "malformed_deltas": [],
        "num_deltas": 0,
    }

    try:
        parsed = parse_json_response(raw_response, model_name)
    except Exception as e:
        batch_log["parse_error"] = str(e)
        print(f"  [warning] JSON parse failed: {e}. Skipping batch.")
        return [], batch_log

    if not isinstance(parsed, list):
        batch_log["parse_error"] = f"Expected list, got {type(parsed).__name__}"
        print(f"  [warning] Expected list, got {type(parsed).__name__}. Skipping.")
        return [], batch_log

    deltas = []
    for item in parsed:
        try:
            op = OpType(item["op"])
            delta = DeltaOp(
                op=op,
                section=item.get("section", "action_execution"),
                target_id=item.get("target_id"),
                content=item.get("content", ""),
                rationale=item.get("rationale", ""),
                source_task_ids=item.get("source_task_ids", []),
                batch_num=batch_num,
            )
            deltas.append(delta)
        except (KeyError, ValueError) as e:
            batch_log["malformed_deltas"].append({"raw": item, "error": str(e)})
            print(f"  [warning] Skipping malformed delta: {e}")

    batch_log["num_deltas"] = len(deltas)
    print(f"  {len(deltas)} deltas proposed")

    return deltas, batch_log


def reflector_loop(
    results_path: str,
    batches: List[List[int]],
    bootstrap_playbook: Playbook,
    model_name: str,
) -> tuple:
    """Run reflector on each batch. Playbook is READ-ONLY — deltas accumulate.
    Returns (all_deltas, log_entries)."""
    all_deltas: List[DeltaOp] = []
    log_entries: List[Dict] = []

    for batch_idx, batch_ids in enumerate(batches):
        batch_num = batch_idx + 2  # batch 1 was bootstrap
        deltas, batch_log = _run_reflector_batch(
            results_path, batch_ids, batch_num, bootstrap_playbook, model_name)
        all_deltas.extend(deltas)
        log_entries.append(batch_log)

    return all_deltas, log_entries


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def _run_single_epoch(
    results_path: str,
    reflector_ids: List[int],
    batch_size: int,
    start_playbook: Playbook,
    model_name: str,
    dedup_threshold: float,
    skip_dedup: bool,
    epoch: int,
    apply_per_batch: bool = False,
) -> tuple:
    """Run one epoch: reflector → curator → dedup.

    If apply_per_batch=False (default): playbook is frozen during reflector,
    all deltas accumulated and applied once at the end.

    If apply_per_batch=True: after each batch's reflector, apply deltas
    immediately and feed updated playbook to next batch.

    Returns (final_pb, epoch_result_dict, diag_log_entries)."""

    reflector_batches = [
        reflector_ids[i:i + batch_size]
        for i in range(0, len(reflector_ids), batch_size)
    ]

    diag_log: List[Dict] = []
    all_deltas: List[DeltaOp] = []

    if not apply_per_batch:
        # --- Frozen mode: reflector sees same playbook for all batches ---
        if reflector_batches:
            all_deltas, reflector_logs = reflector_loop(
                results_path, reflector_batches, start_playbook, model_name)
            diag_log.extend(reflector_logs)
        print(f"\n[Epoch {epoch} Summary] {len(all_deltas)} total deltas across {len(reflector_batches)} batches")

        # Apply all at once
        print(f"\n[Epoch {epoch} Curator] Applying deltas...")
        curated_pb, curator_log = curator_apply(start_playbook, all_deltas)
        diag_log.append({"step": "curator", "epoch": epoch, "actions": curator_log})
        print(f"  Curated playbook: {len(curated_pb.bullets)} bullets")

    else:
        # --- Per-batch mode: apply after each batch, reflector sees updated playbook ---
        current_pb = start_playbook
        print(f"  [per-batch mode] Applying deltas after each batch")

        for batch_idx, batch_ids in enumerate(reflector_batches):
            batch_num = batch_idx + 2

            deltas, batch_log = _run_reflector_batch(
                results_path, batch_ids, batch_num, current_pb, model_name)
            all_deltas.extend(deltas)
            diag_log.append(batch_log)

            # Apply this batch's deltas immediately
            if deltas:
                current_pb, curator_log = curator_apply(current_pb, deltas)
                diag_log.append({"step": "curator_per_batch", "epoch": epoch,
                                 "batch_num": batch_num, "actions": curator_log})
                print(f"  Applied → {len(current_pb.bullets)} bullets")

        print(f"\n[Epoch {epoch} Summary] {len(all_deltas)} total deltas across {len(reflector_batches)} batches")
        curated_pb = current_pb

    # --- Dedup ---
    if not skip_dedup and len(curated_pb.bullets) >= 2:
        print(f"\n[Epoch {epoch} Dedup] Deduplicating (threshold={dedup_threshold})...")
        final_pb, dedup_log = deduplicate_playbook(curated_pb, model_name, dedup_threshold)
        diag_log.append({"step": "dedup", "epoch": epoch, "events": dedup_log})
        print(f"  Final playbook: {len(final_pb.bullets)} bullets")
    else:
        final_pb = curated_pb

    epoch_result = {
        "epoch": epoch,
        "apply_per_batch": apply_per_batch,
        "num_deltas": len(all_deltas),
        "deltas": [
            {
                "op": d.op.value,
                "section": d.section,
                "target_id": d.target_id,
                "content": d.content,
                "rationale": d.rationale,
                "source_task_ids": d.source_task_ids,
                "batch_num": d.batch_num,
            }
            for d in all_deltas
        ],
        "start_playbook": start_playbook.to_dict(),
        "curated_playbook": curated_pb.to_dict(),
        "final_playbook": final_pb.to_dict(),
        "final_policy": final_pb.to_prompt_text(),
        "num_bullets_start": len(start_playbook.bullets),
        "num_bullets_final": len(final_pb.bullets),
    }

    return final_pb, epoch_result, diag_log


def delta_batch_refine(
    results_path: str,
    n_traj: int,
    tool_name: Optional[str] = None,
    model_name: str = "gpt-4o",
    output_path: Optional[str] = None,
    seed: Optional[int] = None,
    success_only: bool = True,
    batch_size: int = 5,
    dedup_threshold: float = 0.85,
    skip_dedup: bool = False,
    seed_path: Optional[str] = None,
    num_epochs: int = 1,
    init_playbook_path: Optional[str] = None,
    apply_per_batch: bool = False,
) -> Dict:
    # --- Select reflector trajectories ---
    selected = select_task_ids(results_path, n_traj, tool_name, success_only, seed)

    # Diagnostic log — collects everything for post-hoc analysis
    diag_log: List[Dict] = []

    # --- Determine bootstrap source ---
    if init_playbook_path is not None:
        # Resume from a previous run's output
        with open(init_playbook_path) as f:
            init_data = json.load(f)
        bootstrap_pb = Playbook.from_dict(init_data["final_playbook"])
        bootstrap_ids = init_data["config"].get("bootstrap_task_ids", [])
        reflector_ids = selected
        print(f"Resuming from playbook: {init_playbook_path} ({len(bootstrap_pb.bullets)} bullets)")
        print(f"Reflector trajectories: {len(reflector_ids)} from {results_path}")
        diag_log.append({"step": "init_from_file", "path": init_playbook_path,
                         "num_bullets": len(bootstrap_pb.bullets)})
    elif seed_path is not None:
        # Bootstrap from a separate seed file (use ALL trajectories in it)
        with open(seed_path) as f:
            seed_data = json.load(f)
        bootstrap_ids = [r["task_id"] for r in seed_data]
        bootstrap_results_path = seed_path
        # All selected trajectories go to reflector
        reflector_ids = selected
        print(f"Bootstrap from seed file: {seed_path} ({len(bootstrap_ids)} trajectories)")
        print(f"Reflector trajectories: {len(reflector_ids)} from {results_path}")

        bootstrap_pb, bootstrap_log = bootstrap(bootstrap_results_path, bootstrap_ids, model_name)
        diag_log.append(bootstrap_log)
    else:
        # Bootstrap from first batch of selected trajectories
        bootstrap_ids = selected[:batch_size]
        bootstrap_results_path = results_path
        reflector_ids = selected[batch_size:]

        bootstrap_pb, bootstrap_log = bootstrap(bootstrap_results_path, bootstrap_ids, model_name)
        diag_log.append(bootstrap_log)

    # --- Epoch loop ---
    epoch_results = []
    current_pb = bootstrap_pb

    for epoch in range(1, num_epochs + 1):
        print(f"\n{'='*60}")
        print(f"EPOCH {epoch}/{num_epochs}")
        print(f"{'='*60}")
        print(f"Starting playbook: {len(current_pb.bullets)} bullets")

        # Shuffle reflector IDs each epoch (different ordering)
        if epoch > 1 and seed is not None:
            random.seed(seed + epoch)
            reflector_ids = random.sample(reflector_ids, len(reflector_ids))

        current_pb, epoch_result, epoch_diag = _run_single_epoch(
            results_path=results_path,
            reflector_ids=reflector_ids,
            batch_size=batch_size,
            start_playbook=current_pb,
            model_name=model_name,
            dedup_threshold=dedup_threshold,
            skip_dedup=skip_dedup,
            epoch=epoch,
            apply_per_batch=apply_per_batch,
        )
        epoch_results.append(epoch_result)
        diag_log.extend(epoch_diag)

    final_pb = current_pb
    final_policy_text = final_pb.to_prompt_text()

    # Build iteration records for backward compatibility with evaluate_prompt.py
    iterations = []

    # Bootstrap iteration
    iterations.append({
        "batch_num": 1,
        "task_ids": bootstrap_ids,
        "step": "bootstrap",
        "playbook": bootstrap_pb.to_dict(),
        "policy": bootstrap_pb.to_prompt_text(),
    })

    # One iteration per epoch
    for er in epoch_results:
        iterations.append({
            "batch_num": er["epoch"] + 1,
            "task_ids": [],
            "step": f"epoch_{er['epoch']}",
            "playbook": er["final_playbook"],
            "policy": er["final_policy"],
        })

    result = {
        "config": {
            "pipeline": "delta_batch_refine",
            "tool_name": tool_name,
            "n_traj": n_traj,
            "batch_size": batch_size,
            "model_name": model_name,
            "results_path": results_path,
            "seed_path": seed_path,
            "init_playbook_path": init_playbook_path,
            "seed": seed,
            "num_epochs": num_epochs,
            "dedup_threshold": dedup_threshold,
            "apply_per_batch": apply_per_batch,
            "bootstrap_task_ids": bootstrap_ids,
            "selected_task_ids": selected,
        },
        "iterations": iterations,
        "epoch_results": epoch_results,
        "bootstrap_playbook": bootstrap_pb.to_dict(),
        "final_playbook": final_pb.to_dict(),
        "final_policy": final_policy_text,
        "diagnostic_log": diag_log,
    }

    # Save
    if output_path is None:
        tool_str = tool_name if tool_name else "all_tools"
        output_path = f"delta_refine_{tool_str}_{n_traj}traj_batch{batch_size}_{model_name}.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"\n[Done] Saved to {output_path}")
    print(f"Final playbook: {len(final_pb.bullets)} bullets, {len(final_policy_text)} chars")

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="ACE-style delta batch policy refinement"
    )
    parser.add_argument("--results_path", type=str, required=True,
                        help="Path to results.json with trajectories")
    parser.add_argument("--n_traj", type=int, required=True,
                        help="Number of trajectories to use")
    parser.add_argument("--tool_name", type=str, default=None,
                        help="Tool name to filter by (None for all)")
    parser.add_argument("--model_name", type=str, default="gpt-4o",
                        help="LLM model name")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Output JSON path (auto-generated if omitted)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for trajectory selection")
    parser.add_argument("--batch_size", type=int, default=5,
                        help="Trajectories per batch")
    parser.add_argument("--dedup_threshold", type=float, default=0.85,
                        help="Cosine similarity threshold for dedup")
    parser.add_argument("--skip_dedup", action="store_true",
                        help="Skip deduplication step")
    parser.add_argument("--include_failed", action="store_true",
                        help="Include failed trajectories")
    parser.add_argument("--seed_path", type=str, default=None,
                        help="Separate trajectory file for bootstrap (uses all trajectories in it). "
                             "If omitted, bootstrap uses first batch from --results_path.")
    parser.add_argument("--num_epochs", type=int, default=1,
                        help="Number of passes through the training data (default: 1)")
    parser.add_argument("--init_playbook_path", type=str, default=None,
                        help="Path to a previous delta_batch_refine output JSON to resume from. "
                             "Skips bootstrap and uses that run's final_playbook as the starting point.")
    parser.add_argument("--apply_per_batch", action="store_true",
                        help="Apply deltas after each batch instead of accumulating all and applying once at the end.")

    args = parser.parse_args()

    delta_batch_refine(
        results_path=args.results_path,
        n_traj=args.n_traj,
        tool_name=args.tool_name,
        model_name=args.model_name,
        output_path=args.output_path,
        seed=args.seed,
        success_only=not args.include_failed,
        batch_size=args.batch_size,
        dedup_threshold=args.dedup_threshold,
        skip_dedup=args.skip_dedup,
        seed_path=args.seed_path,
        num_epochs=args.num_epochs,
        init_playbook_path=args.init_playbook_path,
        apply_per_batch=args.apply_per_batch,
    )


if __name__ == "__main__":
    main()
