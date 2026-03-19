"""
Playbook data structures and utilities for ACE-style delta policy refinement.

Bullet: a single policy rule with stable ID, section, content, and provenance.
DeltaOp: an ADD/MODIFY/REMOVE operation proposed by the Reflector.
Playbook: collection of Bullets with rendering and serialization.
"""

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Any

import openai


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------

SECTIONS = [
    "information_gathering",
    "action_execution",
    "confirmation_communication",
    "precondition_rules",
]

SECTION_PREFIXES = {
    "information_gathering": "info",
    "action_execution": "acti",
    "confirmation_communication": "conf",
    "precondition_rules": "prec",
}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Bullet:
    id: str
    section: str
    content: str
    source_batch: int
    source_task_ids: List[int] = field(default_factory=list)


class OpType(str, Enum):
    ADD = "ADD"
    MODIFY = "MODIFY"
    REMOVE = "REMOVE"


@dataclass
class DeltaOp:
    op: OpType
    section: str
    content: str
    rationale: str
    source_task_ids: List[int] = field(default_factory=list)
    target_id: Optional[str] = None  # required for MODIFY/REMOVE
    batch_num: int = 0  # set during accumulation


@dataclass
class Playbook:
    bullets: List[Bullet] = field(default_factory=list)
    _counters: Dict[str, int] = field(default_factory=dict)

    # -- ID generation -----------------------------------------------------

    def _next_id(self, section: str) -> str:
        prefix = SECTION_PREFIXES.get(section, section[:4])
        count = self._counters.get(section, 0) + 1
        self._counters[section] = count
        return f"{prefix}-{count:03d}"

    def add_bullet(self, section: str, content: str, source_batch: int,
                   source_task_ids: Optional[List[int]] = None) -> Bullet:
        bid = self._next_id(section)
        b = Bullet(
            id=bid,
            section=section,
            content=content,
            source_batch=source_batch,
            source_task_ids=source_task_ids or [],
        )
        self.bullets.append(b)
        return b

    # -- Rendering ---------------------------------------------------------

    def to_prompt_text(self) -> str:
        """Render playbook as text with IDs for the LLM to reference."""
        lines = []
        for section in SECTIONS:
            section_bullets = [b for b in self.bullets if b.section == section]
            if not section_bullets:
                continue
            header = section.replace("_", " ").title()
            lines.append(f"## {header}")
            for b in section_bullets:
                lines.append(f"  [{b.id}] {b.content}")
            lines.append("")
        return "\n".join(lines)

    # -- Serialization -----------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bullets": [
                {
                    "id": b.id,
                    "section": b.section,
                    "content": b.content,
                    "source_batch": b.source_batch,
                    "source_task_ids": b.source_task_ids,
                }
                for b in self.bullets
            ],
            "_counters": dict(self._counters),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Playbook":
        pb = cls()
        pb._counters = dict(d.get("_counters", {}))
        for bd in d.get("bullets", []):
            pb.bullets.append(Bullet(
                id=bd["id"],
                section=bd["section"],
                content=bd["content"],
                source_batch=bd["source_batch"],
                source_task_ids=bd.get("source_task_ids", []),
            ))
        return pb

    def copy(self) -> "Playbook":
        return Playbook.from_dict(self.to_dict())


# ---------------------------------------------------------------------------
# JSON parsing with retry
# ---------------------------------------------------------------------------

def parse_json_response(text: str, model_name: str = "gpt-4o",
                        max_retries: int = 1) -> Any:
    """Parse JSON from LLM response, stripping markdown fences.
    Falls back to asking LLM to fix invalid JSON."""
    # Strip markdown code fences
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*\n?", "", cleaned)
    cleaned = re.sub(r"\n?```\s*$", "", cleaned)
    cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as first_err:
        if max_retries <= 0:
            raise first_err

        # Ask LLM to fix
        from utils import call_llm
        fix_prompt = (
            "The following text was supposed to be valid JSON but failed to parse. "
            "Fix it and return ONLY the corrected JSON, no explanation.\n\n"
            f"{cleaned}"
        )
        fixed = call_llm(fix_prompt, model_name)
        return parse_json_response(fixed, model_name, max_retries=0)


# ---------------------------------------------------------------------------
# Curator: apply accumulated deltas to bootstrap playbook
# ---------------------------------------------------------------------------

def _fuzzy_match_id(playbook: Playbook, target_id: str, section: str) -> Optional[str]:
    """If target_id doesn't exist, try to fuzzy-match within the section."""
    section_bullets = {b.id: b for b in playbook.bullets if b.section == section}
    if target_id in section_bullets:
        return target_id
    # Try prefix match (e.g., "info-1" -> "info-001")
    for bid in section_bullets:
        if bid.replace("-0", "-").replace("-00", "-") == target_id.replace("-0", "-").replace("-00", "-"):
            return bid
    return None


def curator_apply(bootstrap: Playbook, deltas: List[DeltaOp]) -> tuple:
    """Apply accumulated deltas to the bootstrap playbook.

    Returns (playbook, log) where log is a list of dicts describing each action taken.
    """
    pb = bootstrap.copy()
    log: List[Dict[str, Any]] = []

    # Group MODIFYs by target_id — keep only the latest (highest batch_num)
    modify_by_target: Dict[str, DeltaOp] = {}
    superseded: List[Dict[str, Any]] = []  # MODIFYs that lost to a later batch
    remove_ids: set = set()
    adds: List[DeltaOp] = []

    for d in deltas:
        if d.op == OpType.ADD:
            adds.append(d)
        elif d.op == OpType.MODIFY:
            existing = modify_by_target.get(d.target_id)
            if existing is None or d.batch_num > existing.batch_num:
                if existing is not None:
                    superseded.append({
                        "target_id": d.target_id,
                        "superseded_batch": existing.batch_num,
                        "kept_batch": d.batch_num,
                    })
                modify_by_target[d.target_id] = d
            else:
                superseded.append({
                    "target_id": d.target_id,
                    "superseded_batch": d.batch_num,
                    "kept_batch": existing.batch_num,
                })
        elif d.op == OpType.REMOVE:
            if d.target_id:
                remove_ids.add(d.target_id)

    if superseded:
        log.append({"event": "modify_conflicts", "superseded": superseded})

    # Apply MODIFYs
    for target_id, delta in modify_by_target.items():
        if target_id in remove_ids:
            log.append({"event": "modify_skipped_removed", "target_id": target_id,
                         "batch": delta.batch_num})
            continue
        matched = _fuzzy_match_id(pb, target_id, delta.section)
        if matched:
            old_content = next(b.content for b in pb.bullets if b.id == matched)
            for b in pb.bullets:
                if b.id == matched:
                    b.content = delta.content
                    b.source_task_ids = list(set(b.source_task_ids + delta.source_task_ids))
                    break
            entry = {"event": "modify_applied", "target_id": target_id, "batch": delta.batch_num,
                     "old_content": old_content, "new_content": delta.content}
            if matched != target_id:
                entry["fuzzy_matched_from"] = target_id
                entry["fuzzy_matched_to"] = matched
            log.append(entry)
        else:
            log.append({"event": "modify_failed_no_match", "target_id": target_id,
                         "section": delta.section, "batch": delta.batch_num,
                         "content": delta.content})

    # Apply REMOVEs
    for rid in remove_ids:
        found = any(b.id == rid for b in pb.bullets)
        log.append({"event": "remove_applied" if found else "remove_failed_not_found",
                     "target_id": rid})
    pb.bullets = [b for b in pb.bullets if b.id not in remove_ids]

    # Apply ADDs
    for d in adds:
        new_b = pb.add_bullet(
            section=d.section,
            content=d.content,
            source_batch=d.batch_num,
            source_task_ids=d.source_task_ids,
        )
        log.append({"event": "add_applied", "new_id": new_b.id, "section": d.section,
                     "batch": d.batch_num, "content": d.content})

    return pb, log


# ---------------------------------------------------------------------------
# Deduplication via embeddings + LLM merge/resolve
# ---------------------------------------------------------------------------

def _embed_texts(texts: List[str], model: str = "text-embedding-3-small") -> List[List[float]]:
    """Get embeddings from OpenAI."""
    client = openai.OpenAI()
    resp = client.embeddings.create(input=texts, model=model)
    return [item.embedding for item in resp.data]


def _cosine_sim(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def deduplicate_playbook(playbook: Playbook, model_name: str = "gpt-4o",
                         threshold: float = 0.85) -> tuple:
    """Embed all bullets, find near-duplicates within each section,
    ask LLM to merge or resolve contradictions.

    Returns (playbook, log) where log is a list of merge/resolve events.
    """
    from utils import call_llm

    pb = playbook.copy()
    log: List[Dict[str, Any]] = []

    for section in SECTIONS:
        section_bullets = [b for b in pb.bullets if b.section == section]
        if len(section_bullets) < 2:
            continue

        # Embed
        texts = [b.content for b in section_bullets]
        embeddings = _embed_texts(texts)

        # Log all pairwise similarities above a lower bar (0.7) for visibility
        sim_pairs = []
        for i in range(len(section_bullets)):
            for j in range(i + 1, len(section_bullets)):
                sim = _cosine_sim(embeddings[i], embeddings[j])
                if sim >= 0.7:
                    sim_pairs.append({
                        "a": section_bullets[i].id,
                        "b": section_bullets[j].id,
                        "similarity": round(sim, 4),
                        "above_threshold": sim >= threshold,
                    })
        if sim_pairs:
            log.append({"event": "similarity_pairs", "section": section, "pairs": sim_pairs})

        # Find groups above threshold
        merged_away: set = set()
        for i in range(len(section_bullets)):
            if section_bullets[i].id in merged_away:
                continue
            group = [i]
            for j in range(i + 1, len(section_bullets)):
                if section_bullets[j].id in merged_away:
                    continue
                sim = _cosine_sim(embeddings[i], embeddings[j])
                if sim >= threshold:
                    group.append(j)

            if len(group) < 2:
                continue

            # Ask LLM to merge or resolve
            group_bullets = [section_bullets[idx] for idx in group]
            rules_text = "\n".join(
                f"[{b.id}] {b.content}" for b in group_bullets
            )
            merge_prompt = (
                f"The following rules in section '{section}' are very similar.\n\n"
                f"{rules_text}\n\n"
                "Are these rules duplicates or contradictory?\n"
                "- If duplicates, merge into one clear rule.\n"
                "- If contradictory, keep the correct one based on evidence and discard the other.\n\n"
                "Return ONLY the merged/resolved rule text (no IDs, no explanation)."
            )
            merged_content = call_llm(merge_prompt, model_name)

            # Keep first bullet's ID, update content, remove others
            keeper = group_bullets[0]
            original_contents = {b.id: b.content for b in group_bullets}
            keeper.content = merged_content.strip()
            all_task_ids = set()
            for b in group_bullets:
                all_task_ids.update(b.source_task_ids)
            keeper.source_task_ids = sorted(all_task_ids)

            removed_ids = [b.id for b in group_bullets[1:]]
            for b in group_bullets[1:]:
                merged_away.add(b.id)

            log.append({
                "event": "merge",
                "section": section,
                "kept_id": keeper.id,
                "removed_ids": removed_ids,
                "original_contents": original_contents,
                "merged_content": merged_content.strip(),
            })

        # Remove merged-away bullets
        pb.bullets = [b for b in pb.bullets if b.id not in merged_away]

    return pb, log
