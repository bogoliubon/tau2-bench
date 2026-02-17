# Experiments

## Scripts

| File | Description |
|---|---|
| `convert_human_trajectories.py` | Converts 74 tau2-bench human trajectories into tau-bench format for use with refinement scripts |
| `iterative_policy_refinement.py` | Infers agent policy from trajectories one-by-one. Each trajectory updates the policy. Supports `--use_all` for shuffled order |
| `batch_refine.py` | Same as above but feeds multiple trajectories per batch instead of one-by-one |
| `consolidate_policy.py` | Compresses a verbose learned policy to a target character count |
| `evaluate_prompt.py` | Evaluates a policy on tau2-bench tasks (runs agent + user simulator, reports success rate) |
| `utils.py` | Shared utilities (LLM calls, trajectory extraction, tool filtering) |

## Data files

| File | Description |
|---|---|
| `human_trajectories_converted.json` | 74 human trajectories converted to tau-bench format (all reward=1.0) |

---

## `evaluations/` folder contents

### Learned policies (refinement outputs — NOT evaluations)

| File | Refiner | Prompt | Final size | Notes |
|---|---|---|---|---|
| `policy_refinement_74traj_gpt-4o_with_escape.json` | gpt-4o | Old (with "No" escape) | 3,237 chars | **Failed** — said "No" 73/74 times |
| `policy_refinement_74traj_gpt-5_with_escape.json` | gpt-5 | Old (with "No" escape) | 23,737 chars | Too verbose |
| `policy_refinement_74traj_gpt-5_with_escape_consolidated_13k.json` | gpt-5 + consolidation | Post-hoc | 13,658 chars | Compressed version of above |
| `policy_refinement_74traj_gpt-4o_without_escape.json` | gpt-4o | New (no escape) | 10,668 chars | Best performing |

### Evaluations

All evaluations use **gpt-4o as the agent** and **gpt-4.1 as the user simulator** on the **test split (40 tasks)**.

| File | Policy evaluated | Trials | Avg success |
|---|---|---|---|
| `eval_policy_md_gpt-4o_test_5trials.json` | policy.md (hand-written, 6.7K chars) | 5 | 58.50% |
| `eval_gpt5_with_escape_raw_23k_gpt-4o_test_1trial.json` | gpt-5 with_escape raw (23.7K chars) | 1 | 47.50% |
| `eval_gpt5_with_escape_consolidated_13k_gpt-4o_test_5trials.json` | gpt-5 with_escape consolidated (13.7K chars) | 5 | 61.00% |
| `eval_gpt4o_without_escape_10k_gpt-4o_test_5trials.json` | gpt-4o without_escape (10.7K chars) | 5 | 62.00% |

---

## Summary of results (agent=gpt-4o, test split, 40 tasks)

| Policy | Chars | Refiner | Trials | Avg success |
|---|---|---|---|---|
| policy.md (hand-written) | 6,716 | — | 5 | 58.50% |
| Learned gpt-5, raw | 23,737 | gpt-5 | 1 | 47.50% |
| Learned gpt-5, consolidated | 13,658 | gpt-5 + consolidation | 5 | 61.00% |
| **Learned gpt-4o, new prompt** | **10,668** | **gpt-4o** | **5** | **62.00%** |
