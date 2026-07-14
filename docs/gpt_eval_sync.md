# GPT Eval Sync

This document is the local/cloud coordination note for the GPT Eval subtask.
It is intentionally more operational and more fast-moving than
[`AGENTS.md`](../AGENTS.md).

As of 2026-07-14, "GPT Eval" in this repo includes three related tracks:

1. The legacy **old 60 clean questions** evaluation track.
2. The new **high-budget confirmed 15-question** blind-eval track.
3. The current **wide-v2 setting-to-loss** meta-model project.

If a request only says "GPT Eval" without further detail, treat this file as
the router first.

## Router

| Track | Canonical shorthand | Main files | Purpose |
|---|---|---|---|
| Old clean set | `old_60_clean` | `artifacts/quiz_attempt_60/`, `docs/0710_gpteval.html`, `data/meta_model_studies/setting_to_loss_60q_id_v1/` | Reproduce and compare legacy LLM, heuristic, and meta-model results on the original 60 questions. |
| New confirmed set | `high_budget_confirmed_15` | `artifacts/high_budget_public_manifest.json`, `artifacts/high_budget_private_answer_key.json`, `artifacts/high_budget_confirmation.json`, `tools/llm_eval/high_budget_eval.py`, `artifacts/high_budget_gpt54_eval/` | Blind-evaluate models on the new 15 confirmed high-budget questions. |
| Current wide meta-model project | `wide_v2` | `tools/meta_model_dataset/plan_wide_v2.json`, `tools/meta_model_dataset/README.md`, `tools/meta_model_study/wide.py`, `tools/meta_model_study/wide_run.py` | Build the larger setting-to-loss corpus and evaluate generalization beyond the old 60-question setup. |

## Current Priority: `wide_v2`

When collaborators say "the current GPT Eval project", default to the
`wide_v2` setting-to-loss project unless they explicitly mean the 15-question
blind eval.

### Count contract

Use the following wording consistently:

- **15 dataset instances × 2 environments each = 30 environments**.
- **10,000 primary settings total**, not 30,000.
- **510 predeclared reserve settings total**.
- Per environment, the frozen plan usually selects **333 or 334 primary
  settings**, not 1,000.
- The locked split is frozen **before GT**:
  `300 train + 33/34 validation` per environment.

If someone says "30 environments with 1,000 settings each", treat that as a
shorthand that does **not** match the current frozen plan. The source of truth
is [`tools/meta_model_dataset/plan_wide_v2.json`](../tools/meta_model_dataset/plan_wide_v2.json).

### Phase contract

`wide_v2` is split into two declared phases:

- `b1_pilot`: 3,006 primary settings total.
- `b2_scale`: 6,994 primary settings total.

Run `b2_scale` only after the `b1_pilot` runtime and quality audit is accepted.

## `wide_v2` source of truth

| Concern | Canonical file or module | Notes |
|---|---|---|
| Frozen design | `tools/meta_model_dataset/plan_wide_v2.json` | The exact 30 environments, budgets, batches, seeds, splits, and exclusions live here. |
| Human-readable dataset contract | `tools/meta_model_dataset/README.md` | Explains the 10k-setting design and how it differs from the old 60-question track. |
| Prepare-only audit | `tools/meta_model_dataset/wide_v2_prepare_audit.md` | Documents the 2026-07-14 prepare audit. |
| Setting builder | `tools/meta_model_dataset/build.py` | Must own sampling, generated code, GT execution, and export. |
| Wide corpus loader | `tools/meta_model_study/wide.py` | Reads validated `wide_v2` exports only. |
| Wide study runner | `tools/meta_model_study/wide_run.py` | Fits and scores meta-models on frozen `wide_v2` exports. |
| Main output root | `data/meta_model/setting_to_loss_wide_v2/` | Gitignored; contains the built 30-environment corpus. |
| Study output root | `data/meta_model_studies/setting_to_loss_wide_v2/` | Gitignored; contains wide-v2 model evaluation artifacts. |

## Non-negotiable execution rules

These rules apply to both local and cloud collaborators.

1. Never create labels with a parallel training implementation.
2. Every target must follow:
   `candidate spec -> write_candidate() -> generated .py -> run_ground_truth() -> results/summary.json`.
3. Do not silently mutate `plan_wide_v2.json` or `profiles/meta_wide_v2.yaml`
   after any artifact is treated as frozen.
4. Keep `old_60_clean`, `high_budget_confirmed_15`, and `wide_v2` artifacts
   logically separate. Do not overwrite one track's outputs with another track's
   tooling.
5. When reporting scores, always say which track and which protocol produced
   them.

## Recommended local/cloud division of work

Use one of these scopes when parallelizing work:

- By phase: one side owns `b1_pilot`, the other owns `b2_scale`.
- By environment list: assign explicit `experiment_id` groups.
- By stage: one side owns `prepare/audit`, the other owns `gt/export`, then
  both consume the frozen export for `wide_run`.

Avoid splitting by "random subset of candidates" unless that subset is already
frozen by environment or phase. The plan, split, and exclusion story is much
clearer when ownership follows declared environment boundaries.

## Minimal handoff template

When either side starts or hands off work, record a short note using this
structure:

```text
Track:
Scope:
Branch:
Command:
Output root:
Inputs frozen from:
Status:
Blockers:
Next safe resume step:
```

The note can live in chat, commit message, or a temporary scratch file, but it
must mention the exact `track`, `scope`, and `output root`.

## Suggested commands for `wide_v2`

Prepare only:

```bash
.venv/bin/python -m tools.meta_model_dataset.build \
  --plan tools/meta_model_dataset/plan_wide_v2.json --stage prepare

.venv/bin/python -m tools.meta_model_dataset.audit_prepare \
  --plan tools/meta_model_dataset/plan_wide_v2.json
```

Pilot GT smoke:

```bash
.venv/bin/python -m tools.meta_model_dataset.build \
  --plan tools/meta_model_dataset/plan_wide_v2.json \
  --stage gt --phase b1_pilot --limit-per-experiment 3 --workers 8
```

Pilot full run:

```bash
.venv/bin/python -m tools.meta_model_dataset.build \
  --plan tools/meta_model_dataset/plan_wide_v2.json \
  --stage all --phase b1_pilot --workers 8
```

Wide study after exports exist:

```bash
.venv/bin/python -m tools.meta_model_study.wide_run
```

## Relationship to the old 60 and the new 15-question bundle

The tracks are related but not interchangeable:

- `old_60_clean` is the historical benchmark question set.
- `high_budget_confirmed_15` is the new prompt-only blind bundle for direct LLM
  evaluation.
- `wide_v2` is a setting-level meta-model dataset, not a 30-question benchmark.

`wide_v2` may later support new GPT-evaluable question selection, but today its
main unit is the **setting**, not the **question**.
