# Quiz bundle publisher

This maintainer tool publishes existing canonical ArchitectureIQ artifacts. It
does not regenerate questions, execute ground truth, or upload anything.

Publish a full question run:

```bash
python tools/publish_quiz_bundle.py \
  datasets/univariate_regression/sym_62678b/questions/run_20q_3c_b09206
```

Publish one question (a **partial run**):

```bash
python tools/publish_quiz_bundle.py \
  datasets/univariate_regression/sym_62678b/questions/run_20q_3c_b09206/q_417b4e
```

For a partial run, the source `run.json` is copied unchanged. The generated
`quiz_manifest.json` records the question IDs actually present under
`source_runs[].selected_question_ids` and sets `source_runs[].partial` to
`true`.

Use `--dry-run` to validate and print the projected manifest without touching
the target. The default target is `examples/quiz_demo/bundle`; override it with
`--target`. Refresh a manifest for an already assembled bundle with:

```bash
python tools/publish_quiz_bundle.py --refresh-manifest --target path/to/bundle
```

Publishing rejects unsafe or incomplete references, duplicate question IDs,
and different content at an existing artifact path. Replacing a published
question is intentionally unsupported.

For a formal publish, the publisher installs only the bytes from the validated
staging bundle. Each new artifact is first copied to a temporary file in its
destination directory and then installed atomically without replacing an
existing path. If copying, final bundle validation, or manifest writing raises
an exception in the publishing process, the publisher removes every artifact
and empty directory created by that attempt and restores the previous manifest.
On such a handled exception, a newly created target is removed completely.
`--dry-run` never creates or changes the target.

This is in-process exception rollback, not a crash-atomic or multi-process
transaction. Process termination, power loss, filesystem failure during
rollback, and competing publishers still require operator inspection. Run only
one publisher for a target at a time and verify the resulting manifest before
commit or deployment. Registry export and deployment remain separate release
steps.

Source arguments are absolute paths or paths relative to `--data-root`
(default: repository `data/`), so the relative form starts with `datasets/`.

## Release smoke validation

Every dry-run, publish, manifest refresh, and direct manifest build validates
the stored canonical artifacts before producing a release. This is an
integrity check; it does not regenerate a question or rerun ground truth.

The validation covers:

- artifact references, required files, safe paths, complete candidate sets,
  set counts, and dataset/candidate/question identity;
- a canonical file allowlist: runtime caches, legacy `custom_settings`, notes,
  and any other undeclared physical file are never copied, while direct
  manifest builds reject extras already present in a bundle;
- positive candidate budget fields and
  `training_steps * batch_size == total_samples_seen`, followed by agreement
  between each set budget and its candidates and between uniform or mixed
  question budgets and their selected choices;
- summary identity, including `candidate_id`, selection metric,
  `execution == "candidate_py_files"`, seed configuration and sequence,
  failed-seed count, exclusion state, and the dynamic mean/std/final metric
  fields;
- every selected choice has `failed_seeds == 0` and `excluded == false`;
  fully failed non-choice candidates use JSON `null` for aggregate mean/std,
  and all JSON rejects non-standard `NaN`/`Infinity` constants and duplicate
  object keys;
- the metric chain from `dataset_spec.json` through question evaluation and
  significance metadata to every candidate summary;
- the stored-summary winner, `correct_letter`, the winner/runner-up
  significance gap, and a finite `win_rate` in `[0, 1]`; and
- prompt smoke checks for structural GT markers, common explicit
  answer/result disclosures, and stale “best test MSE” wording on a non-MSE
  question.

`test_mse` and `test_ce` are legacy lower-is-better metrics and may omit a
direction. Every new selection metric must declare the same boolean
`higher_is_better` in both `dataset_spec.json` and
`question.json -> evaluation`. If either artifact declares a direction, both
must declare it and agree. The winner check follows that direction.

Prompt checks are deliberately conservative heuristics, not proof that a
prompt cannot leak an answer. Maintainers must still inspect rendered prompts,
especially after changing templates or metric wording, before publishing.

When replacing an entire release, publish all desired runs into a fresh target
and validate its manifest before swapping that target into the versioned bundle.
Publishing a replacement run into the existing target would append questions
instead of removing the superseded run.

## Feedback registry export

After publishing an immutable bundle, export its authoritative question and
answer registry as reviewable JSON plus an insert-only PostgreSQL data
migration:

```bash
python tools/export_feedback_registry.py \
  --bundle path/to/bundle \
  --json-output build/feedback_registry.json \
  --sql-output build/feedback_registry.sql
```

Both destinations must be outside the bundle so its release attestation remains
valid. The exporter first performs the inspector's complete runtime manifest
and artifact attestation, then independently rebuilds the publisher manifest
with full ground-truth validation. The stored and rebuilt manifests must match
exactly.

The JSON carries a content-hashed `registry_id`, release and manifest identity,
counts, canonical question metadata, the correct letter/candidate, and every
letter-to-candidate mapping. The registry identity hashes the schema, release,
counts, and question rows. `manifest_sha256` is retained as provenance but is
not part of that identity, so changing only the manifest's descriptive
`generated_at` does not invent a different answer registry. The SQL contains
one explicit-column `INSERT` for each of `feedback_quiz_releases`,
`feedback_quiz_questions`, and `feedback_quiz_choices`, wrapped only in
`begin`/`commit`; it never updates, deletes, upserts, or performs a network
import.

Use `--check` in release automation to compare both existing outputs
byte-for-byte without writing:

```bash
python tools/export_feedback_registry.py \
  --bundle path/to/bundle \
  --json-output build/feedback_registry.json \
  --sql-output build/feedback_registry.sql \
  --check
```
