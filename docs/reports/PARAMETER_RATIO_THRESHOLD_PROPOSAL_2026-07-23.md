# Parameter-ratio threshold proposal for future profiles

## Evidence and boundary

This proposal is based on the frozen 46-question release distribution in
`outputs/demo_release_integration/parameter_ratio_distribution_20260723.md`.
It is a policy for **new** profiles and new question collections only; it does
not reinterpret or edit v1, v2, v2.1, or v2.2.

The historical collection is not a target distribution. Its architecture
tracks have many large-capacity comparisons (classification v2 median 18.49x;
KAN v2.2 median 32.94x; maximum 189.73x). Those numbers demonstrate why a
new capacity policy is needed; they are not numbers to preserve.

`parameter_ratio = max(choice parameter counts) / min(choice parameter
counts)`. It is a screening aid, not a definition of fairness, significance,
or difficulty. Any exception must be recorded in question provenance.

## Proposed policy for the first easy/hard architecture profiles

| New track | normal | warning (manual review) | fail (not an ordinary question) | Rationale |
|---|---:|---:|---:|---|
| `architecture_easy` | <= 2.0x | (2.0x, 4.0x] | > 4.0x | Easy should arise from an intelligible architectural or optimization effect, not a large capacity shortcut. |
| `architecture_hard` | <= 1.5x | (1.5x, 2.0x] | > 2.0x | Keeps the existing “within 2x” hard-design intent, while making closer capacity matching the default. |
| KAN-vs-MLP ordinary comparison | <= 2.0x | (2.0x, 3.0x] | > 3.0x | KAN grid choices make exact matching harder, but a large KAN/MLP capacity gap would turn the model family into an answer cue. |
| `capacity_diagnostic` (separate, opt-in) | n/a | n/a | n/a | Ratios above the ordinary cap are allowed only with explicit `diagnostic_reason`, must not enter the hard bundle, and are reported separately. |

For future `mixed` tracks, add a separate table after a dedicated pilot. The
existing v1 mixed distribution combines model, optimizer and loss changes, so
using it to set an architecture-only capacity rule would be misleading.

## Suggested profile schema

Add the following optional, declarative block to the next profile (for example
`v2.3`), rather than putting the rule in a global constant:

```yaml
question_quality:
  parameter_ratio_policy:
    default: {normal_max: 2.0, warning_max: 4.0, fail_above: 4.0}
    rules:
      - {track: architecture_easy, question_type: architecture_only,
         normal_max: 2.0, warning_max: 4.0, fail_above: 4.0}
      - {track: architecture_hard, question_type: architecture_only,
         normal_max: 1.5, warning_max: 2.0, fail_above: 2.0}
      - {track: classification_kan_diagnostic, question_type: architecture_only,
         normal_max: 2.0, warning_max: 3.0, fail_above: 3.0}
  capacity_diagnostic:
    require_explicit_reason: true
    excluded_tracks: [architecture_hard]
```

The generator/audit layer should write the two counts, ratio, `log2_ratio`,
policy state (`normal`, `warning`, `fail`, or `diagnostic_exception`) and the
exception reason into the collection-side provenance. It must **not** change
the frozen candidate identifier or hash semantics of old artifacts.

## Relation to gap and GT stability

The parameter rule should run after basic candidate compatibility but before
collection assembly. It does not replace:

1. a family/loss-specific minimum significance gate;
2. seed win-rate and failure checks;
3. family × question-type gap-percentile selection for easy/hard;
4. per-family heuristic-baseline reports (larger-parameter, KAN, depth, etc.);
5. the collection-level candidate-disjoint constraint and human audit.

The numeric **gap** bands should be calibrated from a new candidate pilot in
each `family × question_type × metric` stratum. Deriving them from the already
significance-filtered 46 selected questions would cause selection bias.
