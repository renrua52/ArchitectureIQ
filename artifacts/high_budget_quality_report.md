# ArchitectureIQ high-budget question quality report

## Outcome

This production pass generated and audited a substantially larger raw pool, then
kept only questions that survived independent fresh-seed confirmation.

- Final shortlist: 15 four-choice questions.
- Types: 10 `architecture_only`, 3 `optimizer_only`, 2 `mixed`.
- Coverage: 3 dataset families and 8 dataset instances.
- Budgets: 10 questions at 81,920 samples, 3 at 40,960, and 2 at 20,480.
- Answer distribution: A/B/C/D = 5/3/4/3.
- Choice reuse: 60 placements, 56 unique candidate paths. The only repeated
  candidates are three distractors in one optimizer track; all 15 winners are unique.

The public prompt-only index is `artifacts/high_budget_public_manifest.json`.
Answers and fresh-seed statistics are separated into
`artifacts/high_budget_private_answer_key.json`. The full confirmation evidence is
`artifacts/high_budget_confirmation.json`.

## Confirmation contract and result

Every unique candidate was independently rebuilt and executed through the canonical
pipeline:

```text
candidate_spec.json -> write_candidate(temp) -> run_ground_truth(temp) -> summary
```

The confirmation used seeds 10,000 through 10,019, which do not overlap the original
GT seeds. It used 9 single-threaded CPU workers and never overwrote original results.

- Input: 22 audited questions, 83 unique candidates.
- Confirmed: 15 questions.
- Winner changes: 0.
- Rejected for fresh-seed significance: 5 questions.
- Rejected because a choice had failed seeds: 2 questions.
- Worker or validation errors: 0.
- Wall time: 45.9 minutes.
- Aggregate candidate elapsed time: 5.90 CPU-hours.
- Candidate elapsed time: median 111.6 s, mean 256.1 s, maximum 2,267.3 s.

The 15 retained questions have confirmation win-rate 1.0 except one at 0.95. Their
fresh-seed absolute gap ranges from 0.0528 to 0.4310.

## What the noisy-dataset experiment revealed

Adding label noise alone does not make a good ArchitectureIQ task.

Two sampled univariate targets looked nonlinear syntactically but simplified exactly
to affine functions:

- `sym_d6bbf5`: approximately `-1.6666 * x`.
- `sym_f5a3cd`: approximately `1.3889 + 0.5 * x`.

Neither produced a significant four-choice architecture question. This validates the
concern that unconstrained symbolic targets can collapse into either trivial formulas
or difficult-looking but semantically weak expressions.

Noise scale was also inconsistent because it was absolute rather than normalized to
signal variance. Estimated noise-variance / signal-variance ratios ranged from below
1% on some multivariate datasets to above 300% on one univariate dataset. A single
absolute `gap_min` and `fail_threshold` therefore mean different things across data.

Randomly assembled multivariate architecture choices also had severe capacity
shortcuts:

- On `mvar_fcc2d5`, “choose the largest parameter count” solved about 91% of passing
  four-choice subsets.
- On `mvar_866b4e`, it solved about 97%; “choose the widest model” was similarly strong.

The final noisy questions were selected explicitly to defeat these rules: their winner
is not the widest, deepest, largest-parameter, or largest-LayerNorm choice, and a larger
model is present as a losing distractor.

## What made the retained architecture questions better

The strongest questions use controlled comparisons rather than arbitrary random
configurations. Examples include:

- Equal-parameter residual versus non-residual models.
- A depth-width sweet spot where both a wider shallow model and a deeper narrow model
  lose.
- Transformer choices where both smaller and larger models lose to an intermediate
  configuration.
- Optimizer comparisons in which family, learning rate, and weight decay are each
  shared with at least one distractor, so no single field identifies the winner.

The raw large-budget pool was not accepted wholesale. In one 40-question architecture
run, the family-aware rule “largest regression model, smallest bigram model” achieved
77.5%. Strict manual shortcut and reuse auditing reduced that run to four core
questions plus a small reserve before fresh-seed confirmation.

## Recommended next-generation dataset gates

Future synthesis should reject semantically weak targets before candidate training.

1. **Affine-fit gate:** fit a linear baseline on a dense noiseless sample and reject
   targets with near-perfect affine R-squared.
2. **Effective-dimension gate:** measure feature-ablation or Sobol contributions and
   require more than one materially relevant input for multivariate tasks.
3. **Interaction gate:** require a minimum non-additive interaction contribution for
   tasks intended to test depth or compositional structure.
4. **Dominant-term cap:** reject formulas where one term contributes most of target
   variance; one noisy multivariate target here was 82.4% dominated by a single cosine
   term.
5. **Scale normalization:** standardize targets, define noise by SNR, and use NMSE or
   target-normalized gaps and failure thresholds.
6. **Tail/conditioning gate:** quantify protected-division clamp mass and tail energy
   so optimizer questions do not accidentally become outlier-threshold questions.

## Recommended next-generation question settings

The following settings are more directly diagnostic of architecture quality than
unconstrained symbolic formulas:

- Hierarchical composition with matched shallow and deep parameter budgets.
- Controlled high-order feature interactions plus irrelevant dimensions.
- Multi-frequency targets that expose spectral bias and aliasing.
- Piecewise or gated regimes that reward conditional computation without singularities.
- Sequence tasks with controllable memory length, rather than only first-order bigrams.
- Parameter- or FLOP-matched depth/width/residual/normalization interventions.
- Cross-budget learning-curve questions using sparse evaluation checkpoints.

For candidate generation, first screen a wide pool with fewer seeds and sparse curve
checkpoints, then spend the saved compute on 20-seed confirmation of anti-shortcut
subsets. This pass showed that evaluating the full test split after every training step
can cost more than the training forward pass itself; sparse checkpoints would buy more
useful model/data scale without weakening final GT.
