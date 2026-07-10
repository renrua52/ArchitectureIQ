# Ranking Questions

Generate calibration-plus-ranking questions from an existing trained candidate set.

Run commands from the repository root with the project environment installed.

```bash
python tools/ranking_questions/generate.py \
  data/datasets/univariate_regression/sym_62678b/candidates/set_2048_var_var_fix_696edb \
  --num-questions 12 --max-candidates 60
```

Each question shows 5 calibration settings with full learning-curve images and final mean metric, then asks for a best-to-worst ordering of 5 target settings. Human UI is written to `index.html` in the generated run directory.

For all-at-once LLM evaluation, prefer `--layout anchored`:

```bash
python tools/ranking_questions/generate.py \
  data/datasets/univariate_regression/sym_62678b/candidates/set_2048_var_var_fix_696edb \
  --layout anchored --num-questions 11 --max-candidates 60
```

The default `cyclic` layout can make 12 sliding-window questions from 60 candidates, but adjacent windows reveal a previous question's targets as the next question's calibration examples. That is useful for sequential human practice, but it leaks answers if a model sees the whole batch at once.

Score JSON answers:

```bash
python tools/ranking_questions/score_answers.py artifacts/ranking_questions/<run> answers.json
```

Create a de-identified bundle for an agent evaluation. The private answer key
must be written outside the public bundle:

```bash
python tools/ranking_questions/make_blind_bundle.py \
  artifacts/ranking_questions/<run> \
  artifacts/ranking_questions/<run>_blind \
  --answer-key-output artifacts/ranking_questions/<run>_blind_key.json
```

Generated runs under `artifacts/ranking_questions/` are gitignored. The tools
refuse to overwrite an existing run or place a private answer key inside a
public blind bundle.

Answer JSON can be either:

```json
{
  "rq_01_xxxxxx": ["T3", "T1", "T5", "T2", "T4"]
}
```

or:

```json
{
  "answers": {
    "rq_01_xxxxxx": "T3,T1,T5,T2,T4"
  }
}
```
