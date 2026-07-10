# Question Inspector

Streamlit UI for inspecting ArchitectureIQ questions and trying custom training
settings. Existing question and candidate artifacts are never modified.

## Install

From the repo root:

```bash
pip install -e ".[inspector]"
```

## Run

```bash
.venv/bin/python tools/start_quiz.py
```

On a fresh clone, the launcher copies the bundled demo question into `data/`
before starting Streamlit. Existing generated questions are left unchanged.

Optional: open a specific question or run first:

```bash
.venv/bin/python tools/start_quiz.py --question-run data/datasets/univariate_regression/sym_XXXXXX/questions/run_5q_2c_XXXXXX/q_XXXXXX
.venv/bin/python tools/start_quiz.py --question-run data/datasets/univariate_regression/sym_XXXXXX/questions/run_5q_2c_XXXXXX
```

Or run Streamlit directly:

```bash
streamlit run tools/question_inspector/app.py
```

## Features

- **Question** tab — dataset panel, candidate cards, quiz flow, and file inspector
- **Prompt** tab — full rendered benchmark prompt (`prompt.txt`)
- **Sidebar** — pick from existing questions, **Next**, **Random**
- **Quiz** — click **Select** on a choice to lock in your answer; metrics and ranked results appear immediately
- **Custom settings** — while solving, choose architecture, optimizer, loss, budget,
  batch size, and seed parameters, or inherit all editable values from Choice A/B/C;
  confirm to train and add a new curve
- **After answering** — use **View** or the info button on any choice to browse files (`summary.json` included once answered)

Set **Data root** to the directory containing `datasets/` (default: `data`). Questions from dataset-scoped runs and legacy `data/questions/` appear in the sidebar dropdown.

Custom runs are isolated under
`<question>/custom_settings/<setting_id>/`. Every run receives a unique sequence id
and display name. At most two runs are retained: the newest run and the historical run
with the lowest final loss. Custom runs do not alter the choices, answer key, or score.
