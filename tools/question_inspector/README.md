# Question Inspector

Standalone Streamlit UI for inspecting ArchitectureIQ **output files only**. It does not import `architecture_iq`.

## Install

From the repo root:

```bash
pip install -e ".[inspector]"
```

## Run

```bash
python tools/question_inspector/run.py
```

Optional: open a specific question or run first:

```bash
python tools/question_inspector/run.py data/datasets/univariate_regression/sym_XXXXXX/questions/run_5q_2c_XXXXXX/q_XXXXXX
python tools/question_inspector/run.py data/datasets/univariate_regression/sym_XXXXXX/questions/run_5q_2c_XXXXXX
```

Or directly:

```bash
streamlit run tools/question_inspector/app.py
```

## Features

- **Question** tab — dataset panel, candidate cards, quiz flow, and file inspector
- **Prompt** tab — full rendered benchmark prompt (`prompt.txt`)
- **Sidebar** — pick from existing questions, **Next**, **Random**
- **Quiz** — click **Select** on a choice to lock in your answer; metrics and ranked results appear immediately
- **After answering** — use **View** or the info button on any choice to browse files (`summary.json` included once answered)

Set **Data root** to the directory containing `datasets/` (default: `data`). Questions from dataset-scoped runs and legacy `data/questions/` appear in the sidebar dropdown.
