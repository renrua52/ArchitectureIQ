# ArchitectureIQ Static Quiz Exporter

Build a portable offline quiz from the generated `data/` artifacts.

The output is a normal static website. Recipients can unzip it and open
`index.html` directly on Windows or macOS; they do not need Python, Streamlit,
PyTorch, or this repository.

## Build

From the repository root:

```powershell
python tools/question_static_exporter/export.py --data-root data --out outputs/ArchitectureIQ-quiz --zip outputs/ArchitectureIQ-quiz.zip --overwrite
```

The default command exports all questions found under `data/`.

Useful options:

- `--limit 5`: export the first five questions for a quick smoke test.
- `--exclude-code`: omit embedded `.py` source files from the inspector panel.
- `--no-zip`: write only the folder.

## Output

```text
outputs/ArchitectureIQ-quiz/
  index.html
  app.js
  style.css
  data.js
  manifest.json
  README.txt
  assets/
    *_dataset.png
    *_curves.png
outputs/ArchitectureIQ-quiz.zip
```

## Distribution Notes

- Send `outputs/ArchitectureIQ-quiz.zip`.
- Users should unzip it, then double-click `index.html`.
- The answers are embedded in local static data, so this is suitable for
  self-practice, demos, or teaching, not for hidden-answer exams.
- Rebuild the ZIP whenever `data/` changes.
