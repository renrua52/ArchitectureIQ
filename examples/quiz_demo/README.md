# Bundled quiz demo

This directory contains one small, generated ArchitectureIQ question together
with the dataset tensors and three candidate results needed by the question
inspector. `tools/start_quiz.py` copies `bundle/` into the gitignored `data/`
directory only when the normal default question is not already available.

The source bundle is treated as read-only. Quiz answers are written to the runtime
copy under `data/`. Custom training settings in the inspector are kept only for the
current browser session and cleared when you switch questions.
