# Bundled quiz demo

This directory contains a generated 60-question ArchitectureIQ snapshot (20 each
for univariate regression, multivariate regression, and bigram LM), together with
the dataset tensors and candidate results needed by the question inspector.
Its manifest identifies the current snapshot as
`release_4e752ad75ce29cebe0252cb5705880b6e346baf66c8c25fc49cb536de711084f`.
`tools/start_quiz.py` copies `bundle/` into the gitignored `data/` directory only
when the normal default question is not already available.

The source bundle is treated as read-only. Answers, proposed settings, custom-run
outcomes, and comments are first kept in the current Streamlit session; when a
feedback endpoint is configured, users can upload one comment immediately or the
complete session trace. Custom training files remain per-session temporary data
and are cleared when their question is left.
