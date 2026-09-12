# Release state archive — 2026-09-12

This archive records the repository state before the active GitHub Actions workflow was reduced to a manual-only sanity workflow.

- Main commit: `893a319` (`ci: use tracked bake fixture for API smoke`)
- Previous full CI workflow: [ci-full-20260912.yml](../ci/ci-full-20260912.yml)
- Existing immutable release tags at archive time: `v1-llm-bundle`, `v1.2-llm-bundle`
- Main branch backup tag: `archive/main-before-manual-ci-20260912`
- Full CI backup tag: `archive/ci-full-20260912`

To restore the previous workflow, copy the archived file back to `.github/workflows/ci.yml`, then commit and push it on `main`.
