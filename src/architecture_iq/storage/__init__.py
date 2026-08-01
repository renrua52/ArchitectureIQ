"""Backend storage: columnar problem-instance repository under ``backend/data/``.

The storage layer is the single interface for writing problem instances
(problems / trainers / candidates / results) and for reading them back.
The generator suite writes through this API; the evaluation suite reads
through it. See ``docs/backend-storage.md`` for the authoritative design.
"""
