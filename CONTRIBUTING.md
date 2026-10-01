# Contributing

Open an issue describing the input, expected behavior, and observed failure before a large change.
Use Python 3.11 or 3.12 and install `pip install -e '.[dev]'`.
Run `ruff check .`, `ruff format --check .`, `pytest -q`, and `python -m build`.
Include the exact command, model digest, input frame rate, package versions, and metric reference definition when changing a solver.
Report numerical changes separately from formatting. Never silently replace targets used for evaluation.
Keep third-party licenses and provenance with any contributed data or assets. Do not commit model weights, credentials, private videos, caches, or environment directories.
