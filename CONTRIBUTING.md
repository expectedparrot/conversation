# Contributing

## Development setup

```bash
python -m pip install -e ".[test]"
python -m ruff check .
python -m mypy conversation
python -m pytest
python -m build
```

Offline tests are the default. Integration tests require provider credentials and
must be selected explicitly with `python -m pytest -m integration`.

## Pull requests

Keep behavioral changes covered by offline tests. Public API or serialized-format
changes should include migration notes in the README and an entry in `CHANGELOG.md`.
All supported Python versions must pass CI before merge.

## Releases

1. Update the version in `pyproject.toml` and move pending changelog entries into a
   dated release section.
2. Run lint, tests, and `python -m build` from a clean checkout.
3. Tag the merge commit as `vX.Y.Z` and publish the wheel and source distribution.
