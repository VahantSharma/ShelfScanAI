# Contributing to ShelfScan

## Git Branch Conventions

Use the following branch naming conventions:

| Prefix | Purpose | Example |
|--------|---------|---------|
| `feature/` | New functionality | `feature/data-pipeline` |
| `feature/` | ML components | `feature/clip-faiss` |
| `docs/` | Documentation only | `docs/readme` |
| `fix/` | Bug fixes | `fix/augmentation-bbox` |
| `refactor/` | Code restructuring | `refactor/compliance-engine` |
| `experiment/` | W&B experiment branches | `experiment/yolo-lr-sweep` |

## Branch Workflow

1. Create feature branch from `main`
2. Work in atomic commits
3. Run `pre-commit` before pushing
4. Merge via PR or fast-forward

## Conventional Commits

Use conventional commit messages:

```
feat(data): add SKU110K CSV-to-YOLO conversion
fix(detection): correct NMS threshold for dense shelves
docs: add architecture diagram to README
refactor(aligner): extract common interface for ORB/SIFT
test(compliance): add unit tests for scoring function
chore: update requirements.txt
```

## Pre-commit Hooks

Hooks run automatically on `git commit`:

- **ruff-check**: Lint with auto-fix
- **ruff-format**: Format code
- **mypy**: Type checking
- **pre-commit-hooks**: File hygiene (trailing whitespace, YAML check, etc.)

Install hooks:

```bash
pre-commit install
```

Run manually:

```bash
pre-commit run --all-files
```

## Testing

Run tests before merging:

```bash
# Unit tests only
pytest tests/test_compliance.py tests/test_aligner.py -v

# Full suite (CI)
pytest -v
```

## Code Style

- **Formatter/Linter**: Ruff (replaces Black, isort, flake8)
- **Type hints**: Every function signature
- **Docstrings**: Every module and public function
- **No magic numbers**: All constants in `configs/`
- **No hardcoded paths**: Use `pathlib.Path` and config objects
