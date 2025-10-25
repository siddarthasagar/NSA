---
inclusion: always
---

## Development Workflow

1. Implement changes
2. Run `make format` before completion (applies pyupgrade, ruff format, ruff fix)
3. Use `getDiagnostics` tool to verify no errors (NEVER use bash for diagnostics)
4. Run tests only when explicitly requested or implementation is complete

## Testing Guidelines

**Running Tests:**
- `make test` - Run full test suite with coverage analysis
- `make test-quick` - Run fast integration tests only
- `make coverage` - Generate coverage report (HTML + terminal)
- `make coverage-html` - Open HTML coverage report in browser

**Test Organization:**
- All tests live in `tests/` directory at project root
- Integration tests verify end-to-end workflows (data generation, training, evaluation)
- Tests use minimal configurations (small datasets, few epochs) for speed
- Shared fixtures and configuration in `tests/conftest.py`

**Coverage Reports:**
- HTML report: `htmlcov/index.html` (detailed line-by-line coverage)
- Terminal report: Shows coverage percentages and missing lines
- Configuration: `.coveragerc` excludes tests, migrations, and virtual environments
- Target coverage: Core modules 60%+, ML modules 40%+, transformations 30%+

**Interpreting Coverage:**
- Green lines: Executed during tests
- Red lines: Not executed (potential dead code)
- Yellow lines: Partially covered (branches)
- 0% coverage files: Strong candidates for removal if not intentionally unused

**Dead Code Identification Process:**
1. Run `make coverage` to generate full coverage report
2. Review HTML report in browser (`make coverage-html`)
3. Identify files with 0% coverage
4. Check if file imports PyTorch (indicates migration residue)
5. Verify file is not imported anywhere in codebase
6. Review `dead_code_candidates.md` for documented findings
7. Manual review before deletion (some files may be intentionally unused)

**PathConfig Usage:**
- Import from `utils.py`: `from utils import PathConfig`
- Always use PathConfig methods for file paths in cache directory
- Never hardcode paths to checkpoints, data files, or logs
- Call `PathConfig.ensure_cache_dirs()` at module initialization if needed

## Code Style

- **Type hints:** Required on all function signatures and complex variables
- **Ruff config:** Line length 100, double quotes, spaces (not tabs), LF line endings
- **Allowed exceptions:** F403, F405 (star imports in transformation modules only)
- **Naming:** Descriptive names, explicit over implicit
- **Debugging:** Include debug output by default until functionality is stable
- **Resolution:** All linting and type errors must be fixed before completion

## Critical Rules

- **Package manager:** ONLY `uv` - NEVER `pip` or `conda`
- **Python execution:** ALWAYS prefix with `uv run` (e.g., `uv run python -m module`, `uv run python -c "code"`)
  - NEVER use: `python`, `python3`, `./script.py`
  - ALWAYS use: `uv run python`, `uv run python -m`, `uv run python -c`
- **Commands:** ALWAYS use Makefile tasks (`make format`, `make train`, etc.)
- **Makefile targets:** Add new target if a command will be frequently reused
- **Testing:** Do NOT auto-run tests after each change (slows development)
- **Diagnostics:** Use `getDiagnostics` tool, not bash commands

