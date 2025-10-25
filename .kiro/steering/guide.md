---
inclusion: always
---

## Development Workflow

1. Implement changes
2. Run `make format` before completion (applies pyupgrade, ruff format, ruff fix)
3. Use `getDiagnostics` tool to verify no errors (NEVER use bash for diagnostics)
4. Run tests only when explicitly requested or implementation is complete

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

