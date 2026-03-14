# Code Conventinos Command

Carefully perform a comprehensive code review of $ARGUMENTS.

---
description: Enforce Python coding standards for this project, including typing, linting, testing, documentation, and shinyapps.io deployment readiness.
---

# Python Coding Standards

Apply these standards to all Python code in this repository.

## Core Style

- Follow **PEP 8**.
- Use **ruff** for formatting and linting.
- Maximum line length: **88 characters**.
- Prefer clear, readable code over clever code.
- Keep modules, classes, and functions small and focused.

## Typing

- Require **type hints** for all function and method signatures.
- Avoid `Any` unless absolutely necessary and justified.
- Prefer precise standard-library and `typing` types.
- Ensure code passes **mypy** with no avoidable warnings.
- Add return type annotations for all public functions.

## Documentation

- Add docstrings for all **public functions, classes, and methods**.
- Docstrings must document:
  - purpose
  - parameters
  - return values
  - raised exceptions
- Keep docstrings accurate and update them when behavior changes.

## Error Handling

- Never use bare `except:`.
- Catch **specific exceptions** only.
- Provide meaningful, actionable error messages.
- Do not silently swallow exceptions.
- Validate inputs early and fail clearly.

## Function Design

- Enforce **single responsibility**.
- Prefer functions with **5 parameters or fewer**.
- Use helper functions or small dataclasses/config objects when needed.
- Return early to reduce nesting.
- Avoid deep indentation and long conditional chains.

## Modern Python Practices

- Use **f-strings** for string formatting.
- Prefer **list comprehensions** and **generator expressions** when they improve clarity.
- Use `is` / `is not` when comparing with `None`, `True`, or `False`.
- Use `with` statements for resource management.
- Prefer `pathlib.Path` over raw string paths.
- Prefer `enum`, `dataclass`, and standard-library utilities where they improve clarity.

## Project Quality Gates

Before considering work complete, ensure code passes:

- `ruff format .`
- `ruff check .`
- `mypy .`
- `pytest`

Do not claim code is complete if these checks have not been considered.

## Testing

- Add or update tests for all non-trivial behavior changes.
- Prefer **pytest**.
- Test happy paths, edge cases, and failure cases.
- Keep tests deterministic and fast.
- Avoid network-dependent tests unless explicitly required.

## Dependencies and Reproducibility

- Keep dependencies minimal.
- Pin or lock dependencies for reproducible deployment.
- Do not introduce a new dependency unless it provides clear value.
- Prefer well-maintained, widely used packages.

## Configuration and Secrets

- Never hardcode secrets, tokens, or credentials.
- Use environment variables or deployment platform configuration.
- Keep configuration centralized and documented.
- Provide safe defaults for local development when possible.

## Logging and Observability

- Use the `logging` module instead of `print` for application diagnostics.
- Log useful context for failures without leaking secrets.
- Keep user-facing errors concise and developer-facing logs informative.

## Shiny for Python Standards

- Keep reactive logic simple and predictable.
- Separate UI construction, reactive calculations, and data/business logic.
- Avoid duplicated reactive code.
- Keep expensive computations isolated and cached where appropriate.
- Make server logic easy to test outside the UI layer where possible.

## shinyapps.io Deployment Standards

- Ensure the app is deployable to **shinyapps.io** with a clean environment.
- Verify all required packages are declared in project dependencies.
- Avoid assumptions about local files, local services, or machine-specific paths.
- Use relative paths and package assets correctly.
- Keep startup behavior fast and deterministic.
- Fail clearly when required environment variables or external resources are missing.
- Document deployment steps and required environment variables.

## Code Review Preferences

When editing code in this project:

- preserve existing behavior unless a change is requested
- make the smallest reasonable change
- update tests and docs with code changes
- prefer explicitness over magic
- avoid broad refactors unless necessary for the task

## Output Expectations

When generating or modifying code:

- produce production-quality Python
- include type hints
- include docstrings for public interfaces
- keep functions focused
- add or update tests
- follow the tooling and deployment standards above
