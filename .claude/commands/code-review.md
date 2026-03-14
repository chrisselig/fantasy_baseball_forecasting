# Code Review Command

Carefully perform a comprehensive code review of $ARGUMENTS.

## Review Standards
Review the standards in code-conventions.md

## Process
1. **First**: Read the example files above to understand our design patterns, naming conventions, and code style
2. **Second**: Analyze $ARGUMENTS against these standards
3. **Third**: Create a detailed critique covering:
   - Code structure and organization
   - Adherence to PEP 8 and project conventions
   - Type hint completeness
   - Docstring quality and completeness
   - Performance considerations (especially vectorized pandas operations)
   - Security implications (SQL injection, credential exposure, etc.)
   - Maintainability concerns
   - Test coverage gaps

## Output Requirements
- Save review as `ai-code-reviews/{filename}.review.md` for each file reviewed
- Include specific line references for all issues found
- Provide concrete code suggestions for improvements
- Rate overall quality: Excellent / Good / Needs Improvement / Poor
- Estimate refactoring effort: Low / Medium / High

## Review Checklist
- [ ] Follows PEP 8; would pass `black .` and `flake8` without changes
- [ ] All function signatures and class definitions have type hints
- [ ] All public functions and classes have docstrings
- [ ] No hardcoded credentials, connection strings, or magic numbers
- [ ] Environment variables used for all secrets and config
- [ ] Pandas operations are vectorized where possible (no unnecessary row-level loops)
- [ ] Appropriate error handling at system boundaries
- [ ] Unit tests exist and cover edge cases
- [ ] No raw data files referenced or committed
- [ ] Consistent with the data flow: scrape → transform → storage → app
