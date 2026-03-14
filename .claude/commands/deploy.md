# Deploy Command

Run the full pre-flight checklist and deploy the Shiny app to shinyapps.io.

---

## Step 1 — Code quality gates

Run all quality checks and fix any failures before proceeding:

```bash
ruff format .
ruff check .
mypy .
pytest
```

Do not proceed if any step fails.

---

## Step 2 — Dependency check

Verify `requirements.txt` is complete and up to date:
- All packages imported anywhere in `src/` must be present
- Check for: `shiny`, `duckdb`, `yahoo_oauth`, `requests`, `pandas`, `numpy`,
  `pybaseball`, `beautifulsoup4`, `apscheduler`, `rsconnect-python`
- Pin versions if not already pinned
- Run `pip install -r requirements.txt` in a clean environment to verify no missing deps

---

## Step 3 — Environment variables

Confirm all required environment variables are documented and set on shinyapps.io:

| Variable | Purpose |
|---|---|
| `MOTHERDUCK_TOKEN` | MotherDuck cloud DuckDB connection |
| `YAHOO_CONSUMER_KEY` | Yahoo Fantasy API OAuth key |
| `YAHOO_CONSUMER_SECRET` | Yahoo Fantasy API OAuth secret |
| `YAHOO_ACCESS_TOKEN` | Yahoo OAuth access token (pre-generated) |
| `YAHOO_REFRESH_TOKEN` | Yahoo OAuth refresh token |

Check that no secrets exist in code, config files, or committed `.env` files.

---

## Step 4 — App startup check

Verify the app starts cleanly in a local environment:

```bash
shiny run src/app/app.py
```

- App must load without errors
- Dashboard must render with data (or graceful empty state if DB is empty)
- No hardcoded local paths or machine-specific assumptions

---

## Step 5 — shinyapps.io deployment

Deploy using `rsconnect-python`:

```bash
rsconnect deploy shiny . \
  --name <your-shinyapps-account> \
  --title "Fantasy Baseball Intelligence" \
  --entrypoint src/app/app.py
```

If this is a first-time deploy, add the account first:
```bash
rsconnect add --account <account-name> --token <TOKEN> --secret <SECRET>
```

---

## Step 6 — Post-deploy verification

After deployment completes:
1. Open the app URL and confirm the Dashboard loads
2. Verify environment variables are set under App Settings → Vars on shinyapps.io
3. Check that MotherDuck connection succeeds (no connection error on Dashboard)
4. Confirm Yahoo API token loads without re-auth prompts

---

## Step 7 — Tag the release

```bash
git tag -a v<version> -m "Deploy: <short description>"
git push origin --tags
```

---

## Output

```
Deploy status: success / failed
App URL: https://<account>.shinyapps.io/fantasy-baseball-intelligence/
Version tagged: v<version>
Pre-flight checks passed: ruff ✓ | mypy ✓ | pytest ✓ | deps ✓ | secrets ✓
Issues found: [list any warnings or items to follow up on]
```
