# Debug API Command

Diagnose and fix an API issue with: $ARGUMENTS

---

## Step 1 — Identify the API source

Determine which API is involved:

| API | Client file | Auth method | Common failure modes |
|---|---|---|---|
| Yahoo Fantasy Sports | `src/api/yahoo_client.py` | OAuth 2.0 token | Expired token, bad league ID, wrong endpoint version |
| MLB.com transactions | `src/api/mlb_client.py` | None (public) | Rate limiting, endpoint changes, HTML vs JSON response |
| Baseball Savant / Statcast | `src/api/mlb_client.py` via `pybaseball` | None (public) | `pybaseball` cache stale, player ID mismatch |
| FanGraphs projections | `src/api/mlb_client.py` | None (scraped) | Page structure change, missing player, encoding issue |

---

## Step 2 — Reproduce the error

1. Read the full error message and traceback from $ARGUMENTS
2. Identify the exact function and line where the failure occurs
3. Check whether this is:
   - **Auth failure** (401, 403, token expired)
   - **Bad request** (400, wrong parameters or endpoint)
   - **Not found** (404, player/league ID mismatch)
   - **Rate limit** (429, too many requests)
   - **Parse error** (unexpected response format or missing field)
   - **Network error** (timeout, DNS failure)

---

## Step 3 — Yahoo-specific diagnostics

If the issue is with the Yahoo API:

- Check if the OAuth token is expired: tokens expire after 1 hour, refresh tokens last longer
- Verify the league ID matches `config/league_settings.yaml` (`league.id: 87941`)
- Confirm the Yahoo API endpoint version — Yahoo uses a versioned XML/JSON API
- Check `yahoo_oauth` library token cache file — may need to delete and re-auth
- Test with a minimal request:
  ```python
  from yahoo_oauth import OAuth2
  sc = OAuth2(None, None, from_file="oauth2.json")
  response = sc.session.get("https://fantasysports.yahooapis.com/fantasy/v2/users;use_login=1/games")
  print(response.status_code, response.text[:500])
  ```

---

## Step 4 — MLB / pybaseball diagnostics

If the issue is with MLB data:

- For `pybaseball`, try clearing the cache: `pybaseball.cache.disable()` then re-run
- Check if the MLB player ID (`mlb_id` in `dim_players`) matches the Baseball Savant ID
- For scraped sources (Rotowire, FanGraphs), inspect the raw HTML to see if the page structure changed
- Add a raw response dump to verify what's actually being returned before parsing

---

## Step 5 — MotherDuck / DuckDB diagnostics

If the API call succeeds but loading to MotherDuck fails:

- Confirm `MOTHERDUCK_TOKEN` environment variable is set
- Test connection in isolation: `duckdb.connect("md:fantasy_baseball")`
- Check if the table schema has changed and the loader columns no longer match
- Verify primary key constraints aren't being violated on upsert

---

## Step 6 — Fix and harden

After identifying the root cause:
1. Fix the immediate issue
2. Add a specific exception handler with a meaningful error message
3. Add logging at DEBUG level for request/response details
4. Add or update a test that would catch this failure in future (mock the failing response)

---

## Step 7 — Verify the fix

```bash
pytest tests/<relevant_test_file>.py -v
```

Confirm the fix resolves the original error and no new failures were introduced.

---

## Output

```
API affected: [Yahoo / MLB / Statcast / FanGraphs / MotherDuck]
Root cause: [description]
Fix applied: [file:line — description]
Error handling added: [yes / no]
Test added/updated: [file]
Verified: [yes / no]
```
