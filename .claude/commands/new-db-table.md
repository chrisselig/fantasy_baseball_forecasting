# New MotherDuck Table Command

Scaffold a new table in the MotherDuck schema for: $ARGUMENTS

## Step 1 — Understand the requirement

Before writing any SQL or Python, answer:
- What entity or fact does this table represent?
- Is this a dimension (`dim_`) or a fact table (`fact_`)?
- What is the grain (one row per what)?
- What are the natural primary key columns?
- Which existing tables does it join to (foreign keys)?
- How often is it updated (daily, weekly, on-event)?

---

## Step 2 — Design the schema

Follow these conventions:
- **Dimensions** (slowly changing reference data): prefix `dim_`, use a single `VARCHAR PRIMARY KEY`
- **Facts** (measurements, events, snapshots): prefix `fact_`, use composite primary keys
- Column naming: `snake_case`, descriptive, no abbreviations except established baseball stats (hr, sb, whip, etc.)
- Always include an `updated_at TIMESTAMP` on dimension tables
- Use `DECIMAL(p, s)` for rate stats (AVG, OPS, WHIP, etc.), `INTEGER` for counting stats
- Use `VARCHAR[]` for multi-value columns (e.g., list of positions)
- Use `JSON` only for truly variable structure (e.g., per-category score maps)
- Add inline comments on non-obvious columns

---

## Step 3 — Add the table to `src/db/schema.py`

1. Open `src/db/schema.py`
2. Add a `CREATE TABLE IF NOT EXISTS` statement for the new table
3. Add the table name to the `ALL_TABLES` list (or equivalent registry in that file)
4. Add a `drop_table` entry if one exists for rollback support

---

## Step 4 — Add a loader to `src/db/loaders.py`

Create a function with the signature:
```python
def load_<table_name>(conn: duckdb.DuckDBPyConnection, data: pd.DataFrame) -> int:
    """Load records into <table_name>. Returns row count inserted."""
```

- Use `INSERT OR REPLACE` (upsert) unless append-only is explicitly required
- Validate required columns are present before inserting
- Log the row count on success

---

## Step 5 — Write tests

Create or update `tests/db/test_<table_name>.py`:
- Test that the table is created without error
- Test that a valid DataFrame loads correctly and returns the right row count
- Test that an invalid/missing column raises a clear error
- Test upsert behavior (inserting the same primary key twice updates, not duplicates)

---

## Step 6 — Update documentation

- Add the new table to the schema section of `docs/project_description.md`
- Add the new table to the schema section of `README.md`
- Include: table name, grain, update frequency, and key columns

---

## Output

After completing all steps, summarize:
```
Table created: <table_name>
Grain: one row per [x]
Update frequency: [daily / weekly / on-event]
Schema added to: src/db/schema.py
Loader added to: src/db/loaders.py
Tests written: tests/db/test_<table_name>.py
Docs updated: docs/project_description.md, README.md
```
