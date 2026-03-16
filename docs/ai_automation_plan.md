# AI-Managed Fantasy Baseball: Automation Plan

> **Goal:** A fully autonomous fantasy baseball manager that operates within a set of
> human-defined rules. This season: validate rules and automate the daily lineup.
> Next season: full autopilot — adds, drops, waiver claims, and trades.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Phase 1 — Rules Engine & Shadow Mode](#phase-1--rules-engine--shadow-mode)
3. [Phase 2 — Lineup Automation](#phase-2--lineup-automation)
4. [Phase 3 — Waiver Wire Semi-Automation](#phase-3--waiver-wire-semi-automation)
5. [Phase 4 — Full Autonomy](#phase-4--full-autonomy)
6. [This Season: Milestones](#this-season-milestones)
7. [New Files Summary](#new-files-summary)
8. [Risk Log](#risk-log)
9. [Rules Reference](#rules-reference)

---

## Architecture Overview

The system is built on three conceptual layers that never bleed into each other.

### Layer 1 — Decision Engine: *"What should I do?"*

Reads from MotherDuck (roster state, matchup standings, waiver scores, hot/cold
streaks, projections, news sentiment) and produces a ranked list of candidate actions
with scores and natural-language reasoning. This layer **never touches the Yahoo API**.
It only reads and recommends.

Already partially built: `lineup_optimizer.py`, `waiver_ranker.py`,
`matchup_analyzer.py`.

### Layer 2 — Policy Engine: *"Am I allowed to do that?"*

Applies your rules against every candidate action before anything executes.

- **Hard rules** — veto an action entirely regardless of score
- **Soft rules** — adjust scores up or down (a good player on a cold streak is still
  worth adding, just less urgently)
- **Strategic rules** — shift weights based on your current matchup position (trailing
  by 3 categories mid-week means different priorities than leading by 5)

Rules live in a human-authored YAML file (`config/ai_policy.yaml`). Changing a rule
means editing one line of config — no code change required. The file is version-controlled
so you have a full history of how your strategy evolved.

### Layer 3 — Execution Layer: *"Do it."*

Sends POST/PUT requests to the Yahoo Fantasy API. Operates in one of three modes,
configurable **per action type independently**:

| Mode | Behaviour |
|---|---|
| `shadow` | Log the decision and reasoning. Never execute. Default for everything. |
| `approval` | Send a notification with the proposed action. Execute only after explicit sign-off. |
| `autopilot` | Execute immediately if the policy score exceeds the confidence threshold. Alert if below threshold. |

Lineup can be on `autopilot` while adds/drops stay on `approval` and trades stay on
`shadow` indefinitely — you graduate each type independently when you trust it.

---

## Phase 1 — Rules Engine & Shadow Mode

**When:** Current season, starting now
**Goal:** Define and tune your rules against real data without any risk of execution

### What gets built

#### `config/ai_policy.yaml`

The single source of truth for all rules. Full annotated example:

```yaml
# ── Execution modes (per action type) ──────────────────────────────────────
mode:
  lineup:    shadow      # shadow | approval | autopilot
  adds_drops: shadow
  trades:    shadow

# ── Confidence thresholds (only relevant in autopilot mode) ────────────────
confidence:
  lineup_min_score:    0.70   # bench/start decision must be this confident
  add_drop_min_score:  0.85   # higher bar — adds are permanent within a week
  trade_min_score:     0.90   # highest bar — trades are very hard to reverse

# ── Hard rules: veto an action entirely ────────────────────────────────────
hard_rules:
  never_exceed_weekly_adds: true          # reads limit from league_settings.yaml
  never_drop_player_ranked_above: 60      # by waiver_score — protects core roster
  minimum_active_sp: 2                    # always keep 2 SPs in active slots
  minimum_active_rp: 2
  never_drop_player_on_il_within_days: 2  # might need activation window
  never_exceed_ip_pace_pct: 0.95          # stop streaming when near IP limit
  never_add_injured_player: true          # no point claiming IL-listed players
  never_bench_player_with_no_replacement: true  # don't bench if nobody better available

# ── Soft rules: score multipliers ──────────────────────────────────────────
soft_rules:
  hot_streak_add_bonus:       1.25   # 🔥 Hot player — boost waiver score 25%
  cold_streak_add_penalty:    0.70   # ❄️ Cold player — reduce waiver score 30%
  warm_streak_add_bonus:      1.10   # ☀️ Warm pitcher — slight boost
  home_starter_bonus:         1.15   # SP starting at home
  tough_matchup_sp_penalty:   0.80   # SP facing top-5 offense
  two_category_coverage_bonus: 1.10  # player helps 2+ of your weak categories
  bad_news_sentiment_penalty: 0.85   # 🚨 Bad news story in last 48 hours
  good_news_sentiment_bonus:  1.10   # ✅ Good news (return from IL, hot run)

# ── Strategic rules: shift based on matchup position ───────────────────────
strategy:
  trailing_by_2_or_more_categories:
    boost_power_categories: true      # prioritise HR, SB adds
    accept_riskier_sp: true           # tolerate higher WHIP risk for K upside
    max_category_focus: [hr, sb, k]   # concentrate on these 3

  leading_by_3_or_more_categories:
    conservative_mode: true           # protect the lead — no volatile moves
    max_adds_this_week: 2             # don't burn add slots unnecessarily
    prefer_safe_wins_over_upside: true

  final_two_days_of_matchup_week:
    bench_sp_with_bad_matchup: true   # don't give opponent stats back
    bench_sp_if_safe_win_in_k: true   # lock K category if already won
    lock_safe_win_categories: true    # don't risk a safe win with a risky play

  playoff_weeks: [22, 23, 24, 25]
    max_ip_usage_pct: 0.99            # use every available IP in playoffs
    stream_pitchers_aggressively: true
    add_budget_strategy: front_load   # use most adds early in the week

# ── Weekly add budget strategy ─────────────────────────────────────────────
add_budget:
  reserve_adds_for_emergency: 1      # always keep 1 add in reserve
  max_adds_per_day: 2                # don't blow the budget in one morning
```

#### `src/analysis/policy.py`

The rule evaluator. Public API:

```python
@dataclass
class PolicyDecision:
    action_type: str          # 'lineup_start' | 'lineup_bench' | 'add' | 'drop' | 'trade'
    candidate: dict           # the proposed action (player, slot, etc.)
    raw_score: float          # score from the decision engine before policy
    policy_score: float       # score after soft rule adjustments
    is_vetoed: bool           # True if any hard rule fired
    veto_reason: str | None   # which hard rule triggered the veto
    rules_fired: list[str]    # all rules that adjusted the score
    reasoning: str            # human-readable explanation of the decision
    confidence: float         # 0–1 confidence in the recommendation
    execution_mode: str       # the mode that was active when this ran

def evaluate(action_type, candidate, policy_cfg, matchup_state, roster_state) -> PolicyDecision
def evaluate_batch(candidates, ...) -> list[PolicyDecision]
```

Every `PolicyDecision` gets written to `fact_ai_decisions` — this is the audit trail.

#### `fact_ai_decisions` — new MotherDuck table

```sql
CREATE TABLE fact_ai_decisions (
    decision_id       VARCHAR PRIMARY KEY,  -- UUID
    decided_at        TIMESTAMP NOT NULL,
    action_type       VARCHAR NOT NULL,     -- lineup_start | add | drop | trade
    candidate_json    JSON NOT NULL,        -- full proposed action
    raw_score         DOUBLE,
    policy_score      DOUBLE,
    is_vetoed         BOOLEAN NOT NULL,
    veto_reason       VARCHAR,
    rules_fired       VARCHAR[],
    reasoning         VARCHAR NOT NULL,     -- human-readable explanation
    confidence        DOUBLE,
    execution_mode    VARCHAR NOT NULL,     -- shadow | approval | autopilot
    was_executed      BOOLEAN NOT NULL DEFAULT false,
    approved_by       VARCHAR,             -- 'human' | 'autopilot' | NULL
    outcome_notes     VARCHAR             -- filled in post-hoc for evaluation
);
```

#### `src/pipeline/decision_log.py`

Thin write layer that inserts `PolicyDecision` objects into `fact_ai_decisions`. Also
handles batch upserts (same decision_id = idempotent re-runs).

#### Shadow mode pipeline step

Added to the existing daily GitHub Actions run (`daily_pipeline.yml`). After the data
pipeline completes:

1. Load fresh roster, waiver scores, matchup state from MotherDuck
2. Run lineup optimizer → candidate lineup
3. Run each slot assignment through `policy.evaluate()`
4. Run waiver ranker → top 10 add/drop candidates
5. Run each through `policy.evaluate()`
6. Write all `PolicyDecision` objects to `fact_ai_decisions`
7. If mode is `shadow`: stop here, never call Yahoo write API

#### App: Decision Log tab

New read-only tab in the Shiny app. Shows the AI's daily decisions:

| Date | Type | Action | Score | Confidence | Rules Fired | Reasoning | Would Execute? |
|---|---|---|---|---|---|---|---|
| 2026-06-08 | Lineup | Start Judge (RF) | 0.94 | High | hot_streak_bonus | Judge is 🔥 Hot (7-day AVG .380) and faces weak SP | ✓ Yes |
| 2026-06-08 | Add | Add Julio Rodríguez, Drop Carter | 0.87 | High | two_category_bonus | Adds SB+H upside, Carter below rank 60 threshold | ✓ Yes |
| 2026-06-08 | Add | Add Bobby Miller, Drop Johnson | 0.41 | Low | tough_matchup_penalty | VETOED: Johnson ranked above hard rule threshold | ✗ Vetoed |

You review this daily. Disagreements drive rule refinements — edit the YAML, re-run
the pipeline in shadow mode, see if the decisions change.

### Success criteria to advance to Phase 2

- [ ] Two full weeks of shadow decisions reviewed
- [ ] You agree with ≥ 80% of lineup start/sit decisions
- [ ] You've added or changed at least 3 rules based on disagreements found
- [ ] Zero hard rule violations in the last 7 days of shadow run
- [ ] `fact_ai_decisions` has at least 100 rows of logged history to inspect

---

## Phase 2 — Lineup Automation

**When:** Current season, after Phase 1 criteria are met
**Goal:** AI sets your starting lineup every morning automatically

### Why lineup first

Lineup changes are the **lowest-risk** automation target:

- **Reversible** — you can override any time by logging into Yahoo directly
- **Daily feedback loop** — wrong decisions are visible within 24 hours
- **No permanent cost** — a bad bench/start decision is annoying but doesn't consume
  a weekly add slot or permanently remove a player from your roster

Adds/drops are permanent within a week. You can't un-use an add.

### What gets built

#### Yahoo write API — `src/api/yahoo_client.py`

New write methods added to the existing client:

```python
def set_lineup(
    team_key: str,
    lineup: list[LineupSlot],  # [{player_id, slot, date}]
    date: date,
    dry_run: bool = False,
) -> LineupResult:
    """PUT /fantasy/v2/team/{team_key}/roster

    Constructs XML body, executes PUT, logs raw response.
    Returns LineupResult with success flag, slots changed, and any errors.
    dry_run=True logs what would happen without executing.
    """
```

Yahoo requires XML (not JSON) for lineup changes. The body format:

```xml
<fantasy_content>
  <roster>
    <coverage_type>date</coverage_type>
    <date>2026-06-08</date>
    <players>
      <player>
        <player_key>422.p.10897</player_key>
        <position>RF</position>
      </player>
      ...
    </players>
  </roster>
</fantasy_content>
```

#### `src/pipeline/lineup_executor.py`

Full orchestration of the lineup decision and execution flow:

```
1. Check execution mode from config/ai_policy.yaml
2. Load roster state + today's schedule from MotherDuck
3. Run lineup_optimizer.py → candidate lineup with scores
4. Run each slot through policy.evaluate()
5. Apply veto logic (hard rules)
6. If mode == 'shadow':
     → log to fact_ai_decisions, send summary notification, STOP
7. If mode == 'approval':
     → send approval notification with proposed lineup
     → wait for APPROVE signal (see Approval Workflow below)
     → if approved: execute yahoo_client.set_lineup()
     → if rejected or timeout: log rejection, keep current lineup
8. If mode == 'autopilot':
     → check confidence threshold per slot
     → execute all slots above threshold
     → notify about any slots below threshold (human decides those)
     → log everything to fact_ai_decisions
9. Log execution results (success/failure per slot)
10. On any Yahoo API failure:
     → send immediate alert
     → do NOT partially apply (all-or-nothing per run)
     → keep previous lineup intact
```

#### `src/pipeline/notifier.py`

Lightweight email notifier. Uses `smtplib` + Gmail app password stored as a GitHub
Actions secret. No external service dependencies.

```python
def send_lineup_summary(decisions: list[PolicyDecision], mode: str) -> None
def send_approval_request(decisions: list[PolicyDecision], approve_url: str) -> None
def send_execution_result(result: LineupResult) -> None
def send_alert(subject: str, body: str) -> None
```

**Notification format (autopilot mode):**

```
⚾ Daily Lineup Set — June 8, 2026

STARTED                          BENCHED
────────────────────────────────────────────────────────
C   Salvador Perez               —
1B  Freddie Freeman (0.94 ✓)     —
2B  Marcus Semien (0.88 ✓)       —
3B  José Ramírez (0.91 ✓)        —
SS  Bobby Witt Jr. (0.86 ✓)      —
OF  Aaron Judge (0.97 ✓)         —
OF  Yordan Alvarez (0.93 ✓)      —
OF  Shohei Ohtani (0.95 ✓)       —
Util Kyle Tucker (0.82 ✓)        —
SP  Gerrit Cole (0.89 ✓)         —
SP  Spencer Strider (0.77 ✓)     —
RP  Josh Hader (0.91 ✓)          Emmanuel Clase (cold streak)
P   Félix Bautista (0.85 ✓)      —

Rules fired: hot_streak_bonus (Judge, Ohtani), cold_streak_penalty (Clase)

Matchup: YOU lead 7–5 heading into Day 4.
Today's focus: protect WHIP — avoid Clase's poor matchup.
```

#### Approval workflow

For `approval` mode, the notification includes a link to a GitHub Actions manual
trigger. You tap "Approve" on your phone via the GitHub mobile app and the workflow
continues. Simple, no external webhook server needed.

Approval timeout: 90 minutes. If no response by game-time threshold, the AI falls back
to the previous day's lineup (safe default) and sends an alert.

#### GitHub Actions — `lineup_automation.yml`

Separate workflow from the data pipeline. Runs at **9:00 AM MT** on days with MLB games.

```yaml
name: Daily Lineup Automation
on:
  schedule:
    - cron: '0 16 * * *'   # 9 AM MT = 16:00 UTC
  workflow_dispatch:         # manual trigger for approval responses
    inputs:
      approve_decision_id:
        description: 'Decision ID to approve (leave blank for dry run)'
        required: false
```

#### IP pace safety guard

Checked **before** every pitcher start decision, not after. Logic:

```
projected_ip_remaining = current_ip + sum(projected_ip for each active SP start remaining)
if projected_ip_remaining / ip_limit > hard_rule_never_exceed_ip_pace_pct:
    veto all additional SP starts for today
    bench any SP whose start would push over the limit
    send alert: "IP limit approaching — {n} SP starts blocked today"
```

### Safety mechanisms

- **`--dry-run` flag** always available: test any execution without touching Yahoo
- **Manual override always wins**: if you change your lineup in Yahoo, the next run
  reads fresh state and won't overwrite what you set
- **All-or-nothing execution**: if any Yahoo API call fails, the run aborts and notifies.
  Never leaves lineup in a partial state
- **Rollback log**: every lineup change is logged with the before/after state, so you
  can see exactly what the AI changed and when

### Success criteria to advance to Phase 3

- [ ] Three full weeks of autopilot lineup
- [ ] Fewer than 5 manual overrides total across those three weeks
- [ ] No hard rule violations
- [ ] No IP pace violations
- [ ] Measurable improvement in category win rate vs your pre-automation baseline

---

## Phase 3 — Waiver Wire Semi-Automation

**When:** End of current season / start of next season
**Goal:** AI submits add/drop and waiver claims, with your approval for the first month

### What gets built

#### Yahoo write API — transaction methods

```python
def add_player(
    team_key: str,
    add_player_id: str,
    drop_player_id: str | None,
    claim_type: str,            # 'add' | 'waiver'
    waiver_priority: int | None,
    dry_run: bool = False,
) -> TransactionResult

def drop_player(
    team_key: str,
    player_id: str,
    dry_run: bool = False,
) -> TransactionResult

def submit_waiver_claim(
    team_key: str,
    add_player_id: str,
    drop_player_id: str,
    priority: int,
    dry_run: bool = False,
) -> TransactionResult
```

#### `src/analysis/category_needs.py`

Computes your real-time category gap: for each of the 12 scoring categories, how far
behind are you, how many days remain, and what type of player would close the gap?

```python
@dataclass
class CategoryNeed:
    category: str
    gap: float              # how far behind opponent (negative = leading)
    days_remaining: int
    closeable: bool         # can a single add realistically close this?
    player_type_needed: str # 'power_hitter' | 'speed_hitter' | 'sp' | 'closer' | etc.
    urgency: float          # 0–1 priority score

def compute_category_needs(
    matchup_state: dict,
    days_remaining: int,
    roster: pd.DataFrame,
) -> list[CategoryNeed]
```

This feeds directly into waiver scoring weights — the AI prioritises adds that help
your most urgent categories.

#### `fact_weekly_adds` — new MotherDuck table

Tracks add budget consumption in real time:

```sql
CREATE TABLE fact_weekly_adds (
    week_number   INTEGER NOT NULL,
    season        INTEGER NOT NULL,
    adds_used     INTEGER NOT NULL DEFAULT 0,
    adds_limit    INTEGER NOT NULL,  -- from league_settings.yaml
    last_updated  TIMESTAMP NOT NULL,
    PRIMARY KEY (week_number, season)
);
```

Policy engine reads this before recommending any add. If
`adds_used + reserved_emergency_adds >= adds_limit`, all add actions are vetoed for
the rest of the week regardless of score.

#### Transaction decision engine

Runs nightly after scoring updates (approx. 11:00 PM MT):

```
1. Load fresh waiver scores, category needs, roster state
2. For each top-20 waiver candidate:
   a. Identify best drop candidate (weakest player at same position need)
   b. Compute net category impact: add_value - drop_value
   c. Run add/drop pair through policy.evaluate()
   d. Apply hard rules: positional minimums, add budget, injury check
   e. Apply soft rules: streak adjustments, news sentiment, category urgency
3. Rank surviving add/drop pairs by policy_score
4. Surface top 3 recommendations (config: max_daily_recommendations)
5. Log all to fact_ai_decisions
6. If mode == 'approval': send notification for each recommendation
7. If mode == 'autopilot': execute top recommendation if score >= threshold
```

#### Conservative defaults for Phase 3 launch

These defaults are intentionally cautious. Loosen them only after reviewing results.

```yaml
# In ai_policy.yaml — conservative Phase 3 launch settings
confidence:
  add_drop_min_score: 0.85     # very clear wins only

hard_rules:
  never_drop_player_ranked_above: 60  # protects most of your roster
  reserve_adds_for_emergency: 1

add_budget:
  max_adds_per_day: 2                 # never blow budget in one morning
  max_recommendations_per_run: 2      # surface at most 2 options per day
```

---

## Phase 4 — Full Autonomy

**When:** Next season, from opening day
**Goal:** AI manages the full team. You read the weekly summary memo.

### What gets built

#### Trade engine — `src/analysis/trade_evaluator.py`

Upgraded from the current stub to a full decision model.

**Value differential calculator:**

For any proposed trade, compute the projected rest-of-season (ROS) category impact for
both teams:

```
your_gain  = ROS_value(players_received) - ROS_value(players_given)
their_gain = ROS_value(players_given)    - ROS_value(players_received)
net_fairness = abs(your_gain - their_gain)  # lower = more balanced
```

A good trade improves your weakest categories while giving away depth in categories
where you're already winning comfortably.

**Acceptance probability model:**

Estimate how likely the opponent is to accept based on:
- Their current category weaknesses (infer from their stats and standings)
- Their roster depth at the positions you're offering
- Historical trade patterns in your specific league (stored in `fact_transactions`)
- Whether the trade is genuinely fair or obviously one-sided

The model outputs a probability: 0–100%. Trades below 50% acceptance probability are
not proposed. This keeps the AI from cluttering your opponents' inboxes with lowball
offers — which sours relationships and reduces future trade receptiveness.

**Proactive offer generator:**

Each week, identify 2–3 trade scenarios where **both sides gain in their weak categories**.
These are the only trades worth proposing. Structure: "I'm deep in K and W but weak in
SB — opponent is deep in SB but weak in K — propose a swap."

**Counter-offer handler:**

When an opponent proposes a trade:
1. Evaluate the net category impact
2. Check hard rules (would this drop below positional minimums?)
3. Run through policy engine
4. If policy_score ≥ threshold: accept
5. If policy_score is close but below threshold: generate a counter-offer that adjusts
   the value differential
6. If clearly unfavorable: decline with a flag so you can review the rejection log

#### Weekly strategy memo

Every Monday morning, the AI generates a written strategy memo stored in
`fact_ai_decisions` (type: `weekly_memo`) and displayed in the app under a new
"AI Strategy" card on the Dashboard.

```
Week 14 Strategy Memo — June 10, 2026
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MATCHUP: vs. Team Banana (currently 3rd place)
Their weaknesses: SB (rank 8), BB (rank 9)
Your weaknesses: W (rank 7), HR (rank 6)

CATEGORY FOCUS THIS WEEK
  Win:   H, SB, K, WHIP, K/BB, SV+H    (6 comfortable leads)
  Fight: HR (+3 ahead), OPS (+0.008)    (hold these, close call)
  Trail: W (-1 behind), AVG (-0.004)    (attack these)

PLANNED ACTIONS
  1. Add Marcus Stroman (SP, good matchup Mon+Fri) for David Peterson
     → +0.4 projected W, WHIP impact neutral. Executing Monday AM.
  2. Stream a second OF with SB upside if waiver wire has options Tuesday
  3. Monitor Corbin Carroll (hamstring) — if he clears, start Tues

LAST WEEK REVIEW
  Lineup decisions: 21/21 agreed with policy rules (100%)
  Add/drop decisions: 3 executed, 3/3 outperformed the dropped player
  IP pace: 17.2 / 21 through Day 5 — on track
  Wins above baseline this week: +0.8 projected cats won
```

#### Season-level strategic calendar

Rules that shift automatically with the calendar date:

| Period | Weeks | Strategy |
|---|---|---|
| Early season | 1–6 | Conservative — prioritise positional depth, limit transactions, gather projection data |
| Regular season | 7–18 | Optimise weekly — aggressive adds when matchup demands it, trade for category balance |
| Playoff bubble | 19–21 | Identify playoff bracket opponents, target their specific weaknesses |
| Playoff run | 22–25 | Maximum aggression — use every add, stream pitchers, maximise IP, protect leading categories ruthlessly |

#### Performance evaluation loop

Automated monthly report comparing:

- Your team's category win rate vs league average
- AI decision accuracy: did the recommended add actually outperform the dropped player
  over the 2 weeks after the add? (tracked via `fact_ai_decisions.outcome_notes`)
- Rule effectiveness: which rules saved or cost the most wins? Which fire most often?
- Categories where the AI is systematically wrong (signal to update YAML rules)
- Comparison: how would the team have performed if you had managed it manually (based
  on your pre-automation decisions from Phase 1 shadow data)?

---

## This Season: Milestones

| # | Milestone | Deliverables | Gate to advance |
|---|---|---|---|
| M1 | Shadow mode live | `ai_policy.yaml`, `policy.py`, `decision_log.py`, `fact_ai_decisions` table, Decision Log tab in app, shadow step in daily pipeline | Complete the build |
| M2 | Rules calibrated | Two weeks of daily shadow review; YAML updated at least 3x based on disagreements | ≥ 80% agreement on lineup decisions over last 7 days |
| M3 | Lineup approval mode | `yahoo_client.set_lineup()`, `lineup_executor.py`, `notifier.py`, `lineup_automation.yml` | One week of approved decisions, all executing correctly |
| M4 | Lineup autopilot | Switch `mode.lineup` to `autopilot` in config | < 2 manual overrides in first week of autopilot |
| M5 | Season review | Performance report, rules audit, Phase 3 plan finalised | End of season |

---

## New Files Summary

```
config/
  ai_policy.yaml                    Rules, modes, and thresholds. Human-authored,
                                    version-controlled. Edit this, not code.

src/analysis/
  policy.py                         Rule evaluator → PolicyDecision dataclass
  category_needs.py                 Per-category gap analysis and urgency ranking
  trade_evaluator.py                Full trade value model (Phase 4, upgrades stub)

src/pipeline/
  decision_log.py                   Writes PolicyDecision objects to fact_ai_decisions
  lineup_executor.py                Lineup decision → notification → execution flow
  transaction_executor.py           Add/drop/waiver decision → execution (Phase 3)
  notifier.py                       Email notifications with approval support

src/api/
  yahoo_client.py                   ADD: set_lineup(), add_player(), drop_player(),
  (existing, extend)                submit_waiver_claim(), propose_trade()

.github/workflows/
  lineup_automation.yml             Daily lineup execution (9 AM MT, separate from
                                    data pipeline)
  transaction_automation.yml        Nightly transaction evaluation (Phase 3)

docs/
  ai_automation_plan.md             This document
```

**New MotherDuck tables:**

| Table | Purpose | Phase |
|---|---|---|
| `fact_ai_decisions` | Audit trail for every decision the AI considers | 1 |
| `fact_weekly_adds` | Real-time add budget tracker | 3 |

---

## Risk Log

### Yahoo API reliability

Yahoo's Fantasy Sports API is not officially supported for third-party apps and has
changed response formats without notice in the past. The write API (POST/PUT) is
particularly sensitive to malformed request bodies.

**Mitigations:**
- Always log the raw HTTP request and response for every write call
- Wrap all calls in retry logic with exponential backoff (max 3 retries)
- On persistent failure: alert immediately, never fail silently
- Keep a local copy of the last successful lineup so you always have a fallback

### OAuth token expiration

If the Yahoo OAuth token expires during a time-sensitive automation window, lineup
changes won't execute. `token_refresh.py` already handles the refresh flow but needs
hardening for the automation context.

**Mitigations:**
- Pre-flight token validation at the start of every automation run
- If token refresh fails, send immediate alert before attempting any writes
- Separate GitHub secret for the automation token refresh so pipeline and app don't
  share the same credential state

### IP limit miscalculation

Getting the IP guard wrong in the wrong direction (under-counting projected IP) could
cause the AI to stream extra pitchers that push you over the 21-IP limit, costing the
WHIP and K/BB categories.

**Mitigations:**
- IP guard is the **first** check, not the last
- Use a conservative estimate: assume every active SP makes their scheduled start and
  every active RP pitches 1 IP
- Config threshold set to 0.95 of the limit (not 1.0) as a buffer
- Alert when projected IP crosses 90% of the limit, regardless of execution mode

### Rule conflicts

As `ai_policy.yaml` grows, rules can contradict each other (e.g. `conservative_mode:
true` from a big lead combined with a `boost_power_categories` trailing rule firing
on the same day if the config is misconfigured).

**Mitigations:**
- Policy engine has a clear precedence order: hard rules → strategic rules → soft rules
- Conflicts within a tier are logged as `CONFLICT` entries in `rules_fired`, never
  silently resolved
- YAML validation step runs at the start of every pipeline run; malformed config aborts
  the run and alerts before any decisions are made

### Over-automation before trust is established

The biggest practical risk is graduating a phase before the success criteria are
actually met — running autopilot for adds/drops before the rules are calibrated
enough to trust. This can burn your add budget on bad decisions and permanently
weaken your roster.

**Mitigation:** The shadow mode data is the antidote. Two weeks of logged decisions
with your agreement rate computed automatically is objective evidence, not gut feeling.
Don't advance phases based on feeling confident — advance based on the numbers.

---

## Rules Reference

This section documents every rule type so future rule additions follow a consistent
pattern.

### Hard rule naming convention

`never_<action>_<condition>` — always a boolean or numeric threshold.

```yaml
never_drop_player_ranked_above: 60      # numeric threshold
never_add_injured_player: true          # boolean
never_exceed_weekly_adds: true          # derived from league_settings.yaml
```

### Soft rule naming convention

`<subject>_<direction>_<multiplier_type>`: the subject, whether it's a bonus or
penalty, and what it multiplies.

```yaml
hot_streak_add_bonus: 1.25          # multiplier > 1 = bonus
cold_streak_add_penalty: 0.70       # multiplier < 1 = penalty
home_starter_bonus: 1.15
bad_news_sentiment_penalty: 0.85
```

### Strategic rule structure

Strategic rules are triggered by a named condition and contain a list of sub-rules
that override or supplement the base soft rules when that condition is true.

```yaml
strategy:
  <condition_name>:
    <sub_rule>: <value>
    ...
```

Conditions are evaluated in order. If multiple conditions are true simultaneously
(e.g., trailing AND final two days), all matching strategic rule sets are merged,
with later conditions taking precedence on conflicts.

---

*Last updated: 2026-03-15*
