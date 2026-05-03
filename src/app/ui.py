"""
ui.py

Baseball Savant-inspired Shiny UI for the Vladimir Guerrero Invitational
Fantasy League.

Tabs (in order):
  1. Dashboard
  2. Matchup Detail
  3. Waiver Wire
  4. Roster
  5. Transactions
  6. Trades
  7. News
"""

from __future__ import annotations

from typing import Any

from shiny import ui

# ── Shared helpers ─────────────────────────────────────────────────────────


def _card(header: str, *body: Any) -> ui.Tag:
    """Dark-navy header card with dotted-teal underline (Savant style)."""
    return ui.card(ui.card_header(header), *body)


def _status_legend() -> ui.Tag:
    """Inline colour legend for matchup status labels."""
    swatch = [
        ("#2e7d32", "Safe Win"),
        ("#558b2f", "Leading"),
        ("#e65100", "Toss-Up / Trailing"),
        ("#c62828", "Safe Loss"),
    ]
    items = [
        ui.tags.span(
            ui.tags.span(
                style=f"display:inline-block;width:10px;height:10px;"
                f"background:{color};border-radius:2px;margin-right:4px;",
            ),
            ui.tags.span(label, style="margin-right:1.25rem;"),
        )
        for color, label in swatch
    ]
    return ui.div(*items, class_="section-legend")


def _note(text: str) -> ui.Tag:
    """Muted annotation paragraph."""
    return ui.tags.p(
        text,
        style="font-size:0.73rem;color:#4a6282;margin:0.3rem 0 0.5rem 0;",
    )


# ── App UI ─────────────────────────────────────────────────────────────────

app_ui = ui.page_navbar(
    # Inject stylesheet
    ui.head_content(
        ui.tags.link(rel="stylesheet", href="styles.css"),
    ),
    # ── Tab 1: Dashboard ──────────────────────────────────────────────────
    ui.nav_panel(
        "Dashboard",
        # Week selector
        ui.layout_columns(
            ui.input_select(
                "week_select",
                "Week",
                choices={"latest": "Latest"},
                selected="latest",
                width="120px",
            ),
            col_widths=[2],
        ),
        # Row 1 — Week summary + refresh
        ui.layout_columns(
            _card("Week at a Glance", ui.output_ui("week_summary_ui")),
            col_widths=[12],
        ),
        # Row 2 — Projected wins
        ui.layout_columns(
            _card("Matchup Outlook", ui.output_ui("projected_wins_ui")),
            col_widths=[12],
        ),
        # Row 3 — Category scoreboard (full width)
        ui.layout_columns(
            _card(
                "Category Scoreboard",
                _status_legend(),
                _note(
                    "Win% is the model's probability that your running total ends"
                    " higher than your opponent's total by end of the week. "
                    "Safe Win >= 70% | Leading 50-70% | Toss-Up 35-50% | "
                    "Trailing 20-35% | Safe Loss < 20%. "
                    "WHIP and negative stats are scored lowest-wins."
                ),
                ui.output_ui("matchup_scoreboard_ui"),
            ),
            col_widths=[12],
        ),
    ),
    # ── Tab 2: Matchup Detail ─────────────────────────────────────────────
    ui.nav_panel(
        "Matchup Detail",
        # Row 1 — Win probability + scenario analysis
        ui.layout_columns(
            _card(
                "Match Win Probability",
                _note(
                    "Poisson-binomial model treating each category as an independent "
                    "trial. Scenario analysis shows your win% under best-case "
                    "(win all toss-ups) and worst-case (lose all toss-ups)."
                ),
                ui.output_ui("match_win_prob_ui"),
            ),
            _card(
                "Batting vs Pitching Split",
                _note(
                    "Category record split by batting (7 cats) and pitching (5 cats). "
                    "Identifies which side of your roster needs the most attention."
                ),
                ui.output_ui("bat_pitch_split_ui"),
            ),
            col_widths=[7, 5],
        ),
        # Row 2 — Battle zones + IP pace
        ui.layout_columns(
            _card(
                "Battle Zones",
                _note(
                    "All 12 categories colour-coded by confidence level. "
                    "Green = safe wins, orange = contested, red = safe losses."
                ),
                ui.output_ui("battle_zones_ui"),
            ),
            _card(
                "Pitching Strategy",
                _note(
                    "Combines IP pace with pitching category analysis. "
                    "If below 21 IP, all pitching categories are forfeited. "
                    "Advises whether streaming pitchers helps or hurts rate stats."
                ),
                ui.output_ui("pitching_strategy_ui"),
            ),
            col_widths=[7, 5],
        ),
        # Row 3 — Category gap analysis (full width)
        ui.layout_columns(
            _card(
                "Category Gap Analysis",
                _note(
                    "Exact margins needed to flip each contested category. "
                    "Focus roster moves and lineup decisions on the smallest gaps."
                ),
                ui.output_ui("category_gap_ui"),
            ),
            col_widths=[12],
        ),
        # Row 4 — Clutch flip analysis (full width)
        ui.layout_columns(
            _card(
                "Clutch Flip Analysis",
                _note(
                    "Shows how much your overall match win% would change if you "
                    "flipped each contested category. Prioritise the categories "
                    "with the largest positive delta."
                ),
                ui.output_ui("clutch_flip_ui"),
            ),
            col_widths=[12],
        ),
    ),
    # ── Tab 3: Waiver Wire ────────────────────────────────────────────────
    ui.nav_panel(
        "Waiver Wire",
        # Row 1 — Recommended adds/drops
        ui.layout_columns(
            _card("Recommended Adds / Drops", ui.output_ui("adds_table")),
            col_widths=[12],
        ),
        # Row 2 — Free agent rankings
        ui.layout_columns(
            _card(
                "Free Agent Rankings",
                ui.tags.details(
                    ui.tags.summary(
                        "How is this scored? ▾",
                        style=(
                            "cursor:pointer;font-size:0.78rem;color:#1a7fa1;"
                            "font-weight:700;list-style:none;margin-bottom:6px;"
                        ),
                    ),
                    ui.tags.div(
                        ui.tags.p(
                            ui.tags.strong("Score"),
                            " is a z-score value-above-replacement rollup across all "
                            "scoring categories, weighted by your current matchup status:",
                            style="margin:4px 0;font-size:0.78rem;color:#132747;",
                        ),
                        ui.tags.ul(
                            ui.tags.li(
                                ui.tags.strong("Flippable categories"),
                                " (close matchups within ~10%) × 2.5 — these are the "
                                "categories your waiver moves can actually flip.",
                                style="font-size:0.76rem;color:#2a3f5f;",
                            ),
                            ui.tags.li(
                                ui.tags.strong("Toss-ups"),
                                " (true coin flips) × 1.5 — also high-leverage.",
                                style="font-size:0.76rem;color:#2a3f5f;",
                            ),
                            ui.tags.li(
                                ui.tags.strong("Safe wins / losses"),
                                " × 0.15 — barely counted, because no waiver move "
                                "changes the outcome.",
                                style="font-size:0.76rem;color:#2a3f5f;",
                            ),
                            style="margin:4px 0 8px 18px;",
                        ),
                        ui.tags.p(
                            "For each category: ",
                            ui.tags.code(
                                "weighted_z = ((player_per_game − drop_per_game) / "
                                "roster_sigma) × status_weight",
                                style=(
                                    "background:#eef4fa;padding:1px 5px;"
                                    "border-radius:3px;font-size:0.72rem;"
                                ),
                            ),
                            (
                                ". Rates come from season-to-date "
                                "fact_player_stats_daily. Hitters skip pitching categories "
                                "and vice versa. The drop baseline is the player the "
                                "recommender would bench for them, so score answers: "
                            ),
                            ui.tags.em(
                                "how much better is this pickup than my worst roster spot?"
                            ),
                            style="margin:4px 0;font-size:0.76rem;color:#132747;",
                        ),
                        ui.tags.p(
                            ui.tags.strong("Fit"),
                            " is the same calculation but sums only the flippable / "
                            "toss-up categories. ",
                            ui.tags.em("Score"),
                            " tells you who is objectively the best available bat or arm; ",
                            ui.tags.em("Fit"),
                            " tells you who most directly closes the gaps in your "
                            "current matchup. When they disagree, trust ",
                            ui.tags.em("Fit"),
                            " for this week's moves and ",
                            ui.tags.em("Score"),
                            " for long-term pickups.",
                            style="margin:4px 0;font-size:0.76rem;color:#132747;",
                        ),
                        ui.tags.p(
                            ui.tags.strong("⬆ badge"),
                            " = recent MLB call-up. ",
                            ui.tags.strong("Streak badge"),
                            " uses the last 7 days (hitters) or 10 days (pitchers) "
                            "of stats — 🔥 Hot means ≥ 2 of 4 hot conditions met.",
                            style="margin:4px 0;font-size:0.76rem;color:#132747;",
                        ),
                        style=(
                            "background:#f4f8fc;border:1px solid #d6e4f0;"
                            "border-radius:4px;padding:10px 14px;margin-bottom:8px;"
                        ),
                    ),
                    style="margin-bottom:8px;",
                ),
                # Filter toolbar
                ui.layout_columns(
                    ui.input_select(
                        "position_filter",
                        "Position",
                        choices={
                            "All": "All",
                            "C": "C",
                            "1B": "1B",
                            "2B": "2B",
                            "3B": "3B",
                            "SS": "SS",
                            "OF": "OF",
                            "SP": "SP",
                            "RP": "RP",
                        },
                        selected="All",
                    ),
                    ui.input_select(
                        "streak_filter",
                        "Streak",
                        choices={
                            "All": "All",
                            "🔥 Hot": "🔥 Hot Only",
                            "❄️ Cold": "❄️ Cold Only",
                        },
                        selected="All",
                    ),
                    ui.div(
                        ui.input_checkbox("callup_only", "Call-ups only", value=False),
                        class_="d-flex align-items-end pb-1",
                    ),
                    col_widths=[3, 3, 3],
                ),
                ui.output_ui("waiver_table_ui"),
            ),
            col_widths=[12],
        ),
    ),
    # ── Tab 4: Roster ─────────────────────────────────────────────────────
    ui.nav_panel(
        "Roster",
        _note(
            "Week-to-date counting stats + season-to-date advanced metrics. "
            "Hitter advanced: wOBA (computed), xwOBA, Barrel%, HardHit%, LA (launch angle), "
            "SwSp% (sweet-spot %), BatSp (bat-speed percentile). "
            "Pitcher advanced: xERA, xwOBA-against, K-BB% (computed), Brl%-against. "
            "Advanced stats come from Baseball Savant and update daily. "
            "Streak labels: 🔥 Hot = ≥ 2 of 4 hot conditions over last 7 days (hitters) "
            "or last 10 days (pitchers). ❄️ Cold = opposite."
        ),
        ui.layout_columns(
            _card("Hitters", ui.output_ui("hitter_roster_ui")),
            col_widths=[12],
        ),
        ui.layout_columns(
            _card("Pitchers", ui.output_ui("pitcher_roster_ui")),
            col_widths=[12],
        ),
    ),
    # ── Tab 5: Transactions ───────────────────────────────────────────────
    ui.nav_panel(
        "Transactions",
        ui.layout_columns(
            _card(
                "League Transactions",
                _note(
                    "Recent MLB transactions: call-ups, demotions, IL moves, "
                    "DFAs, and releases from the last 3 days, sorted by most recent."
                ),
                ui.layout_columns(
                    ui.input_select(
                        "transaction_type_filter",
                        "Transaction Type",
                        choices={
                            "All": "All Types",
                            "call_up": "⬆ Call-Ups",
                            "il_activation": "✅ IL Activated",
                            "il_placement": "🏥 Placed on IL",
                            "demotion": "⬇ Demotions",
                            "dfa": "✂️ DFA",
                            "released": "🚫 Released",
                        },
                        selected="All",
                    ),
                    col_widths=[4],
                ),
                ui.output_ui("transactions_ui"),
            ),
            col_widths=[12],
        ),
    ),
    # ── Tab 6: Trades ─────────────────────────────────────────────────────
    ui.nav_panel(
        "Trades",
        ui.layout_columns(
            _card(
                "Trade Proposals",
                _note(
                    "Algorithmically-generated trade ideas based on your current category "
                    "strengths and weaknesses. Proposals target fair value (acceptance ≥ 50%) "
                    "with a focus on improving your weakest categories. Category pills show "
                    "which scoring areas each player impacts."
                ),
                ui.output_ui("trades_ui"),
            ),
            col_widths=[12],
        ),
    ),
    # ── Tab 7: News ───────────────────────────────────────────────────────
    ui.nav_panel(
        "News",
        ui.layout_columns(
            _card(
                "Roster Player News",
                _note(
                    "Recent baseball news for your active roster players. "
                    "Sentiment is scored by VADER NLP: ✅ Good = positive news "
                    "(hot streak, return from IL), 🚨 Bad = negative news "
                    "(injury, demotion, slump), ℹ️ Informative = neutral coverage. "
                    "Click any headline to read the full article."
                ),
                ui.layout_columns(
                    ui.input_select(
                        "news_sentiment_filter",
                        "Sentiment",
                        choices={
                            "All": "All",
                            "Good": "✅ Good",
                            "Informative": "ℹ️ Informative",
                            "Bad": "🚨 Bad",
                        },
                        selected="All",
                    ),
                    ui.input_select(
                        "news_player_filter",
                        "Player",
                        choices={"All": "All Players"},
                        selected="All",
                    ),
                    col_widths=[3, 4],
                ),
                ui.output_ui("news_ui"),
            ),
            col_widths=[12],
        ),
    ),
    # ── Navbar chrome ─────────────────────────────────────────────────────
    title=ui.tags.span(
        ui.tags.span("⚾", style="margin-right:0.5rem;font-size:1.1rem;"),
        "Vladimir Guerrero Invitational Fantasy League",
    ),
    id="main_navbar",
    header=ui.output_ui("data_status_banner"),
)
