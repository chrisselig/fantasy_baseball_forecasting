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
  6. News
  7. Trades
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
        # Row 3 — Lineup + scoreboard
        ui.layout_columns(
            _card("Today's Lineup", ui.output_data_frame("lineup_table")),
            _card("Category Scoreboard", ui.output_ui("matchup_scoreboard_ui")),
            col_widths=[5, 7],
        ),
    ),
    # ── Tab 2: Matchup Detail ─────────────────────────────────────────────
    ui.nav_panel(
        "Matchup Detail",
        # Category breakdown
        ui.layout_columns(
            _card(
                "Category Breakdown",
                _status_legend(),
                _note(
                    "Win% is the model's probability that your running total ends"
                    " higher than your opponent's total by end of the week. "
                    "Safe Win ≥ 70% | Leading 50–70% | Toss-Up 35–50% | "
                    "Trailing 20–35% | Safe Loss < 20%. "
                    "WHIP and negative stats are scored lowest-wins."
                ),
                ui.output_ui("matchup_detail_ui"),
            ),
            col_widths=[12],
        ),
        # Advanced analytics
        ui.layout_columns(
            _card(
                "Advanced Analytics",
                _note(
                    "Match Win Probability uses the exact Poisson-binomial distribution "
                    "treating each category as an independent trial — P(cats won > 6/12). "
                    "Battle Zones colour-code all 12 categories by confidence level. "
                    "Clutch Flip Analysis shows how much your overall match win % would "
                    "change if you flipped each contested category — prioritise the "
                    "categories with the largest positive Δ."
                ),
                ui.output_ui("matchup_advanced_ui"),
            ),
            col_widths=[12],
        ),
    ),
    # ── Tab 3: Waiver Wire ────────────────────────────────────────────────
    ui.nav_panel(
        "Waiver Wire",
        # Row 1 — Recommended adds/drops
        ui.layout_columns(
            _card("Recommended Adds / Drops", ui.output_data_frame("adds_table")),
            col_widths=[12],
        ),
        # Row 2 — Free agent rankings
        ui.layout_columns(
            _card(
                "Free Agent Rankings",
                _note(
                    "Score = composite value score (0–10) weighted by your team's "
                    "category needs. Higher is better. ⬆ badge = recent MLB call-up. "
                    "Streak badge uses the last 7 days (hitters) or 10 days (pitchers) "
                    "of stats — 🔥 Hot means ≥ 2 of 4 hot conditions met."
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
            "Week-to-date stats. Streak labels: 🔥 Hot = ≥ 2 of 4 hot conditions over "
            "last 7 days (hitters) or last 10 days (pitchers). ❄️ Cold = opposite. "
            "Hitter conditions: hit streak, 7-day AVG ≥ .320, OPS ≥ .920, HR or SB in last 3 days. "
            "Pitcher conditions: WHIP < 1.00, RA9 < 2.50, K/9 > 9.0, K/BB > 3.0."
        ),
        ui.layout_columns(
            _card("Hitters", ui.output_ui("hitter_roster_ui")),
            _card("Pitchers", ui.output_ui("pitcher_roster_ui")),
            col_widths=[7, 5],
        ),
    ),
    # ── Tab 5: Transactions ───────────────────────────────────────────────
    ui.nav_panel(
        "Transactions",
        ui.layout_columns(
            _card(
                "MLB Transactions",
                _note(
                    "Recent MLB roster moves: injuries, activations, call-ups, and "
                    "demotions. Only MLB-level transactions shown."
                ),
                ui.layout_columns(
                    ui.input_select(
                        "transaction_type_filter",
                        "Transaction Type",
                        choices={
                            "All": "All Types",
                            "mlb_injury": "🩹 Injuries",
                            "mlb_activation": "✅ Activations",
                            "mlb_callup": "⬆ Call-ups",
                            "mlb_demotion": "⬇ Demotions",
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
    # ── Tab 6: News ───────────────────────────────────────────────────────
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
                    "Headlines link to the original article."
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
    # ── Tab 7: Trades ─────────────────────────────────────────────────────
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
    # ── Navbar chrome ─────────────────────────────────────────────────────
    title=ui.tags.span(
        ui.tags.span("⚾", style="margin-right:0.5rem;font-size:1.1rem;"),
        "Vladimir Guerrero Invitational Fantasy League",
    ),
    id="main_navbar",
    header=ui.output_ui("data_status_banner"),
)
