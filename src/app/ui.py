"""
ui.py

Baseball Savant-inspired Shiny UI for the Fantasy Baseball Intelligence Platform.
Tabs: Dashboard, Matchup Detail, Waiver Wire, Roster.
"""

from __future__ import annotations

from typing import Any

from shiny import ui

# ── Shared UI helpers ──────────────────────────────────────────────────────


def _section_card(header: str, *body: Any) -> ui.Tag:
    """Card with dark navy header and dotted-teal underline (Savant style)."""
    return ui.card(ui.card_header(header), *body)


def _status_legend() -> ui.Tag:
    """Color legend row for matchup status."""
    swatch = [
        ("#2e7d32", "Safe Win"),
        ("#558b2f", "Flippable"),
        ("#e65100", "Toss-Up"),
        ("#c62828", "Safe Loss"),
    ]
    items = []
    for color, label in swatch:
        items.append(
            ui.span(
                ui.tags.span(
                    style=f"display:inline-block;width:10px;height:10px;"
                    f"background:{color};border-radius:2px;margin-right:4px;",
                ),
                ui.tags.span(label, style="margin-right:1.25rem;"),
            )
        )
    return ui.div(*items, class_="section-legend")


# ── App UI ─────────────────────────────────────────────────────────────────

app_ui = ui.page_navbar(
    # Inject stylesheet (served from src/app/www/styles.css via static_assets)
    ui.head_content(
        ui.tags.link(rel="stylesheet", href="styles.css"),
    ),
    # ── Tab 1: Dashboard ──────────────────────────────────────────────────
    ui.nav_panel(
        "Dashboard",
        # ── Summary row
        ui.layout_columns(
            _section_card(
                "Daily Summary",
                ui.layout_columns(
                    ui.value_box(
                        "Report Date",
                        ui.output_text("header_date"),
                    ),
                    ui.value_box(
                        "Matchup Week",
                        ui.output_text("header_week"),
                    ),
                    ui.value_box(
                        "IP Pace",
                        ui.output_text("header_ip_pace"),
                    ),
                    ui.div(
                        ui.input_action_button(
                            "refresh_btn",
                            "↻  Refresh",
                            class_="btn-primary",
                        ),
                        class_="d-flex align-items-center justify-content-center",
                    ),
                    col_widths=[3, 3, 3, 3],
                ),
            ),
            col_widths=[12],
        ),
        # ── Lineup + Matchup scoreboard
        ui.layout_columns(
            _section_card(
                "Today's Lineup",
                ui.output_data_frame("lineup_table"),
            ),
            _section_card(
                "Matchup Scoreboard",
                ui.output_ui("matchup_scoreboard_ui"),
            ),
            col_widths=[5, 7],
        ),
        # ── Recommended adds
        ui.layout_columns(
            _section_card(
                "Recommended Adds / Drops",
                ui.output_data_frame("adds_table"),
            ),
            col_widths=[12],
        ),
        # ── Call-up alerts
        ui.layout_columns(
            _section_card(
                "Call-Up Alerts",
                ui.output_ui("callup_alerts_ui"),
            ),
            col_widths=[12],
        ),
    ),
    # ── Tab 2: Matchup Detail ─────────────────────────────────────────────
    ui.nav_panel(
        "Matchup Detail",
        ui.layout_columns(
            _section_card(
                "Category Breakdown",
                _status_legend(),
                ui.output_ui("matchup_detail_ui"),
            ),
            col_widths=[12],
        ),
    ),
    # ── Tab 3: Waiver Wire ────────────────────────────────────────────────
    ui.nav_panel(
        "Waiver Wire",
        ui.layout_columns(
            _section_card(
                "Free Agent Rankings",
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
                    ui.div(
                        ui.input_checkbox(
                            "callup_only",
                            "Call-ups only",
                            value=False,
                        ),
                        class_="d-flex align-items-end pb-1",
                    ),
                    col_widths=[3, 3],
                ),
                ui.output_data_frame("waiver_table"),
            ),
            col_widths=[12],
        ),
    ),
    # ── Tab 4: Roster ─────────────────────────────────────────────────────
    ui.nav_panel(
        "Roster",
        ui.layout_columns(
            _section_card(
                "Current Roster — Week-to-Date Stats",
                ui.output_data_frame("roster_table"),
            ),
            col_widths=[12],
        ),
    ),
    # ── Navbar chrome ─────────────────────────────────────────────────────
    title=ui.tags.span(
        ui.tags.span("⚾", class_="brand-icon"),
        "VGI Fantasy Baseball",
    ),
    id="main_navbar",
    header=ui.output_ui("data_status_banner"),
)
