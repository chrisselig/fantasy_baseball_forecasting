"""
ui.py

4-tab Shiny UI for the Fantasy Baseball Intelligence Platform.
Tabs: Dashboard, Matchup Detail, Waiver Wire, Roster.
"""

from __future__ import annotations

from shiny import ui

app_ui = ui.page_navbar(
    # ── Tab 1: Dashboard ──────────────────────────────────────────────────────
    ui.nav_panel(
        "Dashboard",
        ui.layout_columns(
            # Header bar
            ui.card(
                ui.card_header("Daily Summary"),
                ui.layout_columns(
                    ui.value_box(
                        "Date",
                        ui.output_text("header_date"),
                    ),
                    ui.value_box(
                        "Week",
                        ui.output_text("header_week"),
                    ),
                    ui.value_box(
                        "IP Pace",
                        ui.output_text("header_ip_pace"),
                    ),
                    ui.input_action_button(
                        "refresh_btn",
                        "Refresh Data",
                        class_="btn-primary mt-3",
                    ),
                    col_widths=[3, 3, 3, 3],
                ),
            ),
            col_widths=[12],
        ),
        ui.layout_columns(
            # Left panel: Today's Lineup
            ui.card(
                ui.card_header("Today's Lineup"),
                ui.output_data_frame("lineup_table"),
            ),
            # Right panel: Matchup Scoreboard
            ui.card(
                ui.card_header("Matchup Scoreboard"),
                ui.output_data_frame("matchup_scoreboard"),
            ),
            col_widths=[5, 7],
        ),
        ui.layout_columns(
            # Bottom: Recommended Adds
            ui.card(
                ui.card_header("Recommended Adds / Drops"),
                ui.output_data_frame("adds_table"),
            ),
            col_widths=[12],
        ),
        ui.layout_columns(
            # Call-up alerts
            ui.card(
                ui.card_header("Call-up Alerts"),
                ui.output_ui("callup_alerts_ui"),
            ),
            col_widths=[12],
        ),
    ),
    # ── Tab 2: Matchup Detail ─────────────────────────────────────────────────
    ui.nav_panel(
        "Matchup Detail",
        ui.layout_columns(
            ui.card(
                ui.card_header(
                    "Full Category Breakdown — Color: "
                    "green=safe_win, yellow=flippable, orange=toss_up, red=safe_loss"
                ),
                ui.output_data_frame("matchup_detail_table"),
            ),
            col_widths=[12],
        ),
    ),
    # ── Tab 3: Waiver Wire ────────────────────────────────────────────────────
    ui.nav_panel(
        "Waiver Wire",
        ui.layout_columns(
            ui.card(
                ui.card_header("Free Agent Rankings"),
                ui.layout_columns(
                    ui.input_select(
                        "position_filter",
                        "Filter by Position",
                        choices={
                            "All": "All Positions",
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
                    ui.input_checkbox(
                        "callup_only",
                        "Show call-ups only",
                        value=False,
                    ),
                    col_widths=[4, 4],
                ),
                ui.output_data_frame("waiver_table"),
            ),
            col_widths=[12],
        ),
    ),
    # ── Tab 4: Roster ─────────────────────────────────────────────────────────
    ui.nav_panel(
        "Roster",
        ui.layout_columns(
            ui.card(
                ui.card_header("Current Roster — Season Stats"),
                ui.output_data_frame("roster_table"),
            ),
            col_widths=[12],
        ),
    ),
    title="Fantasy Baseball Intelligence",
    id="main_navbar",
)
