"""
tests/db/test_loaders_news.py

Unit tests for src/db/loaders_news.py.
"""

from __future__ import annotations

import datetime

import duckdb
import pandas as pd
import pytest

from src.analysis.news import NEWS_COLUMNS
from src.db.loaders_news import load_player_news
from src.db.schema import FACT_PLAYER_NEWS, create_all_tables


@pytest.fixture
def conn() -> duckdb.DuckDBPyConnection:
    c = duckdb.connect(":memory:")
    create_all_tables(c)
    return c


def _news_row(
    *,
    id_: str = "abc123",
    player_id: str = "p1",
    player_name: str = "Test Player",
    headline: str = "Test hits grand slam",
    label: str = "Good",
    score: float = 0.6,
) -> dict[str, object]:
    now = datetime.datetime.now(datetime.UTC)
    return {
        "id": id_,
        "player_id": player_id,
        "player_name": player_name,
        "headline": headline,
        "url": "https://example.com/a",
        "source": "Example News",
        "published_at": now,
        "sentiment_label": label,
        "sentiment_score": score,
        "fetched_at": now,
    }


class TestLoadPlayerNews:
    def test_empty_df_returns_zero(self, conn: duckdb.DuckDBPyConnection) -> None:
        assert load_player_news(conn, pd.DataFrame(columns=NEWS_COLUMNS)) == 0

    def test_none_returns_zero(self, conn: duckdb.DuckDBPyConnection) -> None:
        assert load_player_news(conn, None) == 0  # type: ignore[arg-type]

    def test_inserts_rows(self, conn: duckdb.DuckDBPyConnection) -> None:
        df = pd.DataFrame(
            [
                _news_row(id_="a1", headline="Walk-off homer"),
                _news_row(id_="a2", player_id="p2", headline="Out with hamstring"),
            ]
        )
        n = load_player_news(conn, df)
        assert n == 2
        rows = conn.execute(
            f"SELECT id, player_id FROM {FACT_PLAYER_NEWS} ORDER BY id"
        ).fetchall()
        assert rows == [("a1", "p1"), ("a2", "p2")]

    def test_upsert_replaces_on_conflict(self, conn: duckdb.DuckDBPyConnection) -> None:
        df1 = pd.DataFrame([_news_row(id_="x1", headline="Initial headline")])
        load_player_news(conn, df1)

        df2 = pd.DataFrame(
            [_news_row(id_="x1", headline="Initial headline", label="Bad", score=-0.7)]
        )
        load_player_news(conn, df2)

        row = conn.execute(
            f"SELECT sentiment_label, sentiment_score FROM {FACT_PLAYER_NEWS} "
            "WHERE id = 'x1'"
        ).fetchone()
        assert row is not None
        assert row[0] == "Bad"
        assert float(row[1]) == pytest.approx(-0.7)
        count = conn.execute(f"SELECT COUNT(*) FROM {FACT_PLAYER_NEWS}").fetchone()
        assert count is not None
        assert count[0] == 1

    def test_drops_rows_missing_id(self, conn: duckdb.DuckDBPyConnection) -> None:
        df = pd.DataFrame(
            [
                _news_row(id_="ok1"),
                {**_news_row(), "id": None},
            ]
        )
        n = load_player_news(conn, df)
        assert n == 1

    def test_missing_columns_raises(self, conn: duckdb.DuckDBPyConnection) -> None:
        bad = pd.DataFrame([{"id": "z", "player_id": "p"}])
        with pytest.raises(ValueError, match="missing columns"):
            load_player_news(conn, bad)
