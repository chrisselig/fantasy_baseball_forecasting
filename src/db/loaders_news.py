"""
loaders_news.py

MotherDuck loader for fact_player_news.

Upserts Google News RSS headlines (with VADER sentiment scores) built by
`src.analysis.news.build_news_df`. The table's primary key is `id`, a stable
MD5 hash of player_id + headline, so INSERT OR REPLACE cleanly dedupes repeat
fetches of the same article.
"""

from __future__ import annotations

import logging

import duckdb
import pandas as pd

from src.analysis.news import NEWS_COLUMNS
from src.db.schema import FACT_PLAYER_NEWS

logger = logging.getLogger(__name__)


def load_player_news(
    conn: duckdb.DuckDBPyConnection,
    news_df: pd.DataFrame,
) -> int:
    """Upsert player news rows into fact_player_news.

    Args:
        conn: Open DuckDB connection.
        news_df: DataFrame matching src.analysis.news.NEWS_COLUMNS.

    Returns:
        Number of rows upserted.
    """
    if news_df is None or news_df.empty:
        logger.info("load_player_news: nothing to load.")
        return 0

    missing = [c for c in NEWS_COLUMNS if c not in news_df.columns]
    if missing:
        raise ValueError(f"news_df missing columns: {missing}")

    insert_df = news_df[NEWS_COLUMNS].dropna(subset=["id", "player_id"]).copy()
    if insert_df.empty:
        logger.info("load_player_news: all rows dropped (missing id/player_id).")
        return 0

    conn.register("_player_news_staging", insert_df)
    try:
        conn.execute(
            f"INSERT OR REPLACE INTO {FACT_PLAYER_NEWS} "
            "SELECT * FROM _player_news_staging"
        )
    finally:
        conn.unregister("_player_news_staging")

    logger.info("load_player_news: upserted %d rows.", len(insert_df))
    return len(insert_df)
