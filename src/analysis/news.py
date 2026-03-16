"""
news.py

Fetches and sentiment-analyses baseball news for fantasy roster players.

Source   : Google News RSS (free, no API key required)
Sentiment: NLTK VADER (offline, no API key required)

Sentiment labels
────────────────
  Good        compound ≥  0.05  (positive news — promotion, hot streak, return from IL)
  Bad         compound ≤ -0.05  (negative news — injury, demotion, slump)
  Informative -0.05 < compound < 0.05  (neutral — stats recap, trade rumour, roster move)
"""

from __future__ import annotations

import hashlib
import logging
import urllib.parse
from datetime import UTC, datetime
from typing import Any

import feedparser
import nltk
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer

logger = logging.getLogger(__name__)

_GOOGLE_NEWS_RSS = (
    "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
)

SENTIMENT_GOOD = "Good"
SENTIMENT_BAD = "Bad"
SENTIMENT_INFO = "Informative"

# Singleton SIA — lazy-initialised on first use
_sia: SentimentIntensityAnalyzer | None = None

NEWS_COLUMNS = [
    "id",
    "player_id",
    "player_name",
    "headline",
    "url",
    "source",
    "published_at",
    "sentiment_label",
    "sentiment_score",
    "fetched_at",
]


def _get_sia() -> SentimentIntensityAnalyzer:
    """Return a cached SentimentIntensityAnalyzer, downloading lexicon if needed."""
    global _sia
    if _sia is None:
        try:
            nltk.data.find("sentiment/vader_lexicon.zip")
        except LookupError:
            nltk.download("vader_lexicon", quiet=True)
        _sia = SentimentIntensityAnalyzer()
    return _sia


def analyze_sentiment(text: str) -> tuple[str, float]:
    """Return (label, compound_score) for the given text.

    Args:
        text: Headline or article snippet.

    Returns:
        Tuple of (label, compound) where label is 'Good', 'Bad', or 'Informative'
        and compound is the VADER compound score in [-1.0, 1.0].
    """
    sia = _get_sia()
    compound = float(sia.polarity_scores(text)["compound"])
    if compound >= 0.05:
        label = SENTIMENT_GOOD
    elif compound <= -0.05:
        label = SENTIMENT_BAD
    else:
        label = SENTIMENT_INFO
    return label, round(compound, 4)


def _news_id(player_id: str, headline: str) -> str:
    """Stable 16-char hash ID for deduplication."""
    return hashlib.md5(f"{player_id}|{headline}".encode()).hexdigest()[:16]  # noqa: S324


def fetch_player_news(
    player_id: str,
    player_name: str,
    max_articles: int = 5,
) -> list[dict[str, Any]]:
    """Fetch recent baseball news for one player via Google News RSS.

    Args:
        player_id: The player's ID (used for dedup hashing).
        player_name: Player's full name — used as the search query.
        max_articles: Maximum number of articles to return.

    Returns:
        List of article dicts with keys matching NEWS_COLUMNS.
    """
    query = urllib.parse.quote(f"{player_name} baseball")
    url = _GOOGLE_NEWS_RSS.format(query=query)

    try:
        feed = feedparser.parse(url)
    except Exception as exc:
        logger.warning("Failed to fetch news for %s: %s", player_name, exc)
        return []

    now = datetime.now(UTC)
    articles: list[dict[str, Any]] = []

    for entry in feed.entries[:max_articles]:
        headline = str(entry.get("title", "")).strip()
        if not headline:
            continue

        # Parse published timestamp
        published_at: datetime
        try:
            pub = entry.get("published_parsed")
            published_at = (
                datetime(pub[0], pub[1], pub[2], pub[3], pub[4], pub[5], tzinfo=UTC)
                if pub
                else now
            )
        except Exception:
            published_at = now

        # Source name
        source = ""
        src = entry.get("source")
        if src:
            source = src.get("title", "") if isinstance(src, dict) else str(src)

        label, score = analyze_sentiment(headline)

        articles.append(
            {
                "id": _news_id(player_id, headline),
                "player_id": player_id,
                "player_name": player_name,
                "headline": headline,
                "url": str(entry.get("link", "")),
                "source": source,
                "published_at": published_at,
                "sentiment_label": label,
                "sentiment_score": score,
                "fetched_at": now,
            }
        )

    return articles


def build_news_df(
    roster_df: pd.DataFrame,
    player_id_col: str = "player_id",
    player_name_col: str = "player_name",
    max_per_player: int = 5,
) -> pd.DataFrame:
    """Fetch and sentiment-score news for all players in a roster DataFrame.

    Args:
        roster_df: DataFrame with at least player_id and player_name columns.
        player_id_col: Column name holding player IDs.
        player_name_col: Column name holding player names.
        max_per_player: Maximum articles to fetch per player.

    Returns:
        DataFrame with columns matching NEWS_COLUMNS, sorted newest-first.
    """
    all_articles: list[dict[str, Any]] = []

    for _, row in roster_df.iterrows():
        pid = str(row[player_id_col])
        name = str(row[player_name_col])
        articles = fetch_player_news(pid, name, max_articles=max_per_player)
        all_articles.extend(articles)

    if not all_articles:
        return pd.DataFrame(columns=NEWS_COLUMNS)

    df = pd.DataFrame(all_articles)
    df = df.sort_values("published_at", ascending=False).reset_index(drop=True)
    return df
