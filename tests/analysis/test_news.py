"""Tests for src/analysis/news.py"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd

from src.analysis.news import (
    NEWS_COLUMNS,
    SENTIMENT_BAD,
    SENTIMENT_GOOD,
    SENTIMENT_INFO,
    analyze_sentiment,
    build_news_df,
    fetch_player_news,
)

# ── analyze_sentiment ───────────────────────────────────────────────────────


def test_positive_headline_is_good() -> None:
    label, score = analyze_sentiment(
        "Aaron Judge crushes two home runs in dominant win"
    )
    assert label == SENTIMENT_GOOD
    assert score >= 0.05


def test_negative_headline_is_bad() -> None:
    label, score = analyze_sentiment(
        "Player placed on 60-day IL with torn UCL, season over"
    )
    assert label == SENTIMENT_BAD
    assert score <= -0.05


def test_neutral_headline_is_informative() -> None:
    label, score = analyze_sentiment("Player discusses upcoming series schedule")
    assert label == SENTIMENT_INFO
    assert -0.05 < score < 0.05


def test_analyze_sentiment_returns_tuple() -> None:
    result = analyze_sentiment("Baseball news")
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(result[0], str)
    assert isinstance(result[1], float)


def test_score_is_rounded_to_4_decimal_places() -> None:
    _, score = analyze_sentiment("Player hits a walk-off grand slam")
    assert round(score, 4) == score


# ── fetch_player_news ───────────────────────────────────────────────────────


def _mock_feed_entry(
    title: str, link: str = "https://example.com", source: str = "MLB.com"
) -> MagicMock:
    entry = MagicMock()
    entry.get.side_effect = lambda k, default=None: {
        "title": title,
        "link": link,
        "published_parsed": (2025, 6, 7, 12, 0, 0),
    }.get(k, default)
    src = MagicMock()
    src.get.return_value = source
    entry.source = src
    return entry


@patch("src.analysis.news.feedparser.parse")
def test_fetch_player_news_returns_list(mock_parse: MagicMock) -> None:
    mock_feed = MagicMock()
    mock_feed.entries = [
        _mock_feed_entry("Judge hits two homers in Yankees win"),
        _mock_feed_entry("Judge discusses his swing mechanics"),
    ]
    mock_parse.return_value = mock_feed

    articles = fetch_player_news("p1", "Aaron Judge", max_articles=5)
    assert isinstance(articles, list)
    assert len(articles) == 2


@patch("src.analysis.news.feedparser.parse")
def test_fetch_player_news_article_has_required_keys(mock_parse: MagicMock) -> None:
    mock_feed = MagicMock()
    mock_feed.entries = [_mock_feed_entry("Judge hits home run")]
    mock_parse.return_value = mock_feed

    articles = fetch_player_news("p1", "Aaron Judge")
    assert len(articles) == 1
    article = articles[0]
    for col in NEWS_COLUMNS:
        assert col in article, f"Missing key: {col}"


@patch("src.analysis.news.feedparser.parse")
def test_fetch_player_news_respects_max_articles(mock_parse: MagicMock) -> None:
    mock_feed = MagicMock()
    mock_feed.entries = [_mock_feed_entry(f"Headline {i}") for i in range(10)]
    mock_parse.return_value = mock_feed

    articles = fetch_player_news("p1", "Aaron Judge", max_articles=3)
    assert len(articles) == 3


@patch("src.analysis.news.feedparser.parse")
def test_fetch_player_news_sentiment_applied(mock_parse: MagicMock) -> None:
    mock_feed = MagicMock()
    mock_feed.entries = [_mock_feed_entry("Player placed on IL with injury")]
    mock_parse.return_value = mock_feed

    articles = fetch_player_news("p1", "Player Name")
    assert articles[0]["sentiment_label"] in (
        SENTIMENT_GOOD,
        SENTIMENT_BAD,
        SENTIMENT_INFO,
    )
    assert isinstance(articles[0]["sentiment_score"], float)


@patch("src.analysis.news.feedparser.parse")
def test_fetch_player_news_returns_empty_on_exception(mock_parse: MagicMock) -> None:
    mock_parse.side_effect = Exception("network error")
    articles = fetch_player_news("p1", "Aaron Judge")
    assert articles == []


@patch("src.analysis.news.feedparser.parse")
def test_fetch_player_news_skips_empty_headlines(mock_parse: MagicMock) -> None:
    entry_empty = MagicMock()
    entry_empty.get.side_effect = lambda k, default=None: {"title": "", "link": ""}.get(
        k, default
    )
    entry_empty.source = MagicMock()

    entry_real = _mock_feed_entry("Judge homers again")

    mock_feed = MagicMock()
    mock_feed.entries = [entry_empty, entry_real]
    mock_parse.return_value = mock_feed

    articles = fetch_player_news("p1", "Aaron Judge")
    assert len(articles) == 1
    assert articles[0]["headline"] == "Judge homers again"


# ── build_news_df ───────────────────────────────────────────────────────────


@patch("src.analysis.news.feedparser.parse")
def test_build_news_df_returns_dataframe(mock_parse: MagicMock) -> None:
    mock_feed = MagicMock()
    mock_feed.entries = [_mock_feed_entry("Player hits HR")]
    mock_parse.return_value = mock_feed

    roster = pd.DataFrame(
        [
            {"player_id": "p1", "player_name": "Aaron Judge"},
            {"player_id": "p2", "player_name": "Shohei Ohtani"},
        ]
    )
    df = build_news_df(roster)
    assert isinstance(df, pd.DataFrame)
    assert set(NEWS_COLUMNS).issubset(df.columns)


@patch("src.analysis.news.feedparser.parse")
def test_build_news_df_empty_roster_returns_empty_df(mock_parse: MagicMock) -> None:
    roster = pd.DataFrame(columns=["player_id", "player_name"])
    df = build_news_df(roster)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0


@patch("src.analysis.news.feedparser.parse")
def test_build_news_df_all_network_errors_returns_empty_df(
    mock_parse: MagicMock,
) -> None:
    mock_parse.side_effect = Exception("connection refused")
    roster = pd.DataFrame([{"player_id": "p1", "player_name": "Aaron Judge"}])
    df = build_news_df(roster)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0
