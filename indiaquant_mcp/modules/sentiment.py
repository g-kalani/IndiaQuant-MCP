"""
Module 5: News & Sentiment Analysis
"""

import os
import asyncio
from datetime import datetime, timedelta
from typing import Optional

import httpx

NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "")
ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY", "")

BULLISH_KEYWORDS = [
    "surge", "rally", "gain", "growth", "profit", "beat", "exceed", "bullish",
    "upgrade", "buy", "outperform", "record", "strong", "positive", "rise",
    "jump", "soar", "boom", "breakout", "momentum", "confident", "recovery",
    "acquisition", "expansion", "contract", "deal", "award", "dividend",
]

BEARISH_KEYWORDS = [
    "fall", "drop", "decline", "loss", "miss", "downgrade", "sell", "underperform",
    "weak", "negative", "crash", "plunge", "bearish", "risk", "concern", "lawsuit",
    "fraud", "scandal", "layoff", "cut", "restructure", "default", "debt",
    "penalty", "fine", "investigation", "slump", "warning", "caution",
]


def _score_headline(headline: str) -> float:

    text = headline.lower()
    bull = sum(1 for w in BULLISH_KEYWORDS if w in text)
    bear = sum(1 for w in BEARISH_KEYWORDS if w in text)
    total = bull + bear
    if total == 0:
        return 0.0
    return round((bull - bear) / total, 4)


async def _fetch_newsapi_headlines(symbol: str, company_name: Optional[str] = None) -> list[dict]:
    """Fetch headlines from NewsAPI.org."""
    if not NEWSAPI_KEY:
        return []

    query = f"{symbol} stock" if not company_name else f"{company_name} OR {symbol}"
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 10,
        "from": (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d"),
        "apiKey": NEWSAPI_KEY,
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
            articles = data.get("articles", [])
            return [
                {
                    "title": a.get("title", ""),
                    "source": a.get("source", {}).get("name", ""),
                    "published_at": a.get("publishedAt", ""),
                    "url": a.get("url", ""),
                }
                for a in articles if a.get("title")
            ]
    except Exception:
        return []


async def _fetch_alphavantage_sentiment(symbol: str) -> list[dict]:
    if not ALPHA_VANTAGE_KEY:
        return []

    url = "https://www.alphavantage.co/query"
    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": f"NSE:{symbol}",
        "apikey": ALPHA_VANTAGE_KEY,
        "limit": 10,
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
            feed = data.get("feed", [])
            results = []
            for item in feed:
                results.append({
                    "title": item.get("title", ""),
                    "source": item.get("source", ""),
                    "published_at": item.get("time_published", ""),
                    "url": item.get("url", ""),
                    "av_sentiment_score": float(item.get("overall_sentiment_score", 0)),
                    "av_sentiment_label": item.get("overall_sentiment_label", "Neutral"),
                })
            return results
    except Exception:
        return []


async def analyze_sentiment(symbol: str, company_name: Optional[str] = None) -> dict:

    news_results, av_results = await asyncio.gather(
        _fetch_newsapi_headlines(symbol, company_name),
        _fetch_alphavantage_sentiment(symbol),
        return_exceptions=True,
    )

    if isinstance(news_results, Exception):
        news_results = []
    if isinstance(av_results, Exception):
        av_results = []

    scored_headlines = []

    for article in news_results:
        score = _score_headline(article["title"])
        scored_headlines.append({
            "title": article["title"],
            "source": article["source"],
            "published_at": article["published_at"],
            "sentiment_score": score,
            "url": article.get("url", ""),
        })

    for article in av_results:
        av_score = article.get("av_sentiment_score", 0)
        our_score = (av_score - 0.5) * 2 if av_score else _score_headline(article["title"])
        scored_headlines.append({
            "title": article["title"],
            "source": article["source"],
            "published_at": article["published_at"],
            "sentiment_score": round(our_score, 4),
            "av_label": article.get("av_sentiment_label", ""),
            "url": article.get("url", ""),
        })

    if not scored_headlines:
        return {
            "symbol": symbol,
            "score": 0.0,
            "signal": "NEUTRAL",
            "confidence": 0,
            "headlines": [],
            "headline_count": 0,
            "note": "No news data available. Set NEWSAPI_KEY or ALPHA_VANTAGE_KEY environment variables.",
        }

    scores = [h["sentiment_score"] for h in scored_headlines]
    aggregate_score = round(sum(scores) / len(scores), 4)

    headline_confidence = min(len(scored_headlines) / 10, 1.0)  
    score_confidence = min(abs(aggregate_score) * 2, 1.0)
    confidence = round((headline_confidence * 0.4 + score_confidence * 0.6) * 100, 1)

    if aggregate_score > 0.1:
        signal = "BULLISH"
    elif aggregate_score < -0.1:
        signal = "BEARISH"
    else:
        signal = "NEUTRAL"

    return {
        "symbol": symbol,
        "score": aggregate_score,
        "signal": signal,
        "confidence": confidence,
        "headlines": sorted(scored_headlines, key=lambda x: x["published_at"], reverse=True)[:10],
        "headline_count": len(scored_headlines),
        "bullish_count": sum(1 for s in scores if s > 0.1),
        "bearish_count": sum(1 for s in scores if s < -0.1),
        "neutral_count": sum(1 for s in scores if -0.1 <= s <= 0.1),
    }
