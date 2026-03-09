"""
Module 1: Market Data Engine
"""

import time
import asyncio
from functools import lru_cache
from typing import Optional
from datetime import datetime, timedelta

import yfinance as yf
import pandas as pd
import numpy as np

NSE_SUFFIX = ".NS"
BSE_SUFFIX = ".BO"

_cache: dict = {}
CACHE_TTL_SECONDS = 60  


def format_symbol(symbol: str, exchange: str = "NSE") -> str:

    symbol = symbol.upper().strip()

    index_map = {
        "NIFTY": "^NSEI",
        "NIFTY50": "^NSEI",
        "NIFTY 50": "^NSEI",
        "BANKNIFTY": "^NSEBANK",
        "BANK NIFTY": "^NSEBANK",
        "NIFTYBANK": "^NSEBANK",
        "SENSEX": "^BSESN",
        "NIFTYMIDCAP": "^NSEMDCP50",
    }
    if symbol in index_map:
        return index_map[symbol]

    suffix = NSE_SUFFIX if exchange.upper() == "NSE" else BSE_SUFFIX
    if not symbol.endswith((".NS", ".BO")):
        return symbol + suffix
    return symbol


def _get_cached(key: str):
    if key in _cache:
        ts, data = _cache[key]
        if time.time() - ts < CACHE_TTL_SECONDS:
            return data
    return None


def _set_cached(key: str, data):
    _cache[key] = (time.time(), data)


async def get_live_price(symbol: str, exchange: str = "NSE") -> dict:
  
    yf_symbol = format_symbol(symbol, exchange)
    cache_key = f"live_{yf_symbol}"

    cached = _get_cached(cache_key)
    if cached:
        return cached

    loop = asyncio.get_event_loop()
    data = await loop.run_in_executor(None, _fetch_live_price, yf_symbol)
    _set_cached(cache_key, data)
    return data


def _fetch_live_price(yf_symbol: str) -> dict:
    ticker = yf.Ticker(yf_symbol)
    info = ticker.fast_info

    try:
        price = float(info.last_price)
        prev_close = float(info.previous_close)
        change = price - prev_close
        change_pct = (change / prev_close) * 100 if prev_close else 0.0
        volume = int(info.three_month_average_volume or 0)
        high = float(info.day_high) if info.day_high else price
        low = float(info.day_low) if info.day_low else price
        open_price = float(info.open) if info.open else price
    except Exception:
        hist = ticker.history(period="2d", interval="1d")
        if hist.empty:
            raise ValueError(f"No data found for symbol: {yf_symbol}")
        row = hist.iloc[-1]
        price = float(row["Close"])
        prev_close = float(hist.iloc[-2]["Close"]) if len(hist) > 1 else price
        change = price - prev_close
        change_pct = (change / prev_close) * 100 if prev_close else 0.0
        volume = int(row["Volume"])
        high = float(row["High"])
        low = float(row["Low"])
        open_price = float(row["Open"])

    return {
        "symbol": yf_symbol,
        "price": round(price, 2),
        "change": round(change, 2),
        "change_pct": round(change_pct, 2),
        "volume": volume,
        "high": round(high, 2),
        "low": round(low, 2),
        "open": round(open_price, 2),
        "prev_close": round(prev_close, 2),
        "timestamp": datetime.now().isoformat(),
    }


async def get_historical_ohlc(
    symbol: str,
    period: str = "3mo",
    interval: str = "1d",
    exchange: str = "NSE",
) -> pd.DataFrame:

    yf_symbol = format_symbol(symbol, exchange)
    cache_key = f"hist_{yf_symbol}_{period}_{interval}"

    cached = _get_cached(cache_key)
    if cached is not None:
        return cached

    loop = asyncio.get_event_loop()
    df = await loop.run_in_executor(None, _fetch_ohlc, yf_symbol, period, interval)
    _set_cached(cache_key, df)
    return df


def _fetch_ohlc(yf_symbol: str, period: str, interval: str) -> pd.DataFrame:
    ticker = yf.Ticker(yf_symbol)
    df = ticker.history(period=period, interval=interval, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No historical data for {yf_symbol} with period={period}")
    df.index = pd.to_datetime(df.index)
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.dropna(inplace=True)
    return df


async def get_multiple_prices(symbols: list[str], exchange: str = "NSE") -> list[dict]:
    tasks = [get_live_price(s, exchange) for s in symbols]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    output = []
    for sym, res in zip(symbols, results):
        if isinstance(res, Exception):
            output.append({"symbol": sym, "error": str(res)})
        else:
            output.append(res)
    return output


NIFTY50_SYMBOLS = [
    "ADANIENT", "ADANIPORTS", "APOLLOHOSP", "ASIANPAINT", "AXISBANK",
    "BAJAJ-AUTO", "BAJFINANCE", "BAJAJFINSV", "BEL", "BPCL",
    "BHARTIARTL", "BRITANNIA", "CIPLA", "COALINDIA", "DRREDDY",
    "EICHERMOT", "ETERNAL", "GRASIM", "HCLTECH", "HDFCBANK",
    "HDFCLIFE", "HEROMOTOCO", "HINDALCO", "HINDUNILVR", "ICICIBANK",
    "ITC", "INDUSINDBK", "INFY", "JSWSTEEL", "KOTAKBANK",
    "LT", "M&M", "MARUTI", "NESTLEIND", "NTPC",
    "ONGC", "POWERGRID", "RELIANCE", "SBILIFE", "SHRIRAMFIN",
    "SBIN", "SUNPHARMA", "TCS", "TATACONSUM", "TATAMOTORS",
    "TATASTEEL", "TECHM", "TITAN", "TRENT", "ULTRACEMCO",
    "WIPRO",
]

SECTOR_MAP = {
    "IT": ["INFY", "TCS", "HCLTECH", "WIPRO", "TECHM"],
    "Banking": ["HDFCBANK", "ICICIBANK", "AXISBANK", "KOTAKBANK", "SBIN", "INDUSINDBK"],
    "FMCG": ["HINDUNILVR", "ITC", "NESTLEIND", "BRITANNIA", "TATACONSUM"],
    "Auto": ["MARUTI", "TATAMOTORS", "BAJAJ-AUTO", "HEROMOTOCO", "EICHERMOT", "M&M"],
    "Pharma": ["SUNPHARMA", "DRREDDY", "CIPLA", "APOLLOHOSP"],
    "Energy": ["RELIANCE", "ONGC", "BPCL", "NTPC", "POWERGRID", "COALINDIA"],
    "Metals": ["TATASTEEL", "JSWSTEEL", "HINDALCO"],
    "Financials": ["BAJFINANCE", "BAJAJFINSV", "HDFCLIFE", "SBILIFE", "SHRIRAMFIN"],
    "Infra": ["LT", "ADANIPORTS", "GRASIM", "ULTRACEMCO"],
    "Conglomerate": ["ADANIENT", "TRENT", "BEL", "TITAN", "ASIANPAINT"],
}
