"""
Module 2: AI Trade Signal Generator
"""

import asyncio
from typing import Optional
import numpy as np
import pandas as pd

try:
    import pandas_ta as ta
    HAS_TA = True
except ImportError:
    HAS_TA = False

from modules.market_data import get_historical_ohlc

def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    if HAS_TA:
        return ta.rsi(close, length=period)
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> dict:
    if HAS_TA:
        macd_df = ta.macd(close, fast=fast, slow=slow, signal=signal)
        if macd_df is not None and not macd_df.empty:
            cols = macd_df.columns.tolist()
            return {
                "macd": macd_df[cols[0]],
                "signal": macd_df[cols[2]],
                "histogram": macd_df[cols[1]],
            }
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return {"macd": macd_line, "signal": signal_line, "histogram": histogram}


def compute_bollinger_bands(close: pd.Series, period: int = 20, std_dev: float = 2.0) -> dict:
    """Upper, middle, lower Bollinger Bands."""
    if HAS_TA:
        bb = ta.bbands(close, length=period, std=std_dev)
        if bb is not None and not bb.empty:
            cols = bb.columns.tolist()
            return {
                "upper": bb[cols[0]],
                "mid": bb[cols[1]],
                "lower": bb[cols[2]],
                "bandwidth": bb[cols[3]] if len(cols) > 3 else None,
                "percent_b": bb[cols[4]] if len(cols) > 4 else None,
            }
    sma = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()
    return {
        "upper": sma + std_dev * std,
        "mid": sma,
        "lower": sma - std_dev * std,
        "bandwidth": None,
        "percent_b": None,
    }


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    if HAS_TA:
        return ta.atr(high, low, close, length=period)
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def compute_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    if HAS_TA:
        adx_df = ta.adx(high, low, close, length=period)
        if adx_df is not None and not adx_df.empty:
            return adx_df.iloc[:, 0]
    return pd.Series([np.nan] * len(close), index=close.index)


def detect_double_top(close: pd.Series, tolerance: float = 0.02) -> bool:
    series = close.dropna().tail(30).values
    if len(series) < 10:
        return False
    peak_idx = []
    for i in range(1, len(series) - 1):
        if series[i] > series[i - 1] and series[i] > series[i + 1]:
            peak_idx.append((i, series[i]))
    if len(peak_idx) < 2:
        return False
    last_two = peak_idx[-2:]
    p1, p2 = last_two[0][1], last_two[1][1]
    return abs(p1 - p2) / max(p1, p2) <= tolerance


def detect_double_bottom(close: pd.Series, tolerance: float = 0.02) -> bool:
    series = close.dropna().tail(30).values
    if len(series) < 10:
        return False
    trough_idx = []
    for i in range(1, len(series) - 1):
        if series[i] < series[i - 1] and series[i] < series[i + 1]:
            trough_idx.append((i, series[i]))
    if len(trough_idx) < 2:
        return False
    last_two = trough_idx[-2:]
    p1, p2 = last_two[0][1], last_two[1][1]
    return abs(p1 - p2) / max(p1, p2) <= tolerance


def detect_head_and_shoulders(close: pd.Series, tolerance: float = 0.03) -> bool:
    series = close.dropna().tail(40).values
    if len(series) < 20:
        return False
    peaks = []
    for i in range(1, len(series) - 1):
        if series[i] > series[i - 1] and series[i] > series[i + 1]:
            peaks.append((i, series[i]))
    if len(peaks) < 3:
        return False
    left, head, right = peaks[-3], peaks[-2], peaks[-1]
    if not (head[1] > left[1] and head[1] > right[1]):
        return False
    shoulder_diff = abs(left[1] - right[1]) / max(left[1], right[1])
    return shoulder_diff <= tolerance


def _detect_patterns(close: pd.Series) -> dict:
    return {
        "double_top": detect_double_top(close),
        "double_bottom": detect_double_bottom(close),
        "head_and_shoulders": detect_head_and_shoulders(close),
    }


def _score_rsi(rsi_val: float) -> tuple[float, str]:

    if pd.isna(rsi_val):
        return 0.0, "RSI: N/A"
    if rsi_val < 25:
        return 1.0, f"RSI={rsi_val:.1f} (Strongly Oversold)"
    if rsi_val < 35:
        return 0.5, f"RSI={rsi_val:.1f} (Oversold)"
    if rsi_val > 75:
        return -1.0, f"RSI={rsi_val:.1f} (Strongly Overbought)"
    if rsi_val > 65:
        return -0.5, f"RSI={rsi_val:.1f} (Overbought)"
    return 0.0, f"RSI={rsi_val:.1f} (Neutral)"


def _score_macd(macd_val: float, signal_val: float, hist_val: float) -> tuple[float, str]:
    if any(pd.isna(v) for v in [macd_val, signal_val, hist_val]):
        return 0.0, "MACD: N/A"
    bullish_cross = macd_val > signal_val and hist_val > 0
    bearish_cross = macd_val < signal_val and hist_val < 0
    if bullish_cross:
        score = min(abs(hist_val) / max(abs(macd_val), 0.01), 1.0)
        return score, f"MACD Bullish (hist={hist_val:.3f})"
    if bearish_cross:
        score = -min(abs(hist_val) / max(abs(macd_val), 0.01), 1.0)
        return score, f"MACD Bearish (hist={hist_val:.3f})"
    return 0.0, f"MACD Neutral (hist={hist_val:.3f})"


def _score_bb(close_val: float, upper: float, lower: float, mid: float) -> tuple[float, str]:
    if any(pd.isna(v) for v in [upper, lower, mid]):
        return 0.0, "BB: N/A"
    band_width = upper - lower
    if band_width == 0:
        return 0.0, "BB: N/A"
    pos = (close_val - lower) / band_width  # 0=lower, 1=upper
    if pos < 0.1:
        return 0.8, f"BB: Price near lower band (pos={pos:.2f})"
    if pos > 0.9:
        return -0.8, f"BB: Price near upper band (pos={pos:.2f})"
    if pos < 0.3:
        return 0.3, f"BB: Price in lower zone (pos={pos:.2f})"
    if pos > 0.7:
        return -0.3, f"BB: Price in upper zone (pos={pos:.2f})"
    return 0.0, f"BB: Price midrange (pos={pos:.2f})"


def _score_pattern(patterns: dict) -> tuple[float, str]:
    notes = []
    score = 0.0
    if patterns.get("double_bottom"):
        score += 0.6
        notes.append("Double Bottom detected (Bullish)")
    if patterns.get("double_top"):
        score -= 0.6
        notes.append("Double Top detected (Bearish)")
    if patterns.get("head_and_shoulders"):
        score -= 0.7
        notes.append("Head & Shoulders detected (Bearish reversal)")
    return max(-1.0, min(1.0, score)), "; ".join(notes) if notes else "No major patterns"


async def generate_trade_signal(
    symbol: str,
    timeframe: str = "3mo",
    exchange: str = "NSE",
    sentiment_score: float = 0.0,  # External sentiment input from Module 3
) -> dict:
    
    df = await get_historical_ohlc(symbol, period=timeframe, exchange=exchange)
    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    rsi_series = compute_rsi(close)
    rsi_val = float(rsi_series.iloc[-1]) if not rsi_series.empty else float("nan")

    macd_data = compute_macd(close)
    macd_val = float(macd_data["macd"].iloc[-1])
    sig_val = float(macd_data["signal"].iloc[-1])
    hist_val = float(macd_data["histogram"].iloc[-1])

    bb = compute_bollinger_bands(close)
    bb_upper = float(bb["upper"].iloc[-1])
    bb_lower = float(bb["lower"].iloc[-1])
    bb_mid = float(bb["mid"].iloc[-1])
    close_val = float(close.iloc[-1])

    patterns = _detect_patterns(close)

    rsi_score, rsi_comment = _score_rsi(rsi_val)
    macd_score, macd_comment = _score_macd(macd_val, sig_val, hist_val)
    bb_score, bb_comment = _score_bb(close_val, bb_upper, bb_lower, bb_mid)
    pat_score, pat_comment = _score_pattern(patterns)
    sent_score = max(-1.0, min(1.0, sentiment_score))
    sent_comment = f"Sentiment score={sent_score:.2f}"

    weights = {"rsi": 0.25, "macd": 0.25, "bb": 0.20, "pattern": 0.15, "sentiment": 0.15}
    composite = (
        weights["rsi"] * rsi_score
        + weights["macd"] * macd_score
        + weights["bb"] * bb_score
        + weights["pattern"] * pat_score
        + weights["sentiment"] * sent_score
    )

    confidence = round(abs(composite) * 100, 1)

    if composite > 0.15:
        signal = "BUY"
    elif composite < -0.15:
        signal = "SELL"
    else:
        signal = "HOLD"

    adx_series = compute_adx(high, low, close)
    adx_val = float(adx_series.iloc[-1]) if not adx_series.isna().all() else None
    trend_strength = (
        "Strong" if adx_val and adx_val > 25
        else "Weak" if adx_val
        else "N/A"
    )

    return {
        "symbol": symbol,
        "signal": signal,
        "confidence": confidence,
        "composite_score": round(composite, 4),
        "trend_strength": trend_strength,
        "adx": round(adx_val, 2) if adx_val else None,
        "indicators": {
            "rsi": {"value": round(rsi_val, 2) if not pd.isna(rsi_val) else None, "comment": rsi_comment},
            "macd": {"value": round(hist_val, 4), "comment": macd_comment},
            "bollinger": {"comment": bb_comment},
            "patterns": {"detected": patterns, "comment": pat_comment},
        },
        "sentiment_input": round(sent_score, 4),
        "timeframe": timeframe,
    }
