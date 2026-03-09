"""
Module 3: Options Chain Analyzer
"""

import asyncio
import math
from datetime import datetime, date
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

from modules.market_data import format_symbol

def _standard_normal_cdf(x: float) -> float:
    if x < -10:
        return 0.0
    if x > 10:
        return 1.0
    return 0.5 * math.erfc(-x / math.sqrt(2))


def _standard_normal_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def _bs_d1_d2(
    S: float,   
    K: float,   
    T: float,   
    r: float,   
    sigma: float,  
) -> tuple[float, float]:
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        raise ValueError("Invalid Black-Scholes inputs (S, K, T, sigma must be positive).")
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return d1, d2


def bs_call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    d1, d2 = _bs_d1_d2(S, K, T, r, sigma)
    return S * _standard_normal_cdf(d1) - K * math.exp(-r * T) * _standard_normal_cdf(d2)


def bs_put_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    d1, d2 = _bs_d1_d2(S, K, T, r, sigma)
    return K * math.exp(-r * T) * _standard_normal_cdf(-d2) - S * _standard_normal_cdf(-d1)


def calculate_greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "CE",  # CE = Call, PE = Put
) -> dict:
   
    if T <= 1e-6:
        # Option has expired – intrinsic value only
        if option_type.upper() in ("CE", "CALL"):
            price = max(S - K, 0)
            delta = 1.0 if S > K else 0.0
        else:
            price = max(K - S, 0)
            delta = -1.0 if S < K else 0.0
        return {
            "price": round(price, 2),
            "delta": round(delta, 4),
            "gamma": 0.0,
            "theta": 0.0,
            "vega": 0.0,
        }

    d1, d2 = _bs_d1_d2(S, K, T, r, sigma)
    nd1 = _standard_normal_cdf(d1)
    nd2 = _standard_normal_cdf(d2)
    n_neg_d1 = _standard_normal_cdf(-d1)
    n_neg_d2 = _standard_normal_cdf(-d2)
    pdf_d1 = _standard_normal_pdf(d1)

    sqrt_T = math.sqrt(T)
    exp_rT = math.exp(-r * T)

    gamma = pdf_d1 / (S * sigma * sqrt_T)

    vega = S * pdf_d1 * sqrt_T * 0.01 

    if option_type.upper() in ("CE", "CALL"):
        price = bs_call_price(S, K, T, r, sigma)
        delta = nd1
        theta = (
            -(S * pdf_d1 * sigma) / (2 * sqrt_T)
            - r * K * exp_rT * nd2
        ) / 365
    else:  
        price = bs_put_price(S, K, T, r, sigma)
        delta = nd1 - 1 
        theta = (
            -(S * pdf_d1 * sigma) / (2 * sqrt_T)
            + r * K * exp_rT * n_neg_d2
        ) / 365

    return {
        "price": round(price, 2),
        "delta": round(delta, 4),
        "gamma": round(gamma, 6),
        "theta": round(theta, 4),  
        "vega": round(vega, 4),    
    }


def implied_volatility_bisection(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str = "CE",
    tol: float = 1e-5,
    max_iter: int = 200,
) -> Optional[float]:

    if T <= 0 or market_price <= 0:
        return None

    low_vol, high_vol = 0.001, 5.0

    def price_at_vol(v: float) -> float:
        try:
            greeks = calculate_greeks(S, K, T, r, v, option_type)
            return greeks["price"]
        except Exception:
            return 0.0

    low_price = price_at_vol(low_vol)
    high_price = price_at_vol(high_vol)

    if market_price < low_price or market_price > high_price:
        return None

    for _ in range(max_iter):
        mid_vol = (low_vol + high_vol) / 2
        mid_price = price_at_vol(mid_vol)
        if abs(mid_price - market_price) < tol:
            return round(mid_vol, 4)
        if mid_price < market_price:
            low_vol = mid_vol
        else:
            high_vol = mid_vol

    return round((low_vol + high_vol) / 2, 4)


async def get_options_chain(symbol: str, expiry: Optional[str] = None, exchange: str = "NSE") -> dict:
    yf_symbol = format_symbol(symbol, exchange)
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, _fetch_options_chain, yf_symbol, expiry)
    return result


def _fetch_options_chain(yf_symbol: str, expiry: Optional[str]) -> dict:
    ticker = yf.Ticker(yf_symbol)
    expirations = ticker.options
    if not expirations:
        raise ValueError(f"No options data available for {yf_symbol}")

    if expiry:
        chosen = min(expirations, key=lambda e: abs(
            (datetime.strptime(e, "%Y-%m-%d") - datetime.strptime(expiry, "%Y-%m-%d")).days
        ))
    else:
        chosen = expirations[0]

    chain = ticker.option_chain(chosen)
    calls = chain.calls.copy()
    puts = chain.puts.copy()

    info = ticker.fast_info
    try:
        spot = float(info.last_price)
    except Exception:
        hist = ticker.history(period="1d")
        spot = float(hist["Close"].iloc[-1]) if not hist.empty else 0.0

    def clean_df(df: pd.DataFrame, opt_type: str) -> list:
        records = []
        for _, row in df.iterrows():
            records.append({
                "strike": float(row.get("strike", 0)),
                "last_price": float(row.get("lastPrice", 0)),
                "bid": float(row.get("bid", 0)),
                "ask": float(row.get("ask", 0)),
                "volume": int(row.get("volume", 0) or 0),
                "open_interest": int(row.get("openInterest", 0) or 0),
                "implied_volatility": round(float(row.get("impliedVolatility", 0) or 0), 4),
                "in_the_money": bool(row.get("inTheMoney", False)),
                "type": opt_type,
            })
        return records

    return {
        "symbol": yf_symbol,
        "expiry": chosen,
        "all_expiries": list(expirations),
        "spot_price": round(spot, 2),
        "calls": clean_df(calls, "CE"),
        "puts": clean_df(puts, "PE"),
    }


def calculate_max_pain(chain: dict) -> dict:

    calls = chain.get("calls", [])
    puts = chain.get("puts", [])

    call_map = {c["strike"]: c["open_interest"] for c in calls if c["open_interest"] > 0}
    put_map = {p["strike"]: p["open_interest"] for p in puts if p["open_interest"] > 0}
    all_strikes = sorted(set(list(call_map.keys()) + list(put_map.keys())))

    if not all_strikes:
        return {"max_pain_strike": None, "details": {}}

    pain_at_strike = {}
    for K in all_strikes:
        call_pain = sum(
            max(K - strike, 0) * oi
            for strike, oi in call_map.items()
        )
        put_pain = sum(
            max(strike - K, 0) * oi
            for strike, oi in put_map.items()
        )
        pain_at_strike[K] = call_pain + put_pain

    max_pain_strike = min(pain_at_strike, key=pain_at_strike.get)
    return {
        "max_pain_strike": max_pain_strike,
        "spot_price": chain.get("spot_price"),
        "expiry": chain.get("expiry"),
        "bias": (
            "Bullish" if chain.get("spot_price", 0) < max_pain_strike
            else "Bearish" if chain.get("spot_price", 0) > max_pain_strike
            else "Neutral"
        ),
        "pain_levels": {
            str(k): round(v, 0)
            for k, v in sorted(pain_at_strike.items(), key=lambda x: x[1])[:5]
        },
    }


def detect_unusual_options_activity(chain: dict, volume_oi_threshold: float = 0.5, oi_spike_pct: float = 3.0) -> dict:

    alerts = []
    anomalies = []

    all_options = chain.get("calls", []) + chain.get("puts", [])
    if not all_options:
        return {"alerts": [], "anomalies": [], "summary": "No options data"}

    all_oi = [o["open_interest"] for o in all_options if o["open_interest"] > 0]
    if not all_oi:
        return {"alerts": [], "anomalies": [], "summary": "No OI data"}

    mean_oi = np.mean(all_oi)
    std_oi = np.std(all_oi)
    oi_spike_threshold = mean_oi + oi_spike_pct * std_oi

    for opt in all_options:
        strike = opt["strike"]
        oi = opt["open_interest"]
        vol = opt["volume"]
        opt_type = opt["type"]
        iv = opt.get("implied_volatility", 0)

        if oi > oi_spike_threshold:
            anomalies.append({
                "type": "OI_SPIKE",
                "option": f"{opt_type} {strike}",
                "open_interest": oi,
                "description": f"OI of {oi:,} is {(oi - mean_oi) / max(std_oi, 1):.1f}σ above mean",
            })

        if oi > 100 and vol > 0:
            vol_oi_ratio = vol / oi
            if vol_oi_ratio > volume_oi_threshold:
                alerts.append({
                    "type": "HIGH_VOL_OI_RATIO",
                    "option": f"{opt_type} {strike}",
                    "ratio": round(vol_oi_ratio, 2),
                    "volume": vol,
                    "open_interest": oi,
                    "description": f"Volume/OI={vol_oi_ratio:.2f} (new positions likely)",
                })

    alerts = sorted(alerts, key=lambda x: x["ratio"], reverse=True)[:10]
    anomalies = sorted(anomalies, key=lambda x: x["open_interest"], reverse=True)[:10]

    summary_parts = []
    if alerts:
        summary_parts.append(f"{len(alerts)} high Vol/OI alerts")
    if anomalies:
        summary_parts.append(f"{len(anomalies)} OI spikes detected")

    return {
        "alerts": alerts,
        "anomalies": anomalies,
        "summary": "; ".join(summary_parts) if summary_parts else "No unusual activity detected",
        "stats": {
            "mean_oi": round(mean_oi, 0),
            "std_oi": round(std_oi, 0),
            "oi_spike_threshold": round(oi_spike_threshold, 0),
        },
    }
