import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

"""
IndiaQuant MCP Server
=====================
A real-time Indian stock market AI assistant using Model Context Protocol.

10 MCP Tools:
1. get_live_price         - Live NSE/BSE prices
2. get_options_chain      - Live options chain data
3. analyze_sentiment      - News sentiment analysis
4. generate_signal        - AI trade signal (BUY/SELL/HOLD)
5. get_portfolio_pnl      - Real-time portfolio P&L
6. place_virtual_trade    - Execute virtual trades
7. calculate_greeks       - Option Greeks (Black-Scholes from scratch)
8. detect_unusual_activity - OI spike & unusual options detection
9. scan_market            - Market-wide stock screener
10. get_sector_heatmap    - Sector-level % change heatmap
"""

import asyncio
import json
from typing import Optional

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field, field_validator, ConfigDict

from modules.market_data import (
    get_live_price,
    get_multiple_prices,
    get_historical_ohlc,
    format_symbol,
    NIFTY50_SYMBOLS,
    SECTOR_MAP,
)
from modules.signals import generate_trade_signal, compute_rsi, compute_macd
from modules.options import (
    get_options_chain,
    calculate_greeks,
    calculate_max_pain,
    detect_unusual_options_activity,
    implied_volatility_bisection,
)
from modules.portfolio import (
    get_portfolio_pnl,
    place_virtual_trade,
    calculate_position_risk,
    init_db,
)
from modules.sentiment import analyze_sentiment

mcp = FastMCP("indiaquant_mcp")

def _json(data: dict | list) -> str:
    return json.dumps(data, indent=2, default=str)


def _error(msg: str) -> str:
    return _json({"error": msg, "status": "failed"})

class SymbolInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    symbol: str = Field(..., description="Stock or index symbol (e.g. 'RELIANCE', 'NIFTY', 'HDFCBANK')", min_length=1, max_length=30)
    exchange: str = Field(default="NSE", description="Exchange: 'NSE' or 'BSE'")


class OptionsChainInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    symbol: str = Field(..., description="Underlying symbol (e.g. 'NIFTY', 'BANKNIFTY', 'RELIANCE')", min_length=1)
    expiry: Optional[str] = Field(default=None, description="Expiry date in 'YYYY-MM-DD' format. If None, nearest expiry is used.")
    exchange: str = Field(default="NSE", description="Exchange: 'NSE' or 'BSE'")


class SentimentInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    symbol: str = Field(..., description="Stock symbol (e.g. 'INFY', 'TCS')", min_length=1)
    company_name: Optional[str] = Field(default=None, description="Full company name for better news search (e.g. 'Infosys')")


class SignalInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    symbol: str = Field(..., description="Stock or index symbol", min_length=1)
    timeframe: str = Field(default="3mo", description="Historical period for analysis: '1mo', '3mo', '6mo', '1y'")
    exchange: str = Field(default="NSE", description="Exchange: 'NSE' or 'BSE'")


class GreeksInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    spot_price: float = Field(..., description="Current price of underlying (e.g. 24500.0 for Nifty)", gt=0)
    strike_price: float = Field(..., description="Option strike price (e.g. 24500)", gt=0)
    days_to_expiry: float = Field(..., description="Days remaining to option expiry (e.g. 7, 30)", gt=0)
    volatility_pct: float = Field(..., description="Implied/historical volatility as percentage (e.g. 15.5 for 15.5% IV)", gt=0, le=500)
    risk_free_rate_pct: float = Field(default=6.5, description="Risk-free rate as percentage (e.g. 6.5 for 6.5%). Default: 6.5 (RBI repo rate approx.)", gt=0)
    option_type: str = Field(default="CE", description="Option type: 'CE' (Call) or 'PE' (Put)")

    @field_validator("option_type")
    @classmethod
    def validate_option_type(cls, v: str) -> str:
        v = v.upper()
        if v not in ("CE", "PE", "CALL", "PUT"):
            raise ValueError("option_type must be 'CE' or 'PE'")
        return v


class VirtualTradeInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    symbol: str = Field(..., description="Stock symbol (e.g. 'RELIANCE', 'INFY')", min_length=1)
    quantity: int = Field(..., description="Number of shares", gt=0)
    side: str = Field(..., description="Trade direction: 'BUY' or 'SELL'")
    exchange: str = Field(default="NSE", description="Exchange: 'NSE' or 'BSE'")
    stop_loss: Optional[float] = Field(default=None, description="Stop-loss price level (optional)", gt=0)
    target: Optional[float] = Field(default=None, description="Target profit price level (optional)", gt=0)

    @field_validator("side")
    @classmethod
    def validate_side(cls, v: str) -> str:
        v = v.upper()
        if v not in ("BUY", "SELL"):
            raise ValueError("side must be 'BUY' or 'SELL'")
        return v


class ScanMarketInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    rsi_max: Optional[float] = Field(default=None, description="Filter: RSI below this value (e.g. 30 for oversold)", ge=0, le=100)
    rsi_min: Optional[float] = Field(default=None, description="Filter: RSI above this value (e.g. 70 for overbought)", ge=0, le=100)
    change_pct_min: Optional[float] = Field(default=None, description="Filter: minimum day change % (e.g. 2.0 for >2% gainers)")
    change_pct_max: Optional[float] = Field(default=None, description="Filter: maximum day change % (e.g. -2.0 for >2% losers)")
    sector: Optional[str] = Field(default=None, description="Filter by sector: IT, Banking, FMCG, Auto, Pharma, Energy, Metals, Financials, Infra")
    volume_min: Optional[int] = Field(default=None, description="Filter: minimum trading volume", ge=0)
    max_results: int = Field(default=10, description="Maximum stocks to return", ge=1, le=50)


class UnusualActivityInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    symbol: str = Field(..., description="Symbol to scan for unusual options activity", min_length=1)
    expiry: Optional[str] = Field(default=None, description="Options expiry date 'YYYY-MM-DD'. Uses nearest if None.")
    exchange: str = Field(default="NSE", description="Exchange")
    volume_oi_threshold: float = Field(default=0.5, description="Vol/OI ratio threshold to flag as unusual (default 0.5)", gt=0)


@mcp.tool(
    name="get_live_price",
    annotations={
        "title": "Get Live Stock Price",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True,
    }
)
async def get_live_price_tool(params: SymbolInput) -> str:

    try:
        data = await get_live_price(params.symbol, params.exchange)
        return _json(data)
    except Exception as e:
        return _error(f"Failed to fetch price for {params.symbol}: {str(e)}")


@mcp.tool(
    name="get_options_chain",
    annotations={
        "title": "Get Live Options Chain",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True,
    }
)
async def get_options_chain_tool(params: OptionsChainInput) -> str:

    try:
        chain = await get_options_chain(params.symbol, params.expiry, params.exchange)
        max_pain = calculate_max_pain(chain)
        chain["max_pain"] = max_pain
        return _json(chain)
    except Exception as e:
        return _error(f"Options chain fetch failed for {params.symbol}: {str(e)}")


@mcp.tool(
    name="analyze_sentiment",
    annotations={
        "title": "Analyze News Sentiment",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True,
    }
)
async def analyze_sentiment_tool(params: SentimentInput) -> str:

    try:
        result = await analyze_sentiment(params.symbol, params.company_name)
        return _json(result)
    except Exception as e:
        return _error(f"Sentiment analysis failed for {params.symbol}: {str(e)}")


@mcp.tool(
    name="generate_signal",
    annotations={
        "title": "Generate AI Trade Signal",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True,
    }
)
async def generate_signal_tool(params: SignalInput) -> str:

    try:
        sentiment_data = await analyze_sentiment(params.symbol)
        sentiment_score = sentiment_data.get("score", 0.0)

        result = await generate_trade_signal(
            params.symbol,
            timeframe=params.timeframe,
            exchange=params.exchange,
            sentiment_score=sentiment_score,
        )
        result["sentiment"] = {
            "score": sentiment_data.get("score"),
            "signal": sentiment_data.get("signal"),
        }
        return _json(result)
    except Exception as e:
        return _error(f"Signal generation failed for {params.symbol}: {str(e)}")


@mcp.tool(
    name="get_portfolio_pnl",
    annotations={
        "title": "Get Portfolio P&L",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True,
    }
)
async def get_portfolio_pnl_tool() -> str:

    try:
        await init_db()
        pnl = await get_portfolio_pnl()
        return _json(pnl)
    except Exception as e:
        return _error(f"Portfolio P&L calculation failed: {str(e)}")


@mcp.tool(
    name="place_virtual_trade",
    annotations={
        "title": "Place Virtual Trade",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True,
    }
)
async def place_virtual_trade_tool(params: VirtualTradeInput) -> str:

    try:
        await init_db()
        result = await place_virtual_trade(
            symbol=params.symbol,
            quantity=params.quantity,
            side=params.side,
            exchange=params.exchange,
            stop_loss=params.stop_loss,
            target=params.target,
        )
        return _json(result)
    except Exception as e:
        return _error(f"Virtual trade failed: {str(e)}")


@mcp.tool(
    name="calculate_greeks",
    annotations={
        "title": "Calculate Option Greeks (Black-Scholes)",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
async def calculate_greeks_tool(params: GreeksInput) -> str:

    try:
        T = params.days_to_expiry / 365.0
        sigma = params.volatility_pct / 100.0
        r = params.risk_free_rate_pct / 100.0
        S = params.spot_price
        K = params.strike_price

        greeks = calculate_greeks(S, K, T, r, sigma, params.option_type)

        if params.option_type.upper() in ("CE", "CALL"):
            intrinsic = max(S - K, 0)
            moneyness = "ITM" if S > K else ("ATM" if abs(S - K) / K < 0.01 else "OTM")
        else:
            intrinsic = max(K - S, 0)
            moneyness = "ITM" if S < K else ("ATM" if abs(S - K) / K < 0.01 else "OTM")

        time_value = max(greeks["price"] - intrinsic, 0)

        result = {
            **greeks,
            "inputs": {
                "spot_price": S,
                "strike_price": K,
                "days_to_expiry": params.days_to_expiry,
                "volatility_pct": params.volatility_pct,
                "risk_free_rate_pct": params.risk_free_rate_pct,
                "option_type": params.option_type,
            },
            "moneyness": moneyness,
            "intrinsic_value": round(intrinsic, 2),
            "time_value": round(time_value, 2),
            "interpretation": {
                "delta": f"Option moves ₹{abs(greeks['delta']):.4f} per ₹1 move in underlying",
                "gamma": f"Delta changes by {greeks['gamma']:.6f} per ₹1 move in underlying",
                "theta": f"Option loses ₹{abs(greeks['theta']):.4f} per calendar day",
                "vega": f"Option gains/loses ₹{greeks['vega']:.4f} per 1% change in IV",
            },
        }
        return _json(result)
    except Exception as e:
        return _error(f"Greeks calculation failed: {str(e)}")


@mcp.tool(
    name="detect_unusual_activity",
    annotations={
        "title": "Detect Unusual Options Activity",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True,
    }
)
async def detect_unusual_activity_tool(params: UnusualActivityInput) -> str:

    try:
        chain = await get_options_chain(params.symbol, params.expiry, params.exchange)
        unusual = detect_unusual_options_activity(chain, params.volume_oi_threshold)
        max_pain = calculate_max_pain(chain)
        unusual["max_pain"] = max_pain
        unusual["spot_price"] = chain.get("spot_price")
        unusual["expiry"] = chain.get("expiry")
        unusual["symbol"] = params.symbol
        return _json(unusual)
    except Exception as e:
        return _error(f"Unusual activity detection failed for {params.symbol}: {str(e)}")


@mcp.tool(
    name="scan_market",
    annotations={
        "title": "Scan Market with Filters",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True,
    }
)
async def scan_market_tool(params: ScanMarketInput) -> str:

    try:
        if params.sector and params.sector in SECTOR_MAP:
            symbols_to_scan = SECTOR_MAP[params.sector]
        else:
            symbols_to_scan = NIFTY50_SYMBOLS

        price_data = await get_multiple_prices(symbols_to_scan)

        async def get_rsi(symbol: str) -> float:
            try:
                df = await get_historical_ohlc(symbol, period="3mo")
                from modules.signals import compute_rsi
                rsi_series = compute_rsi(df["Close"])
                return float(rsi_series.iloc[-1]) if not rsi_series.empty else 50.0
            except Exception:
                return 50.0

        rsi_values = await asyncio.gather(*[get_rsi(s) for s in symbols_to_scan], return_exceptions=True)

        symbol_sector = {}
        for sector, syms in SECTOR_MAP.items():
            for s in syms:
                symbol_sector[s] = sector

        matches = []
        for price_info, rsi_val in zip(price_data, rsi_values):
            if "error" in price_info:
                continue

            symbol = price_info["symbol"].replace(".NS", "").replace(".BO", "")
            change_pct = price_info.get("change_pct", 0)
            volume = price_info.get("volume", 0)
            rsi = float(rsi_val) if not isinstance(rsi_val, Exception) else 50.0

            if params.rsi_max is not None and rsi > params.rsi_max:
                continue
            if params.rsi_min is not None and rsi < params.rsi_min:
                continue
            if params.change_pct_min is not None and change_pct < params.change_pct_min:
                continue
            if params.change_pct_max is not None and change_pct > params.change_pct_max:
                continue
            if params.volume_min is not None and volume < params.volume_min:
                continue

            matches.append({
                "symbol": symbol,
                "price": price_info["price"],
                "change_pct": change_pct,
                "volume": volume,
                "rsi": round(rsi, 1),
                "sector": symbol_sector.get(symbol, "Unknown"),
            })

        if params.rsi_max is not None:
            matches.sort(key=lambda x: x["rsi"])
        elif params.rsi_min is not None:
            matches.sort(key=lambda x: x["rsi"], reverse=True)
        else:
            matches.sort(key=lambda x: abs(x["change_pct"]), reverse=True)

        matches = matches[: params.max_results]

        return _json({
            "matches": matches,
            "count": len(matches),
            "scanned": len(symbols_to_scan),
            "filters_applied": {
                "rsi_max": params.rsi_max,
                "rsi_min": params.rsi_min,
                "change_pct_min": params.change_pct_min,
                "change_pct_max": params.change_pct_max,
                "sector": params.sector,
                "volume_min": params.volume_min,
            },
        })
    except Exception as e:
        return _error(f"Market scan failed: {str(e)}")


# ─── TOOL 10: Get Sector Heatmap ─────────────────

@mcp.tool(
    name="get_sector_heatmap",
    annotations={
        "title": "Get Sector Performance Heatmap",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True,
    }
)
async def get_sector_heatmap_tool() -> str:

    try:
        all_symbols = list({s for syms in SECTOR_MAP.values() for s in syms})
        price_data = await get_multiple_prices(all_symbols)

        price_lookup = {}
        for p in price_data:
            if "error" not in p:
                sym = p["symbol"].replace(".NS", "").replace(".BO", "")
                price_lookup[sym] = p

        sectors = []
        for sector, symbols in SECTOR_MAP.items():
            sector_stocks = []
            for sym in symbols:
                if sym in price_lookup:
                    p = price_lookup[sym]
                    sector_stocks.append({
                        "symbol": sym,
                        "price": p["price"],
                        "change_pct": p["change_pct"],
                        "volume": p["volume"],
                    })

            if not sector_stocks:
                continue

            changes = [s["change_pct"] for s in sector_stocks]
            avg_change = round(sum(changes) / len(changes), 2)
            top_gainer = max(sector_stocks, key=lambda x: x["change_pct"])
            top_loser = min(sector_stocks, key=lambda x: x["change_pct"])

            sectors.append({
                "sector": sector,
                "avg_change_pct": avg_change,
                "stock_count": len(sector_stocks),
                "top_gainer": {"symbol": top_gainer["symbol"], "change_pct": top_gainer["change_pct"]},
                "top_loser": {"symbol": top_loser["symbol"], "change_pct": top_loser["change_pct"]},
                "stocks": sorted(sector_stocks, key=lambda x: x["change_pct"], reverse=True),
                "sentiment": "Bullish" if avg_change > 0.5 else ("Bearish" if avg_change < -0.5 else "Neutral"),
            })

        sectors.sort(key=lambda x: x["avg_change_pct"], reverse=True)

        return _json({
            "heatmap": sectors,
            "market_breadth": {
                "advancing_sectors": sum(1 for s in sectors if s["avg_change_pct"] > 0),
                "declining_sectors": sum(1 for s in sectors if s["avg_change_pct"] < 0),
                "best_sector": sectors[0]["sector"] if sectors else None,
                "worst_sector": sectors[-1]["sector"] if sectors else None,
            },
        })
    except Exception as e:
        return _error(f"Sector heatmap failed: {str(e)}")


if __name__ == "__main__":
    mcp.run()
