"""
Module 4: Portfolio Risk Manager
"""

import asyncio
import json
import uuid
from datetime import datetime
from typing import Optional

import aiosqlite
import numpy as np
import pandas as pd

from modules.market_data import get_live_price, get_historical_ohlc

DB_PATH = "portfolio.db"


async def init_db():
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS portfolio (
                id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                exchange TEXT NOT NULL DEFAULT 'NSE',
                side TEXT NOT NULL,           -- BUY or SELL
                quantity INTEGER NOT NULL,
                entry_price REAL NOT NULL,
                stop_loss REAL,
                target REAL,
                placed_at TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'OPEN'  -- OPEN, CLOSED, STOPPED
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS cash_balance (
                id INTEGER PRIMARY KEY DEFAULT 1,
                balance REAL NOT NULL DEFAULT 1000000.0,
                updated_at TEXT NOT NULL
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS trade_history (
                id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                quantity INTEGER NOT NULL,
                entry_price REAL NOT NULL,
                exit_price REAL,
                pnl REAL,
                opened_at TEXT NOT NULL,
                closed_at TEXT,
                reason TEXT
            )
        """)
        await db.execute("""
            INSERT OR IGNORE INTO cash_balance (id, balance, updated_at)
            VALUES (1, 1000000.0, ?)
        """, (datetime.now().isoformat(),))
        await db.commit()


async def get_cash_balance() -> float:
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute("SELECT balance FROM cash_balance WHERE id=1") as cursor:
            row = await cursor.fetchone()
            return row[0] if row else 1_000_000.0


async def _update_cash(new_balance: float):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "UPDATE cash_balance SET balance=?, updated_at=? WHERE id=1",
            (new_balance, datetime.now().isoformat())
        )
        await db.commit()


async def place_virtual_trade(
    symbol: str,
    quantity: int,
    side: str,  
    exchange: str = "NSE",
    stop_loss: Optional[float] = None,
    target: Optional[float] = None,
) -> dict:
    
    await init_db()
    side = side.upper()
    if side not in ("BUY", "SELL"):
        raise ValueError("Side must be 'BUY' or 'SELL'")

    # Fetch live price
    price_data = await get_live_price(symbol, exchange)
    entry_price = price_data["price"]

    # Check cash for BUY orders
    cash = await get_cash_balance()
    trade_value = entry_price * quantity

    if side == "BUY":
        if trade_value > cash:
            return {
                "status": "REJECTED",
                "reason": f"Insufficient cash. Required: ₹{trade_value:,.2f}, Available: ₹{cash:,.2f}",
                "order_id": None,
            }
        await _update_cash(cash - trade_value)
    else:
        await _update_cash(cash + trade_value)

    order_id = str(uuid.uuid4())[:8].upper()

    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            INSERT INTO portfolio (id, symbol, exchange, side, quantity, entry_price, stop_loss, target, placed_at, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'OPEN')
        """, (order_id, symbol.upper(), exchange.upper(), side, quantity, entry_price, stop_loss, target, datetime.now().isoformat()))
        await db.commit()

    return {
        "order_id": order_id,
        "status": "EXECUTED",
        "symbol": symbol.upper(),
        "side": side,
        "quantity": quantity,
        "entry_price": entry_price,
        "trade_value": round(trade_value, 2),
        "stop_loss": stop_loss,
        "target": target,
        "timestamp": datetime.now().isoformat(),
        "remaining_cash": round(cash - trade_value if side == "BUY" else cash + trade_value, 2),
    }


async def get_portfolio_pnl() -> dict:
    await init_db()

    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT id, symbol, exchange, side, quantity, entry_price, stop_loss, target, placed_at FROM portfolio WHERE status='OPEN'"
        ) as cursor:
            rows = await cursor.fetchall()

    if not rows:
        cash = await get_cash_balance()
        return {
            "positions": [],
            "total_invested": 0.0,
            "total_current_value": 0.0,
            "total_pnl": 0.0,
            "total_pnl_pct": 0.0,
            "cash_balance": cash,
            "portfolio_value": cash,
        }

    symbols_exchange = [(r[1], r[2]) for r in rows]
    price_tasks = [get_live_price(sym, exch) for sym, exch in symbols_exchange]
    price_results = await asyncio.gather(*price_tasks, return_exceptions=True)

    positions = []
    total_invested = 0.0
    total_current = 0.0

    for row, price_res in zip(rows, price_results):
        order_id, symbol, exchange, side, qty, entry_price, stop_loss, target, placed_at = row

        if isinstance(price_res, Exception):
            current_price = entry_price
            price_change_pct = 0.0
        else:
            current_price = price_res["price"]
            price_change_pct = price_res["change_pct"]

        if side == "BUY":
            pnl = (current_price - entry_price) * qty
            invested = entry_price * qty
        else:  
            pnl = (entry_price - current_price) * qty
            invested = entry_price * qty

        pnl_pct = (pnl / invested * 100) if invested > 0 else 0.0
        current_value = current_price * qty

        triggered = None
        if stop_loss and side == "BUY" and current_price <= stop_loss:
            triggered = "STOP_LOSS"
        elif stop_loss and side == "SELL" and current_price >= stop_loss:
            triggered = "STOP_LOSS"
        elif target and side == "BUY" and current_price >= target:
            triggered = "TARGET"
        elif target and side == "SELL" and current_price <= target:
            triggered = "TARGET"

        positions.append({
            "order_id": order_id,
            "symbol": symbol,
            "side": side,
            "quantity": qty,
            "entry_price": round(entry_price, 2),
            "current_price": round(current_price, 2),
            "invested": round(invested, 2),
            "current_value": round(current_value, 2),
            "pnl": round(pnl, 2),
            "pnl_pct": round(pnl_pct, 2),
            "day_change_pct": round(price_change_pct, 2),
            "stop_loss": stop_loss,
            "target": target,
            "trigger_status": triggered,
            "placed_at": placed_at,
        })

        total_invested += invested
        total_current += current_value

    total_pnl = total_current - total_invested
    total_pnl_pct = (total_pnl / total_invested * 100) if total_invested > 0 else 0.0
    cash = await get_cash_balance()

    return {
        "positions": positions,
        "total_invested": round(total_invested, 2),
        "total_current_value": round(total_current, 2),
        "total_pnl": round(total_pnl, 2),
        "total_pnl_pct": round(total_pnl_pct, 2),
        "cash_balance": round(cash, 2),
        "portfolio_value": round(total_current + cash, 2),
        "positions_count": len(positions),
    }


async def calculate_position_risk(
    symbol: str,
    quantity: int,
    entry_price: float,
    exchange: str = "NSE",
) -> dict:

    try:
        df = await get_historical_ohlc(symbol, period="6mo", exchange=exchange)
        close = df["Close"]
        returns = close.pct_change().dropna()

        daily_vol = float(returns.std())
        annual_vol = daily_vol * np.sqrt(252)

        position_value = entry_price * quantity
        var_95 = position_value * daily_vol * 1.645  

        annual_return = float(returns.mean()) * 252
        risk_free = 0.065
        sharpe = (annual_return - risk_free) / annual_vol if annual_vol > 0 else 0

        if annual_vol < 0.10:
            risk_score = 2
        elif annual_vol < 0.20:
            risk_score = 4
        elif annual_vol < 0.30:
            risk_score = 6
        elif annual_vol < 0.40:
            risk_score = 8
        else:
            risk_score = 10

        return {
            "symbol": symbol,
            "risk_score": risk_score,
            "risk_level": ["Low", "Low", "Moderate", "Moderate", "High", "High", "Very High", "Very High", "Extreme", "Extreme", "Extreme"][risk_score],
            "daily_volatility_pct": round(daily_vol * 100, 2),
            "annual_volatility_pct": round(annual_vol * 100, 2),
            "var_95_daily": round(var_95, 2),
            "var_95_pct": round(daily_vol * 1.645 * 100, 2),
            "sharpe_ratio": round(sharpe, 3),
            "position_value": round(position_value, 2),
            "max_recommended_loss": round(position_value * 0.02, 2),  
        }

    except Exception as e:
        return {"symbol": symbol, "error": str(e), "risk_score": 5}
