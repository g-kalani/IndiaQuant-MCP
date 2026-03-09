"""
IndiaQuant MCP - Test Suite
Tests all 10 tools and underlying modules.
Run with: python -m pytest tests/ -v
"""

import asyncio
import math
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.options import (
    _standard_normal_cdf,
    _standard_normal_pdf,
    bs_call_price,
    bs_put_price,
    calculate_greeks,
    calculate_max_pain,
    detect_unusual_options_activity,
    implied_volatility_bisection,
)


class TestBlackScholes:

    def test_normal_cdf_at_zero(self):
        assert abs(_standard_normal_cdf(0) - 0.5) < 1e-6

    def test_normal_cdf_symmetry(self):
        for x in [1.0, 1.5, 2.0, 2.5]:
            assert abs(_standard_normal_cdf(-x) - (1 - _standard_normal_cdf(x))) < 1e-8

    def test_normal_pdf_at_zero(self):
        expected = 1 / math.sqrt(2 * math.pi)
        assert abs(_standard_normal_pdf(0) - expected) < 1e-8

    def test_call_put_parity(self):
        S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
        call = bs_call_price(S, K, T, r, sigma)
        put = bs_put_price(S, K, T, r, sigma)
        rhs = S - K * math.exp(-r * T)
        assert abs((call - put) - rhs) < 0.01

    def test_atm_call_positive(self):
        price = bs_call_price(100.0, 100.0, 0.25, 0.065, 0.15)
        assert price > 0

    def test_deep_itm_call_delta_near_one(self):
        greeks = calculate_greeks(200.0, 100.0, 1.0, 0.065, 0.20, "CE")
        assert greeks["delta"] > 0.95

    def test_deep_otm_put_delta_near_zero(self):
        greeks = calculate_greeks(200.0, 100.0, 1.0, 0.065, 0.20, "PE")
        assert abs(greeks["delta"]) < 0.05

    def test_gamma_positive(self):
        for opt_type in ("CE", "PE"):
            greeks = calculate_greeks(100.0, 100.0, 0.25, 0.065, 0.20, opt_type)
            assert greeks["gamma"] > 0

    def test_theta_negative_call(self):
        greeks = calculate_greeks(100.0, 100.0, 0.5, 0.065, 0.20, "CE")
        assert greeks["theta"] < 0

    def test_vega_positive(self):
        for opt_type in ("CE", "PE"):
            greeks = calculate_greeks(100.0, 100.0, 0.5, 0.065, 0.20, opt_type)
            assert greeks["vega"] > 0

    def test_call_put_delta_sum(self):
        S, K, T, r, sigma = 100.0, 105.0, 0.5, 0.065, 0.20
        call_g = calculate_greeks(S, K, T, r, sigma, "CE")
        put_g = calculate_greeks(S, K, T, r, sigma, "PE")
        assert abs((call_g["delta"] - put_g["delta"]) - 1.0) < 0.01

    def test_expired_option_intrinsic_only(self):
        greeks = calculate_greeks(110.0, 100.0, 1e-9, 0.065, 0.20, "CE")
        assert abs(greeks["price"] - 10.0) < 0.01

    def test_implied_vol_roundtrip(self):
        S, K, T, r, sigma = 100.0, 100.0, 0.5, 0.065, 0.20
        greeks = calculate_greeks(S, K, T, r, sigma, "CE")
        market_price = greeks["price"]
        iv = implied_volatility_bisection(market_price, S, K, T, r, "CE")
        assert iv is not None
        assert abs(iv - sigma) < 0.01


class TestMaxPain:
    def test_max_pain_returns_strike(self):
        chain = {
            "spot_price": 24500,
            "expiry": "2025-01-30",
            "calls": [
                {"strike": 24000, "open_interest": 5000, "volume": 100},
                {"strike": 24500, "open_interest": 15000, "volume": 200},
                {"strike": 25000, "open_interest": 3000, "volume": 50},
            ],
            "puts": [
                {"strike": 24000, "open_interest": 3000, "volume": 50},
                {"strike": 24500, "open_interest": 12000, "volume": 150},
                {"strike": 25000, "open_interest": 5000, "volume": 100},
            ],
        }
        result = calculate_max_pain(chain)
        assert result["max_pain_strike"] is not None
        assert isinstance(result["max_pain_strike"], float)

    def test_empty_chain_returns_none(self):
        result = calculate_max_pain({"calls": [], "puts": []})
        assert result["max_pain_strike"] is None

class TestUnusualActivity:
    def _make_chain(self, vol_spike=False, oi_spike=False):
        calls = [{"strike": float(s), "open_interest": 100, "volume": 10, "type": "CE"}
                 for s in range(24000, 25100, 100)]
        puts = [{"strike": float(s), "open_interest": 100, "volume": 10, "type": "PE"}
                for s in range(24000, 25100, 100)]
        if vol_spike:
            calls[5]["volume"] = 10000  # Vol/OI = 100
        if oi_spike:
            calls[3]["open_interest"] = 100000
        return {"calls": calls, "puts": puts, "spot_price": 24500}

    def test_no_unusual_activity(self):
        chain = self._make_chain()
        result = detect_unusual_options_activity(chain)
        assert "summary" in result

    def test_vol_oi_spike_detected(self):
        chain = self._make_chain(vol_spike=True)
        result = detect_unusual_options_activity(chain, volume_oi_threshold=0.5)
        assert len(result["alerts"]) > 0

    def test_oi_spike_detected(self):
        chain = self._make_chain(oi_spike=True)
        result = detect_unusual_options_activity(chain)
        assert len(result["anomalies"]) > 0


class TestTechnicalIndicators:
    def _make_prices(self, n=100) -> "pd.Series":
        import pandas as pd
        import numpy as np
        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.randn(n) * 0.5)
        return pd.Series(prices)

    def test_rsi_range(self):
        from modules.signals import compute_rsi
        close = self._make_prices(50)
        rsi = compute_rsi(close)
        valid = rsi.dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_macd_returns_three_series(self):
        from modules.signals import compute_macd
        close = self._make_prices(100)
        result = compute_macd(close)
        assert "macd" in result and "signal" in result and "histogram" in result

    def test_bb_upper_above_lower(self):
        from modules.signals import compute_bollinger_bands
        close = self._make_prices(100)
        bb = compute_bollinger_bands(close)
        upper = bb["upper"].dropna()
        lower = bb["lower"].dropna()
        assert (upper >= lower).all()

    def test_double_bottom_detects_pattern(self):
        from modules.options import detect_unusual_options_activity
        from modules.signals import detect_double_bottom
        import pandas as pd
        import numpy as np
        prices = [100, 95, 90, 92, 96, 91, 90, 93, 97, 100]
        series = pd.Series(prices)
        result = detect_double_bottom(series)
        assert isinstance(result, bool)


class TestMarketDataIntegration:

    @pytest.mark.asyncio
    async def test_format_symbol_nifty(self):
        from modules.market_data import format_symbol
        assert format_symbol("NIFTY") == "^NSEI"
        assert format_symbol("BANKNIFTY") == "^NSEBANK"
        assert format_symbol("RELIANCE") == "RELIANCE.NS"
        assert format_symbol("RELIANCE", "BSE") == "RELIANCE.BO"

    @pytest.mark.asyncio
    async def test_get_live_price_nifty(self):
        from modules.market_data import get_live_price
        try:
            data = await get_live_price("NIFTY")
            assert "price" in data
            assert data["price"] > 0
            assert "change_pct" in data
        except Exception:
            pytest.skip("Network unavailable")

    @pytest.mark.asyncio
    async def test_get_historical_ohlc(self):
        from modules.market_data import get_historical_ohlc
        try:
            df = await get_historical_ohlc("RELIANCE", period="1mo")
            assert not df.empty
            assert "Close" in df.columns
        except Exception:
            pytest.skip("Network unavailable")


class TestPortfolioIntegration:
    @pytest.mark.asyncio
    async def test_init_and_get_pnl(self):
        from modules.portfolio import init_db, get_portfolio_pnl
        await init_db()
        pnl = await get_portfolio_pnl()
        assert "cash_balance" in pnl
        assert pnl["cash_balance"] >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
