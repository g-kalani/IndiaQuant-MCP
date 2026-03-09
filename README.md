# IndiaQuant MCP

An MCP server that connects Claude to live Indian stock market data.

Once connected, you can ask Claude things like:
- "Should I buy HDFC Bank right now?"
- "What's the max pain for Nifty this expiry?"
- "Find me oversold IT stocks"
- "Buy 50 shares of Infosys with a stop loss at 1500"

All 10 tools return live NSE/BSE data.

---
<img width="1920" height="1020" alt="image" src="https://github.com/user-attachments/assets/12f25879-506a-4d86-92ae-9882f75ba9a0" />
<img width="1920" height="1021" alt="image" src="https://github.com/user-attachments/assets/882c6a61-e8b7-4390-9eda-a17d310ca0ed" />
<img width="1920" height="1021" alt="image" src="https://github.com/user-attachments/assets/e32832f2-1b75-4353-b006-7115f9996fd1" />

## What's inside

Five modules, ten tools, zero paid APIs.

```
indiaquant_mcp/
├── server.py               # MCP server, all 10 tools live here
├── modules/
│   ├── market_data.py      # yfinance wrapper, caching, symbol formatting
│   ├── signals.py          # RSI, MACD, Bollinger Bands, pattern detection
│   ├── options.py          # Black-Scholes Greeks, options chain, max pain
│   ├── portfolio.py        # SQLite virtual portfolio, P&L, VaR
│   └── sentiment.py        # NewsAPI + Alpha Vantage sentiment scoring
├── tests/
│   └── test_suite.py
├── .env.example
├── generate_claude_config.py
└── requirements.txt
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

If you're on Python 3.14, pandas-ta's numba dependency won't build. Work around it:
```bash
pip install pandas-ta --no-deps
pip install pandas numpy
```

### 2. API keys

Copy `.env.example` to `.env` and fill in your keys:

```
NEWSAPI_KEY=...        # newsapi.org — free, 100 req/day
ALPHA_VANTAGE_KEY=...  # alphavantage.co — free, 25 req/day
```

These are optional. All tools except `analyze_sentiment` work without them.

### 3. Connect to Claude Desktop

Run the config helper:
```bash
python generate_claude_config.py
```

It prints the JSON to paste into your Claude Desktop config. The file location depends on how you installed Claude:

- **Direct download**: `%APPDATA%\Claude\claude_desktop_config.json`
- **Microsoft Store version**: `%LOCALAPPDATA%\Packages\Claude_pzs8sxrjxfjjc\LocalCache\Roaming\Claude\claude_desktop_config.json`
- **Mac**: `~/Library/Application Support/Claude/claude_desktop_config.json`

Add `PYTHONPATH` to the env block so Python can find the modules folder:

```json
{
  "mcpServers": {
    "indiaquant": {
      "command": "cmd",
      "args": ["/c", "C:\\Python314\\python.exe", "D:\\path\\to\\indiaquant_mcp\\server.py"],
      "env": {
        "NEWSAPI_KEY": "your_key",
        "ALPHA_VANTAGE_KEY": "your_key",
        "PYTHONPATH": "D:\\path\\to\\indiaquant_mcp"
      }
    }
  }
}
```

Restart Claude Desktop. You should see **indiaquant** appear in the `+` connectors menu with a blue toggle.

### 4. Test it

```
What's RELIANCE trading at?
```

If Claude calls the tool and returns a live price, everything is working.

---

## The 10 tools

| Tool | What it does |
|------|-------------|
| `get_live_price` | Live price, change%, volume for any NSE/BSE stock or index |
| `get_options_chain` | Full CE/PE chain with OI, volume, IV, and max pain |
| `analyze_sentiment` | Scores recent news headlines for a stock |
| `generate_signal` | BUY/SELL/HOLD with confidence % combining technicals + sentiment |
| `get_portfolio_pnl` | Live P&L on your virtual positions |
| `place_virtual_trade` | Execute a virtual trade at current market price |
| `calculate_greeks` | Delta, Gamma, Theta, Vega via Black-Scholes |
| `detect_unusual_activity` | OI spikes and abnormal Vol/OI ratios in options |
| `scan_market` | Screen Nifty 50 by RSI, sector, change%, volume |
| `get_sector_heatmap` | Live sector performance across all Nifty sectors |

---

## How the signal works

`generate_signal` combines five inputs into a single composite score:

| Input | Weight | Logic |
|-------|--------|-------|
| RSI | 25% | < 30 bullish, > 70 bearish |
| MACD | 25% | Histogram direction and magnitude |
| Bollinger Bands | 20% | Price position within the band |
| Chart patterns | 15% | Double top/bottom, head & shoulders |
| News sentiment | 15% | Aggregate score from headlines |

The composite runs from -1 to +1. Above +0.15 is BUY, below -0.15 is SELL, everything else is HOLD. Confidence is just `abs(composite) * 100`.

The 15% sentiment cap is intentional — news data is sparse and noisy for Indian stocks, so it shouldn't dominate.

---

## Black-Scholes implementation

The Greeks are calculated entirely from first principles in `modules/options.py`. No scipy, no statsmodels.

The normal CDF uses `math.erfc`:
```python
N(x) = erfc(-x / sqrt(2)) / 2
```

From there, d1/d2, then Delta, Gamma, Theta, Vega are all derived analytically. Implied volatility is solved via bisection (not Newton-Raphson — bisection is slower but won't blow up on deep ITM/OTM options where vega ≈ 0).

Put-call parity check: C - P = S - K·e^(-rT), verified to < 1e-10 error.

---

## Virtual portfolio

Starts with ₹10,00,000 cash. Trades execute at live market prices. Positions are stored in a local SQLite file (`portfolio.db`).

Risk scoring per position:

| Annualised volatility | Risk score |
|-----------------------|------------|
| < 10% | 2 |
| 10–20% | 4 |
| 20–30% | 6 |
| 30–40% | 8 |
| > 40% | 10 |

Daily VaR at 95% confidence: `position_value × daily_vol × 1.645`

---

## Design decisions worth noting

**Cache TTL is 60 seconds.** `scan_market` hits 50 stocks at once — without caching, Yahoo Finance rate-limits you immediately. 60 seconds is a reasonable balance between freshness and reliability.

**Bisection for IV, not Newton-Raphson.** NR converges faster but can fail for deep ITM/OTM options. Bisection always converges given a valid bracket.

**pandas-ta with manual fallbacks.** Every indicator has a pure-Python fallback in case pandas-ta fails to import (which happens on Python 3.14 due to the numba dependency). The server won't crash if the library isn't available.

**SQLite over anything else.** This is a single-user virtual trading tool. SQLite is zero-config and sufficient. If this were multi-user, PostgreSQL with proper transaction isolation would be the right call.

---

## Running tests

```bash
pip install pytest pytest-asyncio
python -m pytest tests/ -v
```

The unit tests cover Black-Scholes correctness (put-call parity, Greeks signs, IV roundtrip), max pain calculation, and unusual activity detection without needing a network connection.

---

## API limits

| API | Limit | Used for |
|-----|-------|----------|
| yfinance | Unlimited | Prices, OHLC, options chain |
| NewsAPI | 100 req/day | News headlines |
| Alpha Vantage | 25 req/day | Sentiment scores |

No broker account needed anywhere.
