# Mastertrades

A trading analytics and volatility-scanning web app built from the Mastertrades Python project. Uses ML models to predict volatile days for ETFs/stocks and generate 0DTE options trading signals (GO_JACKPOT, GO_ULTRA_JACKPOT, GO_HOT, SKIP).

## Run & Operate

- `cd mastertrades && streamlit run app.py --server.port 5000 --server.enableCORS false --server.enableXsrfProtection false` — run the Streamlit app
- Or use the "Mastertrades" workflow (auto-configured)

## Stack

- Python 3.11, Streamlit
- Data: yfinance (Yahoo Finance OHLCV)
- ML: scikit-learn (Logistic Regression, Gradient Boosting)
- Analytics: pandas, numpy, scipy
- Model cache: joblib

## Where things live

- `mastertrades/app.py` — main Streamlit app (4 pages)
- `mastertrades/src/` — Python analytics engine (copied from Mastertrades repo)
- `mastertrades/data/` — cached OHLCV CSV files (gitignored)
- `mastertrades/models/` — cached joblib ML models (gitignored)
- `mastertrades/.streamlit/config.toml` — Streamlit server config

## Pages

1. **Command Center** — Today's jackpot signals for SPY/QQQ/IWM/AAPL with hero verdict card and trade tickets
2. **Scanner** — Multi-ticker volatility scanner across 12+ tickers, ranked by P(volatile day)
3. **Account Tracker** — Log trades, track equity curve and milestones ($500→$5k→$50k→$500k)
4. **Weekday Patterns** — Volatility analysis by day of week (most volatile vs flattest)

## Architecture decisions

- All Python: Streamlit wraps the existing Mastertrades src/ modules directly
- Models are cached in `models/` with 7-day expiry; data cached in `data/` with 15-min expiry
- `@st.cache_data(ttl=...)` used throughout for in-process caching
- CORS/XsrfProtection disabled to work behind Replit's proxy
- Working directory is `mastertrades/` so `from src.X import Y` resolves correctly

## Signals

- **GO_ULTRA_JACKPOT** 🌟 — Both vol + P&L classifiers fire at highest confidence (ULTRA tier)
- **GO_JACKPOT** ✅ — Both vol + direct-P&L classifiers fire (trade day)
- **GO_HOT** 🔥 — Vol classifier fires alone (elevated, lower confidence)
- **SKIP** ⏭ — Calm expected, skip trading

## User preferences

- Dark theme matching the original Command Center HTML aesthetic
- Inline HTML cards for signal/verdict displays
- Sidebar navigation with refresh button

## Data Sources

- **Polygon.io** (primary) — exchange-quality adjusted daily OHLCV via `/v2/aggs/ticker/{T}/range/1/day/...`
  - Requires `POLYGON_API_KEY` secret (set in Replit Secrets)
  - Used for all historical OHLCV bars fed into ML models and gap analysis
  - Real-time snapshots/intraday require a higher Polygon plan (403 on free/starter)
- **Yahoo Finance** (fallback) — used automatically when Polygon key is absent or API fails
- Source priority handled in `src/scanner.py:fetch_or_load_daily()` — no changes needed elsewhere
- Polygon module: `mastertrades/src/polygon_feed.py`

## Gotchas

- First run trains ML models (~60s per ticker) — subsequent runs use cached models
- Yahoo Finance data is 15-min delayed — labelled as "delayed" in live quotes
- The `from src.X import Y` imports only work if CWD is `mastertrades/`
- Streamlit must be started with `--server.enableCORS false --server.enableXsrfProtection false` in Replit

## Pointers

- Source repo: https://github.com/buywitheze-droid/Mastertrades
- See the `pnpm-workspace` skill for workspace structure
