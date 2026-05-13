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

Sidebar is grouped into 4 sections:

**⚡ TRADE NOW**
1. **Today's Plays** — Unified ranked list of every actionable signal from every validated source. Strict edge gate (avg_ret > 0, win_rate ≥ 50%, n ≥ 3). Single normalized edge score (`win_rate × avg_ret × confidence × sqrt(n/(n+5))`) makes signals across MA Bounce / ML Jackpot / Gap Fill / 0DTE Drop directly comparable. Source-health banner surfaces partial failures so "no plays" never hides a broken feed.

**🎯 SIGNAL DETAILS** (drill-downs for the unified list)
2. **Command Center** — ML jackpot signals for SPY/QQQ/IWM/AAPL with hero verdict card
3. **MA Bounce Setups** — 24 high-edge weekly MA-touch plays (universe-scanned)
4. **Gap Reversal** — Gap fill & reversal setups (WATCH_FILL ≥70% fill rate is the strong signal; NEAR_FILL ≥50% is moderate)
5. **0DTE Lottery** — 1000%+ options plays & sweet spots

**📊 RESEARCH** (analysis-only, not direct trade signals)
6. **Scanner** — Ranked vol universe across 12+ tickers
7. **Weekly MAs** — Per-ticker MA + order flow drill-down
8. **Reversal Levels** — Intraday low/high reversal zones
9. **Weekday Patterns** — Volatility analysis by day of week

**💰 ACCOUNT**
10. **Account Tracker** — Log trades, track equity curve and milestones ($500→$5k→$50k→$500k)

### MA Bounce Setups (universe scan)

- 24 (ticker, MA) pairs validated on 1 yr daily data: ≥5 touches AND ≥75% positive 5-day return
- Top setups: AVGO 50w EMA (100% / +12.21%), GOOGL 50w EMA (100% / +4.42%), GOOGL 30w EMA (100% / +4.18%), XLI 30w SMA/EMA (100% / +3.77%), SHOP 30w SMA (90% / +4.39%), INTC 30w SMA (80% / +5.02%)
- Live state per setup: TOUCHING (within ±0.6%), APPROACHING from above (within 2.5% + falling), EXTENDED, BELOW
- Frozen results in `mastertrades/src/ma_setups_universe.py` with `SCAN_DATE` constant; live distances cached 10 min
- Real-options validation (AMZN/SPY/QQQ/NVDA, weekly 5-day calls): +243% on $39k over 39 trades, 62% win rate

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
