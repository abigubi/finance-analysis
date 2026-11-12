# Quant Research Terminal

An interactive Streamlit research website built from the existing Quant scripts in this repository.

## Start locally

Install dependencies from the repository root:

    pip install -r requirements.txt

Run:

    streamlit run Quant/Research_Terminal.py

Choose any Yahoo Finance equity ticker in the left sidebar. Data is daily historical
adjusted OHLCV, not live execution/quote data. Yahoo Finance can throttle large
scans. Calculations are cached for an hour.

## Deploy as a website

In Streamlit Community Cloud, select the repository, choose the branch to deploy
(main after merging), and set **Main file path** to:

    Quant/Research_Terminal.py

The old Falling_Knife_Web.py application stays intact; existing deployments are
not overwritten. Change the Streamlit entry point to serve the new terminal or
create a second app. Root requirements.txt already contains Streamlit, yfinance,
NumPy, pandas and Plotly.

## Features

- **Light/dark** mode, selected in the sidebar. The Sharpe visualization reuses
  the filled translucent ±σ bands, mean, Sharpe and price layout from sharpe_chart.py.
- **Single ticker:** 30/45/60/90/120-session Sharpe; 126/252/504 prior-observation
  z-score baselines; configurable ±σ threshold; adjustable fixed risk-free rate.
- **Falling knife:** return, volume and volatility panic flags; causal six-day
  cluster intensity; latest price and 52-week drawdown.
- **Relative underperformance:** raw 20-day return z-score and market-/sector-
  adjusted residual 20-day z-scores, with historical lagged 120-day beta.
- **Trend geometry:** local end-of-window price slope and curvature.
- **Backtest:** next-adjusted-open entry/exit, one position at a time, specified
  per-side costs (entered as total round trip), max holding window, test period,
  trade log, export and 20/60/120-day event studies against simpler alternatives.
- **Fundamentals:** user-entered fair value and margin of safety, checklist and
  optional Yahoo metadata with explicit source limitations.
- **Watchlist scanner:** up to 15 tickers, CSV export, watchlist-only
  advance–decline breadth and breadth volatility z-score.

## Correctness / caveats

The original Falling_Knife.py and Falling_Knife_Web.py scripts used a centered
11-day panic window with 5 future dates. For historically actionable signals,
this version instead looks back over the previous six trading days. All rolling
z-score reference windows exclude the current day, and signals execute at the
NEXT trading day's adjusted open.

The event study reports descriptive outcomes that can overlap. The backtest is
NOT a full daily equity curve/CAGR: it compounds only completed trade returns,
assumes zero returns on idle cash, omits outstanding trades and uses simple
fill/cost assumptions. Historical fundamentals are not point-in-time, so
the manually entered fair value is deliberately not backfilled into trades.
Results do not establish statistical significance, profitability, or alpha.

These indicators are primarily a way to rank and investigate fundamentally
researched potential falling knives.

## Tests

    python -m unittest discover -s tests -p test_research_signals.py

Tests cover causal invariance under changes to future prices, prior-only Sharpe
reference distributions, next-open execution, price normalization and breadth.
