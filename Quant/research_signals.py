"""Causal quantitative features for the Research Terminal.

Signals use current/past observations only. Trade simulations execute on the
NEXT trading day's open and never use future prices to create signals.
This module intentionally does not import Streamlit.
"""
from __future__ import annotations

import re
import numpy as np
import pandas as pd

DAYS = 252


def clean_ticker(value: str) -> str:
    ticker = value.strip().upper()
    if not re.fullmatch(r"[A-Z0-9^.\-=]{1,18}", ticker):
        raise ValueError("Ticker should contain letters, digits, ^, ., - or =.")
    return ticker


def normalize_prices(download: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Accept yfinance's flat or MultiIndex layout and return adjusted OHLCV."""
    if download is None or download.empty:
        raise ValueError(f"No historical prices returned for {ticker}.")
    d = download.copy()
    if isinstance(d.columns, pd.MultiIndex):
        if ticker in d.columns.get_level_values(0):
            d = d[ticker]
        elif ticker in d.columns.get_level_values(1):
            d = d.xs(ticker, axis=1, level=1)
        else:
            d.columns = d.columns.get_level_values(0)
    if "Close" not in d.columns:
        raise ValueError(f"No closing prices returned for {ticker}.")
    result = pd.DataFrame(index=pd.to_datetime(d.index).tz_localize(None)
                          if getattr(d.index, "tz", None) is not None
                          else pd.to_datetime(d.index))
    for key in ("Open", "High", "Low", "Close", "Volume"):
        if key in d:
            result[key] = pd.to_numeric(d[key], errors="coerce").to_numpy()
        else:
            result[key] = np.nan
    result = result[~result.index.duplicated(keep="last")].sort_index()
    result = result.dropna(subset=["Close"])
    if result.empty:
        raise ValueError(f"No valid closing prices returned for {ticker}.")
    return result


def prior_z(value: pd.Series, window: int) -> pd.Series:
    """Today versus trailing window observations ENDING YESTERDAY."""
    past = value.shift(1)
    mean = past.rolling(window, min_periods=window).mean()
    std = past.rolling(window, min_periods=window).std()
    return (value - mean) / std.replace(0, np.nan)


def _geometry(close: pd.Series, window: int) -> pd.DataFrame:
    """Quadratic trend on log prices, derivative evaluated at newest sample."""
    y = np.log(close.where(close > 0))
    velocity = np.full(len(y), np.nan)
    curvature = np.full(len(y), np.nan)
    inst_sharpe = np.full(len(y), np.nan)
    x = np.arange(window, dtype=float)
    for end in range(window - 1, len(y)):
        chunk = y.iloc[end - window + 1:end + 1].to_numpy(dtype=float)
        if not np.isfinite(chunk).all():
            continue
        quadratic, linear, _ = np.polyfit(x, chunk, 2)
        velocity[end] = linear + 2 * quadratic * (window - 1)
        curvature[end] = 2 * quadratic
        vol = np.diff(chunk).std(ddof=1)
        if vol > 0:
            inst_sharpe[end] = velocity[end] / vol
    return pd.DataFrame({"Velocity": velocity, "Curvature": curvature,
                         "InstantaneousSharpe": inst_sharpe}, index=close.index)


def compute_features(
    prices: pd.DataFrame,
    market: pd.DataFrame | None = None,
    sector: pd.DataFrame | None = None,
    sharpe_window: int = 60,
    normal_window: int = 252,
    geometry_window: int = 20,
    risk_free_annual: float = 0.0,
    sigma: float = 2.0,
) -> pd.DataFrame:
    """Compute signals without any future-dependent features."""
    if sharpe_window < 15 or normal_window < 60 or geometry_window < 5:
        raise ValueError("Window sizes are too short.")
    d = prices.copy().sort_index()
    d["Return"] = d["Close"].pct_change(fill_method=None)
    daily_rf = risk_free_annual / DAYS
    rolling = d["Return"].rolling(sharpe_window, min_periods=sharpe_window)
    d["Sharpe"] = ((rolling.mean() - daily_rf) /
                   rolling.std().replace(0, np.nan)) * np.sqrt(DAYS)

    prior_sharpe = d["Sharpe"].shift(1)
    d["SharpeMean"] = prior_sharpe.rolling(normal_window, min_periods=normal_window).mean()
    d["SharpeStd"] = prior_sharpe.rolling(normal_window, min_periods=normal_window).std()
    d["SharpeZ"] = (d["Sharpe"] - d["SharpeMean"]) / d["SharpeStd"].replace(0, np.nan)
    d["SharpeUpper"] = d["SharpeMean"] + sigma * d["SharpeStd"]
    d["SharpeLower"] = d["SharpeMean"] - sigma * d["SharpeStd"]

    d["ReturnZ"] = prior_z(d["Return"], 60)
    d["VolumeZ"] = prior_z(d["Volume"], 60)
    d["Return20"] = d["Close"].pct_change(20, fill_method=None)
    d["Return20Z"] = prior_z(d["Return20"], normal_window)
    d["Vol20"] = d["Return"].rolling(20).std() * np.sqrt(DAYS)
    d["Vol100"] = d["Return"].rolling(100).std() * np.sqrt(DAYS)
    d["VolSpike"] = d["Vol20"] > 2 * d["Vol100"]
    d["Drawdown52w"] = d["Close"] / d["Close"].rolling(252, min_periods=30).max() - 1
    d["MA50"] = d["Close"].rolling(50).mean()
    d["MA200"] = d["Close"].rolling(200).mean()

    # Keep the three original Falling_Knife panic components, but with a
    # trailing (not centered) cluster count to avoid look-ahead.
    d["ReturnPanic"] = d["ReturnZ"] < -3
    d["VolumePanic"] = d["VolumeZ"] > 3
    d["VolatilityPanic"] = d["VolSpike"].fillna(False)
    d["PanicRaw"] = (d["ReturnPanic"].astype(int) +
                     d["VolumePanic"].astype(int) +
                     d["VolatilityPanic"].astype(int))
    d["PanicDays6"] = (d["PanicRaw"] > 0).astype(int).rolling(6, min_periods=1).sum()
    d["PanicSeverity"] = 0
    d.loc[d["PanicRaw"] > 0, "PanicSeverity"] = 1
    d.loc[(d["PanicRaw"] >= 2) | (d["PanicDays6"] >= 2), "PanicSeverity"] = 2
    d.loc[(d["PanicRaw"] == 3) & (d["PanicDays6"] >= 2), "PanicSeverity"] = 3
    d["Capitulation"] = (d["ReturnPanic"] &
                         (d["VolumePanic"] | d["VolatilityPanic"]) &
                         (d["PanicDays6"] >= 2))

    delta = d["Close"].diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    d["RSI14"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))

    geom = _geometry(d["Close"], geometry_window)
    for col in geom:
        d[col] = geom[col]
    d["GeometryRegime"] = np.select(
        [(d["Velocity"] > 0) & (d["Curvature"] > 0),
         (d["Velocity"] > 0) & (d["Curvature"] <= 0),
         (d["Velocity"] <= 0) & (d["Curvature"] < 0)],
        ["Up, accelerating", "Up, decelerating", "Down, accelerating"],
        default="Down, decelerating",
    )
    d.loc[d["Velocity"].isna(), "GeometryRegime"] = "Unknown"

    # Lagged rolling beta makes today's stock-specific residual use only
    # information known before today's close.
    for name, reference in (("Market", market), ("Sector", sector)):
        d[f"{name}Relative20"] = np.nan
        d[f"{name}Residual20Z"] = np.nan
        if reference is None or reference.empty:
            continue
        ref_close = reference["Close"].reindex(d.index).ffill()
        ref_ret = ref_close.pct_change(fill_method=None)
        beta = (d["Return"].rolling(120, min_periods=120).cov(ref_ret) /
                ref_ret.rolling(120, min_periods=120).var().replace(0, np.nan)).shift(1)
        residual = d["Return"] - beta * ref_ret
        residual20 = residual.rolling(20, min_periods=20).sum()
        d[f"{name}Residual20Z"] = prior_z(residual20, normal_window)
        d[f"{name}Relative20"] = d["Return20"] - ref_close.pct_change(20, fill_method=None)

    d["EntryCross"] = (d["SharpeZ"] <= -sigma) & (d["SharpeZ"].shift(1) > -sigma)
    d["ExitCross"] = (d["SharpeZ"] >= sigma) & (d["SharpeZ"].shift(1) < sigma)
    d["Stabilizing"] = (d["Curvature"] > 0) & (d["Close"].pct_change(5) > 0)
    return d


def backtest(
    df: pd.DataFrame,
    entry_z: float = -2.0,
    exit_z: float = 1.0,
    max_hold: int = 120,
    cost_bps_per_side: float = 10.0,
) -> pd.DataFrame:
    """Long-only non-overlapping signal trades; next-open execution.

    Does not apply fair-value filtering: today's manually entered valuation
    would be a future-known input if applied to historical trades.
    """
    trades = []
    in_position = False
    entry_price = 0.0
    entry_idx = -1
    entry_signal_idx = -1
    for i in range(1, len(df) - 1):
        z = df["SharpeZ"].iloc[i]
        prev = df["SharpeZ"].iloc[i - 1]
        if not in_position:
            if pd.notna(z) and pd.notna(prev) and prev > entry_z and z <= entry_z:
                next_open = df["Open"].iloc[i + 1]
                if pd.notna(next_open) and next_open > 0:
                    entry_price = float(next_open)
                    entry_idx = i + 1
                    entry_signal_idx = i
                    in_position = True
            continue
        hold = i - entry_idx + 1
        exit_signal = pd.notna(z) and z >= exit_z
        if not exit_signal and hold < max_hold:
            continue
        next_open = df["Open"].iloc[i + 1]
        if pd.isna(next_open) or next_open <= 0:
            continue
        gross = float(next_open) / entry_price - 1
        per_side = cost_bps_per_side / 10000
        net = (1 - per_side) * (1 + gross) * (1 - per_side) - 1
        trades.append({
            "Signal date": df.index[entry_signal_idx].date(),
            "Entry date": df.index[entry_idx].date(),
            "Exit date": df.index[i + 1].date(),
            "Entry price": entry_price,
            "Exit price": float(next_open),
            "Holding days": (i + 1) - entry_idx,
            "Exit reason": "Sharpe recovery" if exit_signal else "Max hold",
            "Return %": net * 100,
        })
        in_position = False
    return pd.DataFrame(trades)


def event_study(df: pd.DataFrame, entry_z: float = -2.0,
                horizons: tuple[int, ...] = (20, 60, 120)) -> pd.DataFrame:
    """Descriptive forward returns from the NEXT day's open; not a strategy."""
    z = df["SharpeZ"]
    events = (z <= entry_z) & (z.shift(1) > entry_z)
    entries = np.flatnonzero(events.to_numpy())
    rows = []
    for i in entries:
        if i + 1 >= len(df):
            continue
        start = df["Open"].iloc[i + 1]
        if pd.isna(start) or start <= 0:
            continue
        row = {"Signal date": df.index[i].date()}
        for h in horizons:
            pos = i + h
            price = df["Close"].iloc[pos] if pos < len(df) else np.nan
            row[f"{h}d forward %"] = (price / start - 1) * 100 if pd.notna(price) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def breadth(prices: pd.DataFrame) -> pd.DataFrame:
    """Prices: columns are tickers, rows are dates."""
    ret = prices.sort_index().pct_change(fill_method=None)
    advances = (ret > 0).sum(axis=1)
    declines = (ret < 0).sum(axis=1)
    valid = ret.notna().sum(axis=1)
    d = pd.DataFrame(index=prices.index)
    d["Advances"] = advances
    d["Declines"] = declines
    d["AD Diff"] = advances - declines
    d["AD Ratio"] = (advances - declines) / valid.replace(0, np.nan)
    d["BreadthVol20"] = d["AD Ratio"].rolling(20).std()
    d["BreadthVolZ"] = prior_z(d["BreadthVol20"], 120)
    d.loc[valid < 2, ["AD Diff", "AD Ratio", "BreadthVol20", "BreadthVolZ"]] = np.nan
    return d
