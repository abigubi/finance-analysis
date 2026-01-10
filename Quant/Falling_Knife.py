import argparse
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf


def download_price_data(ticker: str, start: str) -> pd.DataFrame:
    data = yf.download(
        ticker,
        start=start,
        progress=False,
        auto_adjust=False,  # keep the classic OHLCV columns with Adj Close
    )
    if data.empty:
        raise ValueError(f"No price data returned for ticker '{ticker}'.")
    return data


def select_price_series(prices: pd.DataFrame, ticker: str) -> pd.Series:
    if isinstance(prices.columns, pd.MultiIndex):
        for candidate in ("Adj Close", "Close"):
            try:
                series = prices.xs(candidate, axis=1, level=0)
            except KeyError:
                continue
            if isinstance(series, pd.DataFrame):
                if series.shape[1] == 1:
                    return series.iloc[:, 0]
                if ticker.upper() in series.columns:
                    return series[ticker.upper()]
                # If ticker not found, return first column
                return series.iloc[:, 0]
            else:
                return series
    else:
        for candidate in ("Adj Close", "Close"):
            if candidate in prices.columns:
                return prices[candidate]
    raise KeyError(
        "Could not find an adjusted or close price column in the downloaded data."
    )


def select_volume_series(prices: pd.DataFrame, ticker: str) -> pd.Series:
    if isinstance(prices.columns, pd.MultiIndex):
        try:
            series = prices.xs("Volume", axis=1, level=0)
        except KeyError:
            raise KeyError("Could not find Volume column in the downloaded data.")
        if isinstance(series, pd.DataFrame):
            if series.shape[1] == 1:
                return series.iloc[:, 0]
            if ticker.upper() in series.columns:
                return series[ticker.upper()]
            # If ticker not found, return first column
            return series.iloc[:, 0]
        else:
            return series
    else:
        if "Volume" in prices.columns:
            return prices["Volume"]
    raise KeyError("Could not find Volume column in the downloaded data.")


def calculate_panic_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate panic signals with clustering and severity scoring for swing-worthy bottoms."""
    # Daily return
    df["ret"] = df["Adj Close"].pct_change()
    
    # 1) RETURN Z-SCORE MODEL
    ret_mean = df["ret"].rolling(60).mean()
    ret_std = df["ret"].rolling(60).std()
    # Avoid division by zero
    df["z_return"] = np.where(
        ret_std != 0,
        (df["ret"] - ret_mean) / ret_std,
        np.nan
    )
    
    # 2) VOLATILITY SPIKE MODEL
    df["vol20"] = df["ret"].rolling(20).std() * np.sqrt(252)
    df["vol100"] = df["ret"].rolling(100).std() * np.sqrt(252)
    # Avoid division by zero
    df["vol_spike"] = np.where(
        df["vol100"] != 0,
        df["vol20"] > 2 * df["vol100"],
        False
    )
    
    # 3) VOLUME SPIKE
    vol_mean = df["Volume"].rolling(60).mean()
    vol_std = df["Volume"].rolling(60).std()
    # Avoid division by zero
    df["z_volume"] = np.where(
        vol_std != 0,
        (df["Volume"] - vol_mean) / vol_std,
        np.nan
    )
    
    df["volume_spike"] = df["z_volume"] > 3
    
    # Individual panic type flags
    df["return_panic"] = df["z_return"] < -3
    df["volatility_panic"] = df["vol_spike"]
    df["volume_panic"] = df["volume_spike"]
    
    # Raw panic score (0-3)
    df["raw_panic_score"] = (
        df["return_panic"].astype(int) +
        df["volatility_panic"].astype(int) +
        df["volume_panic"].astype(int)
    )
    
    # PANIC CLUSTERING: Count panics within 5 trading days (rolling window)
    # This identifies multi-day capitulation events
    # Use a rolling sum with a window of 11 days (5 days back + current + 5 days forward)
    # Then shift by -5 to center the window on the current day
    df["panic_cluster_count"] = (
        df["raw_panic_score"]
        .rolling(window=11, center=True, min_periods=1)
        .sum()
        .fillna(0)
    )
    
    # SEVERITY SCORING (0-3)
    # Severity 0: No panic or single-day noise
    # Severity 1: Mild panic (raw score 1, no cluster)
    # Severity 2: Serious panic (raw score 2+ OR cluster >= 2 days)
    # Severity 3: Full capitulation (all 3 conditions + cluster >= 2 days)
    df["panic_severity"] = 0
    
    # Severity 1: Single panic type, no clustering
    df.loc[
        (df["raw_panic_score"] == 1) & (df["panic_cluster_count"] <= 1),
        "panic_severity"
    ] = 1
    
    # Severity 2: Multiple panic types OR clustering (2+ days of panic)
    df.loc[
        ((df["raw_panic_score"] >= 2) | (df["panic_cluster_count"] >= 2)) &
        (df["panic_severity"] == 0),
        "panic_severity"
    ] = 2
    
    # Severity 3: All three conditions + clustering (full capitulation)
    df.loc[
        (df["return_panic"] & df["volatility_panic"] & df["volume_panic"]) &
        (df["panic_cluster_count"] >= 2),
        "panic_severity"
    ] = 3
    
    # LONG-TERM BOTTOM DETECTOR
    # Triggers when:
    # - Return Z < -3 (required)
    # - AND (Volume Z > 3 OR Volatility spike)
    # - AND panic lasts >= 2 days (cluster)
    df["long_term_bottom"] = (
        df["return_panic"] &
        (df["volume_panic"] | df["volatility_panic"]) &
        (df["panic_cluster_count"] >= 2) &
        (df["panic_severity"] >= 2)
    )
    
    # Only show severity >= 2 (serious panics and capitulation)
    df["panic_signal"] = df["panic_severity"] >= 2
    
    # Calculate forward returns for backtesting
    df["return_30d"] = df["Adj Close"].pct_change(30).shift(-30) * 100
    df["return_60d"] = df["Adj Close"].pct_change(60).shift(-60) * 100
    df["return_120d"] = df["Adj Close"].pct_change(120).shift(-120) * 100
    
    return df


def identify_panic_clusters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group panic signals into clusters and identify cluster centers.
    Only cluster centers are shown on the chart to reduce clutter.
    """
    df = df.copy()
    df["cluster_id"] = 0
    df["is_cluster_center"] = False
    df["cluster_severity"] = 0
    df["cluster_start"] = None
    df["cluster_end"] = None
    
    panic_dates = df[df["panic_signal"]].index
    if len(panic_dates) == 0:
        return df
    
    # Group consecutive or nearby panic dates into clusters
    clusters = []
    current_cluster = []
    cluster_gap_days = 5  # Max gap between panic days to be in same cluster
    
    for i, date in enumerate(panic_dates):
        if len(current_cluster) == 0:
            current_cluster.append(date)
        else:
            days_gap = (date - current_cluster[-1]).days
            if days_gap <= cluster_gap_days:
                current_cluster.append(date)
            else:
                # Start new cluster
                if len(current_cluster) > 0:
                    clusters.append(current_cluster)
                current_cluster = [date]
    
    # Don't forget the last cluster
    if len(current_cluster) > 0:
        clusters.append(current_cluster)
    
    # Process each cluster
    cluster_num = 1
    for cluster in clusters:
        if len(cluster) == 0:
            continue
        
        cluster_dates = pd.DatetimeIndex(cluster)
        cluster_df = df.loc[cluster_dates]
        
        # Find cluster center (date with lowest price or highest severity)
        cluster_center = cluster_df.loc[cluster_df["Adj Close"].idxmin()]
        center_date = cluster_center.name
        
        # Get max severity in cluster
        max_severity = cluster_df["panic_severity"].max()
        is_long_term_bottom = cluster_df["long_term_bottom"].any()
        
        # Mark all dates in cluster
        df.loc[cluster_dates, "cluster_id"] = cluster_num
        df.loc[center_date, "is_cluster_center"] = True
        df.loc[center_date, "cluster_severity"] = max_severity
        df.loc[center_date, "cluster_start"] = cluster_dates.min()
        df.loc[center_date, "cluster_end"] = cluster_dates.max()
        
        # If it's a long-term bottom, mark the center
        if is_long_term_bottom:
            df.loc[center_date, "long_term_bottom"] = True
        
        cluster_num += 1
    
    return df


def plot_panic_detector(ticker: str, df: pd.DataFrame) -> None:
    """Plot clean, institutional-level panic detector - only cluster centers shown."""
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Price line (yellow/gold)
    ax.plot(df.index, df["Adj Close"], color="yellow", linewidth=2, label="Price", alpha=0.9)
    
    # Get only cluster centers (clean, no clutter)
    cluster_centers = df[df["is_cluster_center"]].copy()
    
    if len(cluster_centers) > 0:
        # Add background shading for severity zones (no vertical lines)
        for _, row in cluster_centers.iterrows():
            if pd.notna(row["cluster_start"]) and pd.notna(row["cluster_end"]):
                start = row["cluster_start"]
                end = row["cluster_end"]
                severity = row["cluster_severity"]
                
                # Color-coded shading by severity
                if severity == 3:
                    color = "red"
                    alpha = 0.2  # Deep red for severity 3
                elif severity == 2:
                    color = "orange"
                    alpha = 0.12  # Light red/orange for severity 2
                else:
                    color = "orange"
                    alpha = 0.08
                
                # Shade the entire cluster period
                ax.axvspan(start, end, color=color, alpha=alpha, zorder=0)
        
        # Plot markers by type (clean shapes, no text)
        long_term_bottoms = cluster_centers[cluster_centers["long_term_bottom"]]
        severity_3 = cluster_centers[(cluster_centers["cluster_severity"] == 3) & ~cluster_centers["long_term_bottom"]]
        severity_2 = cluster_centers[(cluster_centers["cluster_severity"] == 2) & ~cluster_centers["long_term_bottom"]]
        
        # 1. Long-Term Bottom Candidates - White star ⭐ (strongest signal)
        if len(long_term_bottoms) > 0:
            ax.scatter(
                long_term_bottoms.index,
                long_term_bottoms["Adj Close"],
                color="white",
                s=250,
                marker="*",
                label="Strong Bottom ⭐",
                zorder=7,
                edgecolors='red',
                linewidths=2
            )
        
        # 2. Severity 3 (Full Capitulation) - Red circle ●
        if len(severity_3) > 0:
            ax.scatter(
                severity_3.index,
                severity_3["Adj Close"],
                color="red",
                s=180,
                marker="o",
                label="Severity 3 ●",
                zorder=6,
                edgecolors='white',
                linewidths=1.5
            )
        
        # 3. Severity 2 (Serious Panic) - Orange triangle ▲
        if len(severity_2) > 0:
            ax.scatter(
                severity_2.index,
                severity_2["Adj Close"],
                color="orange",
                s=150,
                marker="^",
                label="Severity 2 ▲",
                zorder=5,
                edgecolors='white',
                linewidths=1
            )
    
    # Labels and styling
    ax.set_title(
        f"{ticker.upper()} – Long-Term Bottom Detector",
        fontsize=18,
        fontweight='bold',
        pad=20
    )
    ax.set_ylabel("Price", color="yellow", fontsize=12, fontweight='bold')
    ax.tick_params(axis="y", colors="yellow")
    ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
    ax.legend(loc="upper left", fontsize=11, framealpha=0.9, ncol=1)
    
    plt.tight_layout()
    plt.show()


def parse_args() -> Tuple[Optional[str], str]:
    parser = argparse.ArgumentParser(
        description="Plot panic detector with falling knife model."
    )
    parser.add_argument(
        "ticker",
        type=str,
        nargs="?",
        help="Ticker symbol to download; if omitted you will be prompted.",
    )
    parser.add_argument(
        "--start",
        type=str,
        default="2015-01-01",
        help="Start date for historical data (YYYY-MM-DD, default: 2015-01-01)",
    )
    args = parser.parse_args()
    return args.ticker, args.start


def main() -> None:
    ticker, start = parse_args()
    if not ticker:
        ticker = input("Enter ticker symbol: ").strip().upper()
        if not ticker:
            raise ValueError("Ticker symbol is required.")
    else:
        ticker = ticker.upper()
    
    # Download data
    prices = download_price_data(ticker, start)
    prices.attrs["ticker"] = ticker
    
    # Select price and volume series
    price_series = select_price_series(prices, ticker)
    volume_series = select_volume_series(prices, ticker)
    
    # Create dataframe with required columns
    df = pd.DataFrame(index=prices.index)
    df["Adj Close"] = price_series
    df["Volume"] = volume_series
    
    # Calculate panic signals
    df = calculate_panic_signals(df)
    
    # Drop rows with NaN values
    df = df.dropna(subset=["ret", "z_return", "vol20", "vol100", "z_volume"])

    
    if df.empty:
        raise ValueError(
            "Not enough data to compute panic signals. Try an earlier start date."
        )

    
    # Identify panic clusters (groups nearby panics together)
    df = identify_panic_clusters(df)
    
    # Plot
    plot_panic_detector(ticker, df)
    
    # Print cluster information (clean, professional output)
    cluster_centers = df[df["is_cluster_center"]].copy()
    if len(cluster_centers) > 0:
        print("\n" + "="*80)
        print("Long-Term Bottom Detector - Panic Clusters (Clean Signals Only)")
        print("="*80)
        
        # Create readable summary
        def get_signal_type(row):
            if row["long_term_bottom"]:
                return "⭐ STRONG BOTTOM"
            elif row["cluster_severity"] == 3:
                return "Severity 3 ●"
            else:
                return "Severity 2 ▲"
        
        cluster_centers["Signal_Type"] = cluster_centers.apply(get_signal_type, axis=1)
        cluster_centers["Cluster_Duration"] = cluster_centers.apply(
            lambda row: (row["cluster_end"] - row["cluster_start"]).days + 1 
            if pd.notna(row["cluster_start"]) and pd.notna(row["cluster_end"]) 
            else 1, axis=1
        )
        
        # Display summary - show most recent first
        display_cols = ["Adj Close", "cluster_severity", "Cluster_Duration", "Signal_Type"]
        print("\nPanic Clusters (Most Recent First):")
        print(cluster_centers[display_cols].tail(10).sort_index(ascending=False).to_string())
        
        # Summary statistics
        print("\n" + "-"*80)
        print("Summary Statistics:")
        print(f"  Total Panic Clusters:            {len(cluster_centers)}")
        print(f"  Severity 2 Clusters ▲:            {(cluster_centers['cluster_severity'] == 2).sum()}")
        print(f"  Severity 3 Clusters ●:            {(cluster_centers['cluster_severity'] == 3).sum()}")
        print(f"  ⭐ Strong Bottom Candidates:     {cluster_centers['long_term_bottom'].sum()}")
        avg_duration = cluster_centers["Cluster_Duration"].mean()
        print(f"  Average Cluster Duration:        {avg_duration:.1f} days")
        print("="*80)
        
        # Highlight long-term bottoms
        long_term_bottoms = cluster_centers[cluster_centers["long_term_bottom"]]
        if len(long_term_bottoms) > 0:
            print("\n" + "🔥"*40)
            print("STRONG BOTTOM CANDIDATES (Best Swing Entry Points):")
            print("🔥"*40)
            for date in long_term_bottoms.tail(10).sort_index(ascending=False).index:
                row = long_term_bottoms.loc[date]
                duration = (row["cluster_end"] - row["cluster_start"]).days + 1 if pd.notna(row["cluster_start"]) and pd.notna(row["cluster_end"]) else 1
                print(f"\n  Date: {date.strftime('%Y-%m-%d')}")
                print(f"  Price: ${row['Adj Close']:.2f}")
                print(f"  Severity: {int(row['cluster_severity'])}")
                print(f"  Cluster Duration: {int(duration)} days")
                print(f"  Return Z-Score: {row['z_return']:.2f}")
    else:
        print("\n" + "="*80)
        print("No panic clusters detected (Severity ≥ 2)")
        print("This means no serious multi-day capitulation events found.")
        print("="*80)


if __name__ == "__main__":
    main()
