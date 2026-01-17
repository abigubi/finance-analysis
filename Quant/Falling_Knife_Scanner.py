"""
Falling Knife Signal Scanner
Scans multiple tickers for recent panic signals and strong bottom opportunities.
Uses the same logic as Falling_Knife.py
"""

import sys
from datetime import datetime, timedelta
from typing import List, Dict
import pandas as pd
import numpy as np
import yfinance as yf


def download_price_data(ticker: str, start: str) -> pd.DataFrame:
    """Download price data from yfinance."""
    data = yf.download(
        ticker,
        start=start,
        progress=False,
        auto_adjust=False,
    )
    if data.empty:
        raise ValueError(f"No price data returned for ticker '{ticker}'.")
    return data


def select_price_series(prices: pd.DataFrame, ticker: str) -> pd.Series:
    """Select the appropriate price series from downloaded data."""
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
                return series.iloc[:, 0]
            else:
                return series
    else:
        for candidate in ("Adj Close", "Close"):
            if candidate in prices.columns:
                return prices[candidate]
    raise KeyError("Could not find an adjusted or close price column.")


def select_volume_series(prices: pd.DataFrame, ticker: str) -> pd.Series:
    """Select the appropriate volume series from downloaded data."""
    if isinstance(prices.columns, pd.MultiIndex):
        try:
            series = prices.xs("Volume", axis=1, level=0)
        except KeyError:
            raise KeyError("Could not find Volume column.")
        if isinstance(series, pd.DataFrame):
            if series.shape[1] == 1:
                return series.iloc[:, 0]
            if ticker.upper() in series.columns:
                return series[ticker.upper()]
            return series.iloc[:, 0]
        else:
            return series
    else:
        if "Volume" in prices.columns:
            return prices["Volume"]
    raise KeyError("Could not find Volume column.")


def calculate_panic_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate panic signals with clustering and severity scoring - same logic as Falling_Knife.py"""
    # Daily return
    df["ret"] = df["Adj Close"].pct_change()
    
    # 1) RETURN Z-SCORE MODEL
    ret_mean = df["ret"].rolling(60).mean()
    ret_std = df["ret"].rolling(60).std()
    df["z_return"] = np.where(
        ret_std != 0,
        (df["ret"] - ret_mean) / ret_std,
        np.nan
    )
    
    # 2) VOLATILITY SPIKE MODEL
    df["vol20"] = df["ret"].rolling(20).std() * np.sqrt(252)
    df["vol100"] = df["ret"].rolling(100).std() * np.sqrt(252)
    df["vol_spike"] = np.where(
        df["vol100"] != 0,
        df["vol20"] > 2 * df["vol100"],
        False
    )
    
    # 3) VOLUME SPIKE
    vol_mean = df["Volume"].rolling(60).mean()
    vol_std = df["Volume"].rolling(60).std()
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
    
    # PANIC CLUSTERING
    df["panic_cluster_count"] = (
        df["raw_panic_score"]
        .rolling(window=11, center=True, min_periods=1)
        .sum()
        .fillna(0)
    )
    
    # SEVERITY SCORING (0-3)
    df["panic_severity"] = 0
    
    df.loc[
        (df["raw_panic_score"] == 1) & (df["panic_cluster_count"] <= 1),
        "panic_severity"
    ] = 1
    
    df.loc[
        ((df["raw_panic_score"] >= 2) | (df["panic_cluster_count"] >= 2)) &
        (df["panic_severity"] == 0),
        "panic_severity"
    ] = 2
    
    df.loc[
        (df["return_panic"] & df["volatility_panic"] & df["volume_panic"]) &
        (df["panic_cluster_count"] >= 2),
        "panic_severity"
    ] = 3
    
    # LONG-TERM BOTTOM DETECTOR
    df["long_term_bottom"] = (
        df["return_panic"] &
        (df["volume_panic"] | df["volatility_panic"]) &
        (df["panic_cluster_count"] >= 2) &
        (df["panic_severity"] >= 2)
    )
    
    # Only show severity >= 2 (serious panics and capitulation)
    df["panic_signal"] = df["panic_severity"] >= 2
    
    return df


def identify_panic_clusters(df: pd.DataFrame) -> pd.DataFrame:
    """Group panic signals into clusters and identify cluster centers - same logic as Falling_Knife.py"""
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
    cluster_gap_days = 5
    
    for i, date in enumerate(panic_dates):
        if len(current_cluster) == 0:
            current_cluster.append(date)
        else:
            days_gap = (date - current_cluster[-1]).days
            if days_gap <= cluster_gap_days:
                current_cluster.append(date)
            else:
                if len(current_cluster) > 0:
                    clusters.append(current_cluster)
                current_cluster = [date]
    
    if len(current_cluster) > 0:
        clusters.append(current_cluster)
    
    # Process each cluster
    cluster_num = 1
    for cluster in clusters:
        if len(cluster) == 0:
            continue
        
        cluster_dates = pd.DatetimeIndex(cluster)
        cluster_df = df.loc[cluster_dates]
        
        # Find cluster center (date with lowest price)
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


def scan_ticker(ticker: str, lookback_days: int) -> Dict:
    """Scan a single ticker for signals in the last N days."""
    try:
        # Calculate start date (need extra days for rolling calculations)
        end_date = datetime.today()
        start_date = end_date - timedelta(days=lookback_days + 100)  # Extra buffer for calculations
        start_str = start_date.strftime("%Y-%m-%d")
        
        # Download data
        prices = download_price_data(ticker, start_str)
        price_series = select_price_series(prices, ticker)
        volume_series = select_volume_series(prices, ticker)
        
        df = pd.DataFrame({
            "Adj Close": price_series,
            "Volume": volume_series
        })
        
        # Calculate signals using Falling_Knife.py logic
        df = calculate_panic_signals(df)
        df = identify_panic_clusters(df)
        
        # Filter to lookback period
        cutoff_date = end_date - timedelta(days=lookback_days)
        df_recent = df[df.index >= cutoff_date]
        
        # Find recent cluster centers
        recent_clusters = df_recent[df_recent["is_cluster_center"]].copy()
        
        if len(recent_clusters) == 0:
            return {
                "ticker": ticker,
                "status": "OK",
                "signals_found": False,
                "signal_count": 0,
                "strong_bottoms": 0,
                "panic_signals": 0,
                "details": []
            }
        
        # Prepare details
        details = []
        for date in recent_clusters.sort_index(ascending=False).index:
            row = recent_clusters.loc[date]
            signal_type = "STRONG BOTTOM *" if row["long_term_bottom"] else f"Panic Signal (Severity {int(row['cluster_severity'])})"
            duration = "N/A"
            if pd.notna(row.get('cluster_start')) and pd.notna(row.get('cluster_end')):
                duration = f"{(row['cluster_end'] - row['cluster_start']).days + 1} days"
            
            details.append({
                "date": date.strftime("%Y-%m-%d"),
                "price": f"${row['Adj Close']:.2f}",
                "signal_type": signal_type,
                "severity": int(row['cluster_severity']),
                "duration": duration
            })
        
        strong_bottoms = recent_clusters['long_term_bottom'].sum()
        panic_signals = (recent_clusters['cluster_severity'] == 2).sum()
        
        return {
            "ticker": ticker,
            "status": "OK",
            "signals_found": True,
            "signal_count": len(recent_clusters),
            "strong_bottoms": int(strong_bottoms),
            "panic_signals": int(panic_signals),
            "details": details
        }
    
    except Exception as e:
        return {
            "ticker": ticker,
            "status": "ERROR",
            "error": str(e),
            "signals_found": False,
            "signal_count": 0,
            "strong_bottoms": 0,
            "panic_signals": 0,
            "details": []
        }


def main():
    print("=" * 70)
    print("FALLING KNIFE SIGNAL SCANNER")
    print("=" * 70)
    print()
    
    # Get tickers input
    print("Enter ticker symbols (separated by commas or spaces, or one per line).")
    print("Press Enter twice when done, or type 'file' to read from a file:")
    print()
    
    ticker_input = []
    print("Tickers (type 'file' for file input, or paste tickers):")
    
    first_line = input().strip()
    
    if first_line.lower() == 'file':
        filename = input("Enter filename: ").strip()
        try:
            with open(filename, 'r') as f:
                ticker_input = [line.strip().upper() for line in f if line.strip()]
        except FileNotFoundError:
            print(f"Error: File '{filename}' not found.")
            return
    else:
        # Try comma or space separated on first line
        if ',' in first_line:
            ticker_input = [t.strip().upper() for t in first_line.split(',')]
        elif ' ' in first_line:
            ticker_input = [t.strip().upper() for t in first_line.split()]
        else:
            ticker_input = [first_line.upper()]
        
        # Continue reading lines until empty
        while True:
            line = input().strip()
            if not line:
                break
            if ',' in line:
                ticker_input.extend([t.strip().upper() for t in line.split(',')])
            elif ' ' in line:
                ticker_input.extend([t.strip().upper() for t in line.split()])
            else:
                ticker_input.append(line.upper())
    
    # Remove duplicates and empty strings
    tickers = list(dict.fromkeys([t for t in ticker_input if t]))
    
    if not tickers:
        print("No tickers provided. Exiting.")
        return
    
    # Get lookback days
    print()
    days_input = input(f"Enter number of days to look back (default: 30): ").strip()
    if days_input:
        try:
            lookback_days = int(days_input)
        except ValueError:
            print(f"Invalid input, using default: 30")
            lookback_days = 30
    else:
        lookback_days = 30
    
    print()
    print("=" * 70)
    print(f"Scanning {len(tickers)} ticker(s) for signals in the last {lookback_days} days")
    print(f"Date range: {(datetime.today() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')} to {datetime.today().strftime('%Y-%m-%d')}")
    print("=" * 70)
    print()
    
    # Scan all tickers
    results = []
    for i, ticker in enumerate(tickers, 1):
        print(f"[{i}/{len(tickers)}] Scanning {ticker}...", end=" ", flush=True)
        result = scan_ticker(ticker, lookback_days)
        results.append(result)
        
        if result["status"] == "ERROR":
            print(f"ERROR: {result.get('error', 'Unknown error')}")
        elif result["signals_found"]:
            print(f"✓ Found {result['signal_count']} signal(s) - {result['strong_bottoms']} Strong Bottom(s), {result['panic_signals']} Panic Signal(s)")
        else:
            print("No signals")
    
    # Print summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    
    signals_found = [r for r in results if r["signals_found"]]
    errors = [r for r in results if r["status"] == "ERROR"]
    clean = [r for r in results if r["status"] == "OK" and not r["signals_found"]]
    
    if signals_found:
        print(f"🔍 SIGNALS FOUND ({len(signals_found)} ticker(s)):\n")
        for result in signals_found:
            print(f"  {result['ticker']}:")
            print(f"    Total Signals: {result['signal_count']}")
            print(f"    Strong Bottoms: {result['strong_bottoms']}")
            print(f"    Panic Signals: {result['panic_signals']}")
            print(f"    Details:")
            for detail in result['details']:
                print(f"      • {detail['date']} @ {detail['price']}: {detail['signal_type']} (Duration: {detail['duration']})")
            print()
    else:
        print("  No signals found in any tickers.\n")
    
    if errors:
        print(f"❌ ERRORS ({len(errors)} ticker(s)):\n")
        for result in errors:
            print(f"  {result['ticker']}: {result.get('error', 'Unknown error')}")
        print()
    
    if clean:
        print(f"✓ CLEAN ({len(clean)} ticker(s)): No signals found")
        if len(clean) <= 10:
            print(f"  {', '.join([r['ticker'] for r in clean])}\n")
        else:
            print(f"  {', '.join([r['ticker'] for r in clean[:10]])} ... and {len(clean) - 10} more\n")
    
    print("=" * 70)
    print(f"Scan complete: {len(signals_found)} with signals, {len(clean)} clean, {len(errors)} errors")
    print("=" * 70)
    print()
    
    # Save results to file
    save = input("Save results to file? (y/n): ").strip().lower()
    if save == 'y':
        filename = f"falling_knife_scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(filename, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("FALLING KNIFE SIGNAL SCAN RESULTS\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Scan Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Lookback Period: {lookback_days} days\n")
            f.write(f"Tickers Scanned: {len(tickers)}\n\n")
            
            if signals_found:
                f.write(f"SIGNALS FOUND ({len(signals_found)} ticker(s)):\n\n")
                for result in signals_found:
                    f.write(f"{result['ticker']}:\n")
                    f.write(f"  Total Signals: {result['signal_count']}\n")
                    f.write(f"  Strong Bottoms: {result['strong_bottoms']}\n")
                    f.write(f"  Panic Signals: {result['panic_signals']}\n")
                    f.write(f"  Details:\n")
                    for detail in result['details']:
                        f.write(f"    • {detail['date']} @ {detail['price']}: {detail['signal_type']} (Duration: {detail['duration']})\n")
                    f.write("\n")
            
            if errors:
                f.write(f"\nERRORS ({len(errors)} ticker(s)):\n\n")
                for result in errors:
                    f.write(f"  {result['ticker']}: {result.get('error', 'Unknown error')}\n")
        
        print(f"Results saved to {filename}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nScan interrupted by user.")
        sys.exit(0)
