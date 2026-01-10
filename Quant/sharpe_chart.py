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
            else:
                return series
    else:
        for candidate in ("Adj Close", "Close"):
            if candidate in prices.columns:
                return prices[candidate]
    raise KeyError(
        "Could not find an adjusted or close price column in the downloaded data."
    )


def calculate_rolling_sharpe(
    prices: pd.DataFrame, window: int
) -> pd.DataFrame:
    price_series = select_price_series(prices, prices.attrs.get("ticker", ""))
    returns = price_series.pct_change()
    rolling_mean = returns.rolling(window).mean()
    rolling_std = returns.rolling(window).std()

    sharpe = (rolling_mean / rolling_std) * np.sqrt(252)
    sharpe_mean = sharpe.rolling(window).mean()
    sharpe_std = sharpe.rolling(window).std()

    result = pd.DataFrame(index=prices.index)
    result["Sharpe"] = sharpe
    result["Mean"] = sharpe_mean
    result["+2σ"] = sharpe_mean + 2 * sharpe_std
    result["-2σ"] = sharpe_mean - 2 * sharpe_std
    result["Price"] = price_series
    return result.dropna()


def plot_sharpe_with_price(ticker: str, data: pd.DataFrame, window: int) -> None:
    plt.style.use("default")
    fig, ax1 = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor('white')
    ax1.set_facecolor('white')

    # Plot Sharpe ratio with a nice blue color
    ax1.plot(data.index, data["Sharpe"], color="#1f77b4", linewidth=2, label="Sharpe", alpha=0.9)
    ax1.plot(data.index, data["Mean"], color="#666666", linestyle="--", linewidth=1.5, label="Mean", alpha=0.7)
    ax1.plot(data.index, data["+2σ"], color="#d62728", linestyle="--", linewidth=1.5, label="+2σ", alpha=0.7)
    ax1.plot(data.index, data["-2σ"], color="#d62728", linestyle="--", linewidth=1.5, label="-2σ", alpha=0.7)
    
    # Fill between the bands for better visualization
    ax1.fill_between(data.index, data["+2σ"], data["-2σ"], color="#d62728", alpha=0.1, label="±2σ Band")
    
    ax1.set_ylabel("Sharpe Ratio", color="#1f77b4", fontsize=11, fontweight='bold')
    ax1.tick_params(axis="y", colors="#1f77b4")
    ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_color('#1f77b4')
    ax1.spines['bottom'].set_color('#333333')

    ax2 = ax1.twinx()
    ax2.plot(data.index, data["Price"], color="#ff7f0e", linewidth=2, label="Price", alpha=0.9)
    ax2.set_ylabel("Price", color="#ff7f0e", fontsize=11, fontweight='bold')
    ax2.tick_params(axis="y", colors="#ff7f0e")
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_color('#ff7f0e')
    ax2.spines['left'].set_visible(False)
    ax2.spines['bottom'].set_color('#333333')

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper left", framealpha=0.95, fancybox=True, shadow=True)

    plt.title(f"{ticker.upper()} - Sharpe Ratio ±2σ Bands with Price (Rolling {window})", 
              fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.show()


def parse_args() -> Tuple[Optional[str], str, int]:
    parser = argparse.ArgumentParser(
        description="Plot rolling Sharpe ratio bands alongside price."
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
        default="2023-01-01",
        help="Start date for historical data (YYYY-MM-DD, default: 2023-01-01)",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=30,
        help="Rolling window size in days for Sharpe calculation (default: 30)",
    )
    args = parser.parse_args()
    return args.ticker, args.start, args.window


def main() -> None:
    ticker, start, window = parse_args()
    if not ticker:
        ticker = input("Enter ticker symbol: ").strip().upper()
        if not ticker:
            raise ValueError("Ticker symbol is required.")
    else:
        ticker = ticker.upper()

    prices = download_price_data(ticker, start)
    prices.attrs["ticker"] = ticker
    sharpe_data = calculate_rolling_sharpe(prices, window)
    if sharpe_data.empty:
        raise ValueError(
            "Not enough data to compute rolling Sharpe ratios. "
            f"Try a smaller window or an earlier start date."
        )
    plot_sharpe_with_price(ticker, sharpe_data, window)


if __name__ == "__main__":
    main()
