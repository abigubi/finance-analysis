import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt


# -----------------------------
# 1. Data download
# -----------------------------
def download_price_data(ticker="SPY", years=5):
    """
    Download daily adjusted close & volume for a ticker for the last `years`.
    """
    end = pd.Timestamp.now()
    start = end - pd.DateOffset(years=years)
    df = yf.download(ticker, start=start.strftime("%Y-%m-%d"), auto_adjust=True, progress=False)
    if df.empty:
        return df
    
    # Handle case where download returns Series or DataFrame
    if isinstance(df, pd.Series):
        df = pd.DataFrame({'Close': df, 'Volume': np.nan})
    elif isinstance(df, pd.DataFrame):
        required_cols = ["Close"]
        available_cols = [col for col in required_cols if col in df.columns]
        if "Volume" in df.columns:
            available_cols.append("Volume")
        else:
            df["Volume"] = np.nan
            available_cols.append("Volume")
        df = df[available_cols].copy()
    
    return df


# -----------------------------
# 2. Feature engineering
# -----------------------------
def add_quant_features(df,
                       sharpe_window_short=63,   # ~3 months
                       sharpe_window_long=252,   # ~1 year
                       z_window=63):
    """
    Add daily returns, rolling Sharpe ratios, drawdown, and
    Z-score-based bottom detector features.
    """
    df = df.copy()

    # Daily returns
    df["Return"] = df["Close"].pct_change()

    # Rolling Sharpe ratios (annualized)
    def rolling_sharpe(returns, window):
        roll_mean = returns.rolling(window).mean()
        roll_std = returns.rolling(window).std()
        sharpe = roll_mean / roll_std * np.sqrt(252)
        return sharpe

    df[f"Sharpe_{sharpe_window_short}"] = rolling_sharpe(df["Return"], sharpe_window_short)
    df[f"Sharpe_{sharpe_window_long}"] = rolling_sharpe(df["Return"], sharpe_window_long)

    # Drawdown
    running_max = df["Close"].cummax()
    df["Drawdown"] = df["Close"] / running_max - 1.0

    # Z-scores on returns and volume
    ret_mean = df["Return"].rolling(z_window).mean()
    ret_std = df["Return"].rolling(z_window).std()
    df["Ret_Z"] = (df["Return"] - ret_mean) / ret_std

    vol_mean = df["Volume"].rolling(z_window).mean()
    vol_std = df["Volume"].rolling(z_window).std()
    df["Vol_Z"] = (df["Volume"] - vol_mean) / vol_std

    # Bottom detector: big negative return + high volume
    bottom_condition = (df["Ret_Z"] < -2.0) & (df["Vol_Z"] > 1.0)
    df["BottomSignal"] = bottom_condition

    return df


# -----------------------------
# 3. Plotting
# -----------------------------
def plot_quant_dashboard(df, ticker,
                         sharpe_window_short=63,
                         sharpe_window_long=252):
    """
    3-panel dashboard:
      1) Price with bottom signals
      2) Rolling Sharpe ratios
      3) Drawdown
    """
    df = df.copy()

    fig, (ax_price, ax_sharpe, ax_dd) = plt.subplots(
        3, 1,
        figsize=(14, 10),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 2, 2]}
    )

    # --- Panel 1: Price + bottom signals ---
    ax_price.plot(df.index, df["Close"], label=f"{ticker} Close", linewidth=1.4)

    # Mark bottom signals
    bottoms = df[df["BottomSignal"]]
    ax_price.scatter(bottoms.index, bottoms["Close"], marker="v",
                     label="Bottom signal", zorder=5)

    ax_price.set_title(f"{ticker} Quant Dashboard (Price & Bottom Signals)")
    ax_price.set_ylabel("Price")
    ax_price.grid(True, alpha=0.3)
    ax_price.legend(loc="upper left")

    # --- Panel 2: Rolling Sharpe ratios ---
    ax_sharpe.plot(df.index, df[f"Sharpe_{sharpe_window_short}"],
                   label=f"Sharpe {sharpe_window_short}d", linewidth=1.2)
    ax_sharpe.plot(df.index, df[f"Sharpe_{sharpe_window_long}"],
                   linestyle="--", label=f"Sharpe {sharpe_window_long}d", linewidth=1.0)

    ax_sharpe.axhline(0, linestyle=":", linewidth=0.8)
    ax_sharpe.set_title("Rolling Sharpe Ratios (annualized)")
    ax_sharpe.set_ylabel("Sharpe")
    ax_sharpe.grid(True, alpha=0.3)
    ax_sharpe.legend(loc="upper left")

    # --- Panel 3: Drawdown ---
    ax_dd.plot(df.index, df["Drawdown"] * 100.0, label="Drawdown", linewidth=1.2)
    ax_dd.set_title("Drawdown from Peak")
    ax_dd.set_ylabel("Drawdown (%)")
    ax_dd.grid(True, alpha=0.3)
    ax_dd.legend(loc="upper left")

    plt.tight_layout()
    plt.show()


# -----------------------------
# 4. Summary text
# -----------------------------
def summarize_quant(df, ticker,
                    sharpe_window_short=63,
                    sharpe_window_long=252):
    """
    Print latest Sharpe ratios and recent bottom signals.
    """
    df_clean = df.dropna()
    if df_clean.empty:
        print("\n--- Quant Summary ---")
        print(f"Ticker: {ticker}")
        print("Error: No valid data after processing.")
        return
    
    last = df_clean.iloc[-1]

    try:
        s_short = float(last[f"Sharpe_{sharpe_window_short}"]) if pd.notna(last.get(f"Sharpe_{sharpe_window_short}")) else np.nan
        s_long = float(last[f"Sharpe_{sharpe_window_long}"]) if pd.notna(last.get(f"Sharpe_{sharpe_window_long}")) else np.nan
        price = float(last["Close"])
    except (KeyError, ValueError, TypeError) as e:
        print(f"\nError extracting summary values: {e}")
        return

    print("\n--- Quant Summary ---")
    print(f"Ticker: {ticker}")
    print(f"Latest Close: {price:.2f}")
    if pd.notna(s_short):
        print(f"Sharpe({sharpe_window_short}d): {s_short:.2f}")
    else:
        print(f"Sharpe({sharpe_window_short}d): N/A")
    if pd.notna(s_long):
        print(f"Sharpe({sharpe_window_long}d): {s_long:.2f}")
    else:
        print(f"Sharpe({sharpe_window_long}d): N/A")

    # Last few bottom signals
    try:
        bottoms = df[df["BottomSignal"]].tail(5)
        if not bottoms.empty:
            print("\nRecent bottom signals (date, price, Ret_Z, Vol_Z):")
            for idx, row in bottoms.iterrows():
                ret_z = row.get('Ret_Z', np.nan)
                vol_z = row.get('Vol_Z', np.nan)
                ret_z_str = f"{ret_z:.2f}" if pd.notna(ret_z) else "N/A"
                vol_z_str = f"{vol_z:.2f}" if pd.notna(vol_z) else "N/A"
                print(f"  {idx.date()}  |  {row['Close']:.2f}  |  Ret_Z={ret_z_str}  Vol_Z={vol_z_str}")
        else:
            print("\nNo bottom signals in the lookback period.")
    except Exception as e:
        print(f"\nError processing bottom signals: {e}")


# -----------------------------
# 5. Interactive usage
# -----------------------------
if __name__ == "__main__":
    print("=== Quant Dashboard (Sharpe + Bottom Detector) ===\n")

    while True:
        ticker_in = input("Enter ticker (e.g., MU, ANF, GOOGL) or 'q' to quit: ").strip().upper()
        if ticker_in.lower() == "q":
            print("Goodbye!")
            break
        if not ticker_in:
            print("Please enter a valid ticker.\n")
            continue

        ticker = ticker_in

        print(f"\n--- Downloading data for {ticker} (last ~5 years) ---")
        df = download_price_data(ticker, years=5)
        if df.empty:
            print(f"Error: no data for {ticker}. Try another symbol.\n")
            continue

        df = add_quant_features(df)

        # Some rows at the start will be NaN due to rolling windows
        df_clean = df.dropna().copy()
        if df_clean.empty:
            print("Not enough data after rolling windows. Try a more liquid / older ticker.\n")
            continue

        summarize_quant(df_clean, ticker)
        plot_quant_dashboard(df_clean, ticker)

        print(f"\nAnalysis complete for {ticker}.\n")
