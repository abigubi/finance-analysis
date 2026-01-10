import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt


# -----------------------------
# 1. Data download
# -----------------------------
def download_price_data(ticker="SPY", start="2010-01-01", end=None):
    """
    Download daily OHLCV data for a given ticker using yfinance.
    """
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if df.empty:
        return df
    df = df[['Close', 'Volume']].copy()
    df['Return'] = df['Close'].pct_change()
    return df


# -----------------------------
# 2. Feature engineering
# -----------------------------
def add_regime_features(df,
                        vol_lookback=21,
                        vol_z_lookback=252,
                        dd_lookback=252):
    """
    Add realized volatility, volatility Z-score, moving averages,
    and drawdown features used for regime classification.
    """
    df = df.copy()

    # Realized volatility (annualized) on 21-day returns
    df['RV_21'] = df['Return'].rolling(vol_lookback).std() * np.sqrt(252)

    # Volatility Z-score vs trailing history
    vol_mean = df['RV_21'].rolling(vol_z_lookback).mean()
    vol_std = df['RV_21'].rolling(vol_z_lookback).std()
    df['Vol_Z'] = (df['RV_21'] - vol_mean) / vol_std

    # Trend structure
    df['MA50'] = df['Close'].rolling(50).mean()
    df['MA200'] = df['Close'].rolling(200).mean()
    df['MA50_slope'] = df['MA50'].diff(5)  # 5-day slope

    # Drawdown vs 1-year high
    rolling_max = df['Close'].rolling(dd_lookback).max()
    df['Drawdown'] = df['Close'] / rolling_max - 1

    return df


# -----------------------------
# 3. Regime classification rules
# -----------------------------
def classify_regime_row(row,
                        vol_z_panic=1.0,
                        drawdown_panic=-0.15):
    """
    Classify a single row into a regime based on rules.
    """

    # --- helpers (same as before) ---
    def is_nan_value(val):
        if isinstance(val, pd.Series):
            val = val.iloc[0] if len(val) > 0 else np.nan
        try:
            return pd.isna(val)
        except (ValueError, TypeError):
            return isinstance(val, float) and np.isnan(val)

    def to_scalar(val):
        if isinstance(val, pd.Series):
            return val.iloc[0] if len(val) > 0 else np.nan
        return val

    try:
        ma50 = to_scalar(row['MA50'])
        ma200 = to_scalar(row['MA200'])
        vol_z = to_scalar(row['Vol_Z'])
        drawdown = to_scalar(row.get('Drawdown', np.nan))
        close = to_scalar(row['Close'])
        ma50_slope = to_scalar(row.get('MA50_slope', np.nan))

        if is_nan_value(ma50) or is_nan_value(ma200) or is_nan_value(vol_z):
            return np.nan
    except (KeyError, IndexError, AttributeError):
        return np.nan

    # Panic / Bear: volatility spike or deep drawdown
    if (vol_z > vol_z_panic) or (drawdown < drawdown_panic):
        return 'Panic / Bear'

    # Bull / Trending Up: strong trend structure
    if (close > ma50 > ma200) and (ma50_slope > 0):
        return 'Bull / Trending Up'

    # Otherwise: Sideways / Neutral
    return 'Sideways / Neutral'


def add_regimes(df,
                vol_z_panic=1.0,
                drawdown_panic=-0.15):
    """
    Apply classification across all rows and return DataFrame with a 'Regime' column.
    """
    df = add_regime_features(df)
    df['Regime'] = df.apply(
        lambda row: classify_regime_row(
            row,
            vol_z_panic=vol_z_panic,
            drawdown_panic=drawdown_panic
        ),
        axis=1
    )
    return df


# -----------------------------
# 4. Plotting helpers
# -----------------------------

def _shade_regimes_on_axis(ax, df, regime_colors):  # NEW helper
    """
    Internal helper: shade background according to regime on an axis.
    """
    df = df.copy()
    df = df[df['Regime'].notna()].copy()

    last_regime = None
    start_date = None
    prev_date = None

    for date, row in df.iterrows():
        regime_val = row['Regime']

        if isinstance(regime_val, pd.Series):
            regime_val = regime_val.iloc[0] if len(regime_val) > 0 else None

        try:
            if pd.isna(regime_val):
                prev_date = date
                continue
        except (ValueError, TypeError):
            if regime_val is None:
                prev_date = date
                continue

        regime = str(regime_val)

        if regime != last_regime:
            if last_regime is not None and start_date is not None and prev_date is not None:
                ax.axvspan(start_date, prev_date,
                           color=regime_colors[last_regime],
                           alpha=0.12)
            start_date = date
            last_regime = regime

        prev_date = date

    if last_regime is not None and start_date is not None:
        ax.axvspan(start_date, df.index[-1],
                   color=regime_colors[last_regime],
                   alpha=0.12)


def plot_price_and_vol_with_regimes(df, ticker="SPY", panic_threshold=1.0):  # NEW main plot
    """
    Combined figure:
      - Top: price with regime shading
      - Bottom: volatility Z-score with panic threshold
    """
    regime_colors = {
        'Bull / Trending Up': 'green',
        'Sideways / Neutral': 'orange',
        'Panic / Bear': 'red'
    }

    df = df.copy()

    fig, (ax_price, ax_vol) = plt.subplots(
        2, 1, figsize=(14, 9), sharex=True,
        gridspec_kw={'height_ratios': [3, 1]}
    )

    # ---- Top: Price + regimes ----
    ax_price.plot(df.index, df['Close'], label=f'{ticker} Close', linewidth=1.4)
    _shade_regimes_on_axis(ax_price, df, regime_colors)
    ax_price.set_title(f'{ticker} Price & Volatility Regime (Last ~3 Years)')
    ax_price.set_ylabel('Price')
    ax_price.grid(True, alpha=0.3)
    ax_price.legend(loc='upper left')

    # ---- Bottom: Vol Z-score ----
    ax_vol.plot(df.index, df['Vol_Z'], label='Volatility Z-Score', linewidth=1.2)
    ax_vol.axhline(panic_threshold, linestyle='--', label='Panic threshold')
    ax_vol.axhline(0, linestyle=':', linewidth=0.8)
    ax_vol.set_title('21-day Realized Volatility Z-Score')
    ax_vol.set_ylabel('Z-Score')
    ax_vol.grid(True, alpha=0.3)
    ax_vol.legend(loc='upper left')

    plt.tight_layout()
    plt.show()


# (You can keep the old standalone plot_vol_z if you still want it)
def plot_vol_z(df, ticker="SPY", panic_threshold=1.0):
    """
    Plot volatility Z-score with a panic threshold line.
    """
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(df.index, df['Vol_Z'], label='Volatility Z-Score', linewidth=1)
    ax.axhline(panic_threshold, linestyle='--', label='Panic threshold')
    ax.axhline(0, linestyle=':', linewidth=0.8)
    ax.set_title(f'{ticker} 21-day Volatility Z-Score')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# -----------------------------
# 5. Interactive usage
# -----------------------------
if __name__ == "__main__":
    print("=== Market Regime Detector ===\n")
    print("This tool analyzes price & volatility to detect regimes:")
    print("  - Bull / Trending Up")
    print("  - Sideways / Neutral")
    print("  - Panic / Bear\n")

    while True:
        user_input = input("Enter ticker symbol (e.g., SPY, AAPL, NVDA) or 'q' to quit: ").strip().upper()

        if user_input.lower() == 'q':
            print("Goodbye!")
            break

        if not user_input:
            print("Please enter a valid ticker symbol.")
            continue

        ticker = user_input

        # Ask user for number of years to look back
        print("\nHow many years back would you like to analyze?")
        print("  - Press ENTER for default (2.5 years)")
        print("  - Enter a number (e.g., 3, 3.5, 5)")
        
        years_input = input("Years back: ").strip()
        
        if not years_input:
            # Default: 2.5 years
            years_back = 2.5
        else:
            try:
                years_back = float(years_input)
                if years_back <= 0:
                    print("Years must be positive. Using default (2.5 years).\n")
                    years_back = 2.5
                elif years_back > 20:
                    print("That's a very long period. Using 20 years maximum.\n")
                    years_back = 20.0
            except ValueError:
                print("Invalid input. Using default (2.5 years).\n")
                years_back = 2.5
        
        # Calculate start date based on years back from today
        start_date = (pd.Timestamp.now() - pd.DateOffset(days=int(years_back * 365.25))).strftime('%Y-%m-%d')
        end_date = pd.Timestamp.now().strftime('%Y-%m-%d')

        print(f"\n--- Analyzing {ticker} (Last {years_back} years: {start_date} to {end_date}) ---")

        try:
            df = download_price_data(ticker=ticker, start=start_date, end=end_date)

            if df.empty:
                print(f"Error: No data found for {ticker}. Please check the ticker symbol.")
                continue

            print(f"Downloaded {len(df)} days of data.")
            
            # Check if we have enough data for meaningful analysis
            if len(df) < 300:
                print(f"Warning: Only {len(df)} days of data. Results may be less reliable.")
                print("Try a ticker with more history.\n")
                continue

            # Adjust lookback periods if data is limited (but 2.5 years should be fine)
            df = add_regimes(
                df,
                vol_z_panic=1.0,      # higher = less sensitive to panic
                drawdown_panic=-0.15  # -0.20 for only big bear markets
            )
            
            # Filter to show only data where we have complete regime info
            df_valid = df[df['Regime'].notna()].copy()
            if len(df_valid) == 0:
                print(f"Error: No regime data calculated for {ticker}.")
                print("This may be due to insufficient data.\n")
                continue
            
            print(f"Regime data available for {len(df_valid)} days (initial {len(df) - len(df_valid)} days excluded for indicator calculations).")

            print("\n--- Recent Regime Analysis ---")
            display_df = df_valid[['Close', 'Vol_Z', 'Drawdown', 'Regime']].tail(10)
            print(display_df)

            print(f"\nCurrent Regime: {df_valid['Regime'].iloc[-1]}")
            if pd.notna(df_valid['Vol_Z'].iloc[-1]):
                print(f"Current Volatility Z-Score: {df_valid['Vol_Z'].iloc[-1]:.2f}")
            if pd.notna(df_valid['Drawdown'].iloc[-1]):
                print(f"Current Drawdown: {df_valid['Drawdown'].iloc[-1]:.2%}")
            print()

            # NEW: combined chart instead of 2 separate figures
            plot_price_and_vol_with_regimes(df_valid, ticker=ticker, panic_threshold=1.0)

            print(f"\nAnalysis complete for {ticker}.\n")

        except Exception as e:
            print(f"Error processing {ticker}: {e}")
            print("Please try again with a different ticker.\n")
