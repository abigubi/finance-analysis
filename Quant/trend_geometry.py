import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt


# -----------------------------
# 1. Data download
# -----------------------------
def download_price_data(ticker="SPY", years=5):
    """
    Download daily adjusted close for a ticker for the last `years`.
    """
    end = pd.Timestamp.now()
    start = end - pd.DateOffset(years=years)
    df = yf.download(ticker, start=start.strftime("%Y-%m-%d"), auto_adjust=True)
    if df.empty:
        return df
    df = df[["Close"]].copy()
    return df


# -----------------------------
# 2. Local quadratic fit helpers
# -----------------------------
def compute_local_trend_geometry(df, window=20):
    """
    For each day, fit a quadratic to the last `window` log prices:

        log P(t) ~ a + b*t + c*t^2

    and extract:
        velocity = b          (trend)
        curvature = 2*c       (acceleration)

    Also compute a local volatility and 'instantaneous Sharpe' = b / vol.
    """
    df = df.copy()
    log_price = np.log(df["Close"])

    n = len(df)
    vel = np.full(n, np.nan)
    curv = np.full(n, np.nan)
    inst_sharpe = np.full(n, np.nan)

    x = np.arange(window)

    for i in range(window - 1, n):
        y = log_price.iloc[i - window + 1: i + 1].values
        # Fit quadratic: y = a + b x + c x^2
        try:
            coeffs = np.polyfit(x, y, 2)
            a, b, c = coeffs[2], coeffs[1], coeffs[0]  # polyfit returns [c, b, a]
        except np.linalg.LinAlgError:
            continue

        vel[i] = b
        curv[i] = 2 * c  # second derivative of the quadratic

        # local vol of returns over same window
        local_returns = np.diff(y)
        local_vol = np.std(local_returns)
        if local_vol > 0:
            inst_sharpe[i] = b / local_vol
        else:
            inst_sharpe[i] = np.nan

    df["Velocity"] = vel
    df["Curvature"] = curv
    df["InstSharpe"] = inst_sharpe

    return df


# -----------------------------
# 3. Classify local geometry regime
# -----------------------------
def classify_geometry_regime(df,
                             vel_thresh=0.0,
                             curv_thresh=0.0):
    """
    Use signs of velocity & curvature to create local 'geometry regimes':
      - Strong Up, Accelerating
      - Up, Decelerating (topping)
      - Down, Accelerating
      - Down, Decelerating (potential bottom)
    """
    df = df.copy()

    def regime_row(row):
        v = row["Velocity"]
        c = row["Curvature"]
        if pd.isna(v) or pd.isna(c):
            return np.nan

        if v > vel_thresh and c > curv_thresh:
            return "Up, accelerating"
        elif v > vel_thresh and c <= curv_thresh:
            return "Up, decelerating"
        elif v <= vel_thresh and c < curv_thresh:
            return "Down, accelerating"
        else:  # v <= vel_thresh and c >= curv_thresh
            return "Down, decelerating"

    df["GeoRegime"] = df.apply(regime_row, axis=1)
    return df


# -----------------------------
# 4. Plotting
# -----------------------------
def _shade_geometry_on_axis(ax, df, regime_colors):
    """
    Shade background by geometry regime.
    """
    df = df.copy()
    df = df[df["GeoRegime"].notna()].copy()

    last_regime = None
    start_date = None
    prev_date = None

    for date, row in df.iterrows():
        regime = row["GeoRegime"]

        if regime != last_regime:
            if last_regime is not None and start_date is not None and prev_date is not None:
                ax.axvspan(start_date, prev_date,
                           color=regime_colors[last_regime],
                           alpha=0.10)
            start_date = date
            last_regime = regime
        prev_date = date

    if last_regime is not None and start_date is not None:
        ax.axvspan(start_date, df.index[-1],
                   color=regime_colors[last_regime],
                   alpha=0.10)


def plot_trend_geometry_dashboard(df, ticker, window=20):
    """
    3-panel dashboard:
      1) Price with geometry regimes
      2) Velocity & curvature
      3) Instantaneous Sharpe
    """
    df = df.copy()

    regime_colors = {
        "Up, accelerating": "green",
        "Up, decelerating": "lime",
        "Down, accelerating": "red",
        "Down, decelerating": "orange"
    }

    fig, (ax_price, ax_trend, ax_sharpe) = plt.subplots(
        3, 1,
        figsize=(14, 10),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 2, 2]}
    )

    # --- Panel 1: Price + regimes ---
    ax_price.plot(df.index, df["Close"], label=f"{ticker} Close", linewidth=1.4)
    _shade_geometry_on_axis(ax_price, df, regime_colors)
    ax_price.set_title(f"{ticker} Local Trend Geometry (window={window} days)")
    ax_price.set_ylabel("Price")
    ax_price.grid(True, alpha=0.3)
    ax_price.legend(loc="upper left")

    # --- Panel 2: Velocity & curvature ---
    ax_trend.plot(df.index, df["Velocity"], label="Velocity (b)", linewidth=1.2)
    ax_trend.plot(df.index, df["Curvature"], linestyle="--",
                  label="Curvature (2c)", linewidth=1.0)
    ax_trend.axhline(0, linestyle=":", linewidth=0.8)
    ax_trend.set_title("Local Velocity & Curvature of log-price")
    ax_trend.set_ylabel("Value")
    ax_trend.grid(True, alpha=0.3)
    ax_trend.legend(loc="upper left")

    # --- Panel 3: Instantaneous Sharpe ---
    ax_sharpe.plot(df.index, df["InstSharpe"], label="Instantaneous Sharpe", linewidth=1.2)
    ax_sharpe.axhline(0, linestyle=":", linewidth=0.8)
    ax_sharpe.set_title("Local Drift / Volatility (Inst. Sharpe)")
    ax_sharpe.set_ylabel("Sharpish")
    ax_sharpe.grid(True, alpha=0.3)
    ax_sharpe.legend(loc="upper left")

    plt.tight_layout()
    plt.show()


# -----------------------------
# 5. Summary text
# -----------------------------
def summarize_trend_geometry(df, ticker):
    """
    Print the latest velocity, curvature, Instant Sharpe, and regime.
    """
    last = df.dropna().iloc[-1]

    print("\n--- Local Trend Geometry Summary ---")
    print(f"Ticker: {ticker}")
    print(f"Last Close:     {last['Close']:.2f}")
    print(f"Velocity (b):   {last['Velocity']:.5f}")
    print(f"Curvature (2c): {last['Curvature']:.5f}")
    print(f"Inst. Sharpe:   {last['InstSharpe']:.2f}")
    print(f"Geo Regime:     {last['GeoRegime']}")


# -----------------------------
# 6. Interactive usage
# -----------------------------
if __name__ == "__main__":
    print("=== Local Trend Geometry (velocity + curvature) ===\n")

    while True:
        ticker_in = input("Enter ticker (e.g., MU, ANF, NVDA) or 'q' to quit: ").strip().upper()
        if ticker_in.lower() == "q":
            print("Goodbye!")
            break
        if not ticker_in:
            print("Please enter a valid ticker.\n")
            continue

        ticker = ticker_in

        # You can tweak years/window if you want
        years = 5
        window = 20

        print(f"\n--- Downloading data for {ticker} (last ~{years} years) ---")
        df = download_price_data(ticker, years=years)
        if df.empty:
            print(f"Error: no data for {ticker}. Try another symbol.\n")
            continue

        df = compute_local_trend_geometry(df, window=window)
        df = classify_geometry_regime(df)

        df_clean = df.dropna().copy()
        if df_clean.empty:
            print("Not enough data after rolling windows.\n")
            continue

        summarize_trend_geometry(df_clean, ticker)
        plot_trend_geometry_dashboard(df_clean, ticker, window=window)

        print(f"\nAnalysis complete for {ticker}.\n")
