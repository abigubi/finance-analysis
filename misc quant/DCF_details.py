import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt


# -----------------------------
# 1. Data download
# -----------------------------
def download_price_data(ticker="SPY", years=5):
    """
    Download daily adjusted close data for a ticker for the last `years`.
    """
    end = pd.Timestamp.now()
    start = end - pd.DateOffset(years=years)
    df = yf.download(ticker, start=start.strftime("%Y-%m-%d"), auto_adjust=True, progress=False)
    if df.empty:
        return df
    df = df[['Close']].copy()
    return df


# -----------------------------
# 2. Fair value helpers
# -----------------------------
def get_analyst_fair_value(ticker):
    """
    Try to get analyst mean target price from yfinance.
    Returns None if unavailable.
    """
    try:
        tk = yf.Ticker(ticker)
        info = tk.info  # can be slow but fine for interactive use
        fv = info.get("targetMeanPrice", None)
        if fv is None or fv == 0:
            return None
        return float(fv)
    except Exception:
        return None


def add_deviation_columns(df, fair_value,
                          deep_discount=-0.30,
                          discount=-0.15,
                          premium=0.15,
                          bubble=0.30):
    """
    Add deviation and valuation zone columns to df.
    """
    df = df.copy()
    df["FairValue"] = fair_value
    df["Deviation"] = (df["Close"] - fair_value) / fair_value  # e.g. -0.25 = 25% below FV

    def classify_zone(dev):
        # Handle Series edge cases
        if isinstance(dev, pd.Series):
            dev = dev.iloc[0] if len(dev) > 0 else np.nan
        
        try:
            if pd.isna(dev):
                return np.nan
            dev_val = float(dev)  # Ensure scalar
        except (ValueError, TypeError):
            return np.nan
        
        if dev_val <= deep_discount:
            return "Deep Discount"
        elif dev_val <= discount:
            return "Discount"
        elif dev_val < premium:
            return "Fair"
        elif dev_val < bubble:
            return "Premium"
        else:
            return "Bubble"

    df["Zone"] = df["Deviation"].apply(classify_zone)
    return df


# -----------------------------
# 3. Plotting
# -----------------------------
def plot_fair_value_deviation(df, ticker, fair_value,
                              deep_discount=-0.30,
                              discount=-0.15,
                              premium=0.15,
                              bubble=0.30):
    """
    Combined figure:
      - Top: price with fair value and bands
      - Bottom: deviation (%) with zone thresholds
    """
    df = df.copy()

    # Set up figure
    fig, (ax_price, ax_dev) = plt.subplots(
        2, 1, figsize=(14, 9), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]}
    )

    # --- Top: Price & fair value bands ---
    ax_price.plot(df.index, df["Close"], label=f"{ticker} Close", linewidth=1.4)

    # Fair value and bands
    ax_price.plot(df.index, df["FairValue"], linestyle="--", label="Fair Value")
    ax_price.plot(df.index, df["FairValue"] * (1 + deep_discount),
                  linestyle=":", linewidth=0.9, label="Deep Discount / Premium bands")
    ax_price.plot(df.index, df["FairValue"] * (1 + discount), linestyle=":", linewidth=0.9)
    ax_price.plot(df.index, df["FairValue"] * (1 + premium), linestyle=":", linewidth=0.9)
    ax_price.plot(df.index, df["FairValue"] * (1 + bubble), linestyle=":", linewidth=0.9)

    ax_price.set_title(f"{ticker} Price vs Fair Value (last ~5 years)")
    ax_price.set_ylabel("Price")
    ax_price.grid(True, alpha=0.3)
    ax_price.legend(loc="upper left")

    # --- Bottom: Deviation ---
    ax_dev.plot(df.index, df["Deviation"] * 100.0, label="Deviation from Fair Value (%)", linewidth=1.2)

    # Threshold lines
    ax_dev.axhline(deep_discount * 100, linestyle="--", linewidth=0.9, label="Deep Discount (-30%)")
    ax_dev.axhline(discount * 100, linestyle="--", linewidth=0.9, label="Discount (-15%)")
    ax_dev.axhline(0, linestyle=":", linewidth=0.8)
    ax_dev.axhline(premium * 100, linestyle="--", linewidth=0.9, label="Premium (+15%)")
    ax_dev.axhline(bubble * 100, linestyle="--", linewidth=0.9, label="Bubble (+30%)")

    ax_dev.set_title("Deviation from Fair Value")
    ax_dev.set_ylabel("Deviation (%)")
    ax_dev.grid(True, alpha=0.3)
    ax_dev.legend(loc="upper left")

    plt.tight_layout()
    plt.show()


# -----------------------------
# 4. Text summary
# -----------------------------
def summarize_deviation(df, ticker, fair_value):
    """
    Print useful stats about current and historical deviation.
    """
    last = df.iloc[-1]
    
    # Safely extract values
    try:
        current_price = float(last["Close"])
        current_dev = float(last["Deviation"]) if pd.notna(last["Deviation"]) else np.nan
        current_zone = last["Zone"] if pd.notna(last["Zone"]) else "N/A"
    except (ValueError, TypeError, KeyError):
        print("Error: Could not extract current values from data.")
        return

    print("\n--- Fair Value Deviation Summary ---")
    print(f"Ticker: {ticker}")
    print(f"Your Fair Value: {fair_value:.2f}")
    print(f"Current Price:  {current_price:.2f}")
    if pd.notna(current_dev):
        print(f"Current Deviation: {current_dev * 100:.1f}%")
    else:
        print(f"Current Deviation: N/A")
    print(f"Current Zone: {current_zone}")

    min_dev = df["Deviation"].min()
    max_dev = df["Deviation"].max()
    print(f"\nHistorical min deviation: {min_dev * 100:.1f}%")
    print(f"Historical max deviation: {max_dev * 100:.1f}%")

    # Zone distribution
    counts = df["Zone"].value_counts(dropna=True)
    total = len(df["Zone"].dropna())

    print("\nTime spent in each zone:")
    for zone in ["Deep Discount", "Discount", "Fair", "Premium", "Bubble"]:
        if zone in counts:
            pct = counts[zone] / total * 100
            print(f"  {zone:13s}: {pct:5.1f}% of days")
        else:
            print(f"  {zone:13s}:   0.0% of days")


# -----------------------------
# 5. Interactive usage
# -----------------------------
if __name__ == "__main__":
    print("=== Fair Value Deviation Visualizer ===\n")
    print("This tool compares price to your fair value estimate and")
    print("shows how cheap/expensive the stock is vs history.\n")

    while True:
        ticker_in = input("Enter ticker symbol (e.g., MU, ANF, GOOGL) or 'q' to quit: ").strip().upper()
        if ticker_in.lower() == "q":
            print("Goodbye!")
            break
        if not ticker_in:
            print("Please enter a valid ticker.\n")
            continue

        ticker = ticker_in

        # Get price history
        print(f"\n--- Downloading data for {ticker} (last ~5 years) ---")
        df_prices = download_price_data(ticker, years=5)
        if df_prices.empty:
            print(f"Error: no data found for {ticker}. Try a different symbol.\n")
            continue
        print(f"Downloaded {len(df_prices)} days of data.")

        # Ask user for fair value or use analyst target
        user_fv = input(
            "\nEnter YOUR fair value estimate (e.g., 120),\n"
            "or press Enter to try using analyst mean target from yfinance: "
        ).strip()

        fair_value = None
        if user_fv:
            try:
                fair_value = float(user_fv)
            except ValueError:
                print("Could not parse your input as a number. Trying analyst target instead...")
                fair_value = None

        if fair_value is None:
            fv_analyst = get_analyst_fair_value(ticker)
            if fv_analyst is not None:
                fair_value = fv_analyst
                print(f"Using analyst mean target as fair value: {fair_value:.2f}")
            else:
                # Fallback: last close
                fair_value = float(df_prices["Close"].iloc[-1])
                print(f"No analyst target found. Using last close as fair value: {fair_value:.2f}")

        # Compute deviations
        df_full = add_deviation_columns(df_prices, fair_value)

        # Summarize & plot
        summarize_deviation(df_full, ticker, fair_value)
        plot_fair_value_deviation(df_full, ticker, fair_value)

        print(f"\nAnalysis complete for {ticker}.\n")
