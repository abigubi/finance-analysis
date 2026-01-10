import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt


# -----------------------------
# 1. Data download
# -----------------------------
def download_universe_prices(tickers, years=3):
    """
    Download daily adjusted close prices for a list of tickers
    for the last `years`.
    Returns a DataFrame with columns = tickers.
    """
    end = pd.Timestamp.now()
    start = end - pd.DateOffset(years=years)

    data = yf.download(
        tickers,
        start=start.strftime("%Y-%m-%d"),
        auto_adjust=True
    )

    # yfinance gives multi-index columns when multiple tickers
    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"].copy()
    else:
        close = data[["Close"]].copy()
        close.columns = [tickers]  # single ticker case

    return close


# -----------------------------
# 2. Advance–Decline & volatility
# -----------------------------
def compute_advance_decline_metrics(price_df,
                                    ad_vol_window=21):
    """
    Given a price dataframe (cols=tickers), compute:
      - daily returns
      - advance ratio, decline ratio
      - AD_diff = (adv - dec) / n
      - AD_vol = rolling std of AD_diff
    """
    prices = price_df.copy()
    rets = prices.pct_change()

    # For each day, count adv/dec among stocks with non-NaN returns
    adv_count = (rets > 0).sum(axis=1)
    dec_count = (rets < 0).sum(axis=1)
    n_valid = rets.notna().sum(axis=1)

    adv_ratio = adv_count / n_valid
    dec_ratio = dec_count / n_valid
    ad_diff = (adv_count - dec_count) / n_valid  # in [-1, 1]

    ad_vol = ad_diff.rolling(ad_vol_window).std()

    ad_df = pd.DataFrame(
        {
            "AdvCount": adv_count,
            "DecCount": dec_count,
            "N": n_valid,
            "AdvRatio": adv_ratio,
            "DecRatio": dec_ratio,
            "AD_Diff": ad_diff,
            "AD_Vol": ad_vol,
        },
        index=prices.index,
    )

    return rets, ad_df


# -----------------------------
# 3. Mark extreme AD volatility spikes
# -----------------------------
def find_ad_vol_spikes(ad_df, vol_col="AD_Vol",
                       high_quantile=0.9):
    """
    Return a DataFrame with rows where AD_Vol is in the top
    `high_quantile` fraction of its history.
    """
    series = ad_df[vol_col].dropna()
    if series.empty:
        return ad_df.iloc[0:0]  # empty

    threshold = series.quantile(high_quantile)
    spikes = ad_df[ad_df[vol_col] >= threshold]
    return spikes, threshold


# -----------------------------
# 4. Plotting: price + AD breadth + AD volatility
# -----------------------------
def plot_ad_vol_dashboard(index_price,
                          ad_df,
                          index_ticker="SPY",
                          universe_name="Universe",
                          spikes=None,
                          spike_threshold=None):
    """
    3-panel plot:
      1) Index price (line)
      2) AD_Diff (breadth)
      3) AD_Vol with markers at extreme spikes
    """
    fig, (ax_price, ax_ad, ax_vol) = plt.subplots(
        3, 1,
        figsize=(14, 10),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 2, 2]},
    )

    # --- Panel 1: Index price ---
    ax_price.plot(index_price.index, index_price.values,
                  label=f"{index_ticker} Price", linewidth=1.4)
    ax_price.set_title(f"{index_ticker} & {universe_name} Advance–Decline Volatility")
    ax_price.set_ylabel("Price")
    ax_price.grid(True, alpha=0.3)
    ax_price.legend(loc="upper left")

    # --- Panel 2: AD_Diff (breadth) ---
    ax_ad.plot(ad_df.index, ad_df["AD_Diff"],
               label="AD_Diff (adv - dec, normalized)",
               linewidth=1.2)
    ax_ad.axhline(0, linestyle=":", linewidth=0.8)
    ax_ad.set_title("Advance–Decline Difference (cross-section breadth)")
    ax_ad.set_ylabel("AD_Diff")
    ax_ad.grid(True, alpha=0.3)
    ax_ad.legend(loc="upper left")

    # --- Panel 3: AD_Vol (breadth volatility) ---
    ax_vol.plot(ad_df.index, ad_df["AD_Vol"],
                label="AD_Vol (rolling std of AD_Diff)",
                linewidth=1.2)
    if spike_threshold is not None:
        ax_vol.axhline(spike_threshold, linestyle="--",
                       linewidth=0.9,
                       label=f"{int(100*0.9)}th percentile")

    if spikes is not None and not spikes.empty:
        ax_vol.scatter(spikes.index, spikes["AD_Vol"],
                       marker="v", s=40,
                       label="AD_Vol spike", zorder=5)

    ax_vol.set_title("Advance–Decline Volatility")
    ax_vol.set_ylabel("AD_Vol")
    ax_vol.grid(True, alpha=0.3)
    ax_vol.legend(loc="upper left")

    plt.tight_layout()
    plt.show()


# -----------------------------
# 5. Interactive usage
# -----------------------------
if __name__ == "__main__":
    print("=== Advance–Decline Volatility Dashboard ===\n")

    while True:
        idx_in = input("Index ticker for price (e.g., SPY, QQQ) or 'q' to quit: ").strip().upper()
        if idx_in.lower() == "q":
            print("Goodbye!")
            break
        if not idx_in:
            print("Please enter a valid index ticker.\n")
            continue

        universe_in = input(
            "Universe tickers for breadth (comma-separated, e.g., AAPL,MSFT,NVDA,GOOGL),\n"
            "or press Enter to just use the index itself: "
        ).strip().upper()

        if universe_in:
            universe_tickers = [t.strip() for t in universe_in.split(",") if t.strip()]
        else:
            universe_tickers = [idx_in]

        years = 3
        print(f"\n--- Downloading universe data ({years} years) ---")
        prices_universe = download_universe_prices(universe_tickers, years=years)
        if prices_universe.empty:
            print("No data for universe tickers. Try again.\n")
            continue

        print(f"Got {len(prices_universe)} days of universe prices for {len(universe_tickers)} tickers.")

        # Index price: if index ticker is part of universe, use that column; otherwise download separately
        if idx_in in prices_universe.columns:
            index_price = prices_universe[idx_in]
        else:
            print(f"Downloading index price separately for {idx_in}...")
            idx_df = download_universe_prices([idx_in], years=years)
            if idx_df.empty:
                print(f"No data for index ticker {idx_in}. Try again.\n")
                continue
            index_price = idx_df[idx_in]

        # Align dates
        common_idx = prices_universe.index.intersection(index_price.index)
        prices_universe = prices_universe.loc[common_idx]
        index_price = index_price.loc[common_idx]

        # Compute AD & volatility
        rets, ad_df = compute_advance_decline_metrics(prices_universe)

        # Find spikes in AD volatility
        spikes, thresh = find_ad_vol_spikes(ad_df, high_quantile=0.9)

        # Quick text summary
        print("\n--- AD Volatility Summary ---")
        last = ad_df.dropna().iloc[-1]
        print(f"Latest AD_Diff: {last['AD_Diff']:.3f}")
        print(f"Latest AD_Vol:  {last['AD_Vol']:.3f}")
        print(f"90th percentile AD_Vol threshold: {thresh:.3f}")
        if not spikes.empty:
            print("\nLast few AD_Vol spikes:")
            for idx, row in spikes.tail(5).iterrows():
                print(f"  {idx.date()}  |  AD_Vol={row['AD_Vol']:.3f}")
        else:
            print("\nNo spikes above the 90th percentile yet.")

        # Plot
        plot_ad_vol_dashboard(
            index_price=index_price,
            ad_df=ad_df,
            index_ticker=idx_in,
            universe_name=f"{len(universe_tickers)}-stock universe",
            spikes=spikes,
            spike_threshold=thresh,
        )

        print("\nAnalysis complete.\n")
