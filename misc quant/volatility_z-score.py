import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def run_volatility_z_score_analysis(ticker, start_date='2020-01-01', vol_window=30, z_score_window=252):
    
    ticker = ticker.upper().strip()
    print(f"--- VOLATILITY Z-SCORE ANALYSIS FOR {ticker} ---")

    # 1. DOWNLOAD DATA
    try:
        data = yf.download(ticker, start=start_date, progress=False, auto_adjust=False)
        
        # FIX: Handle yfinance MultiIndex
        if isinstance(data.columns, pd.MultiIndex):
            try:
                data = data.xs(ticker, level=1, axis=1)
            except KeyError:
                data.columns = data.columns.get_level_values(0)

        if data.empty:
            print(f"Error: No data found for {ticker}.")
            return

        # Ensure we have a Series
        if 'Close' in data.columns:
            prices = data['Close']
        elif 'Adj Close' in data.columns:
            prices = data['Adj Close']
        else:
            prices = data.iloc[:, 0]

        # Squeeze to ensure it's a Series (1D) and clean it
        prices = prices.squeeze().dropna()
        
        if len(prices) < z_score_window + vol_window:
             print("Error: Not enough data for a meaningful Z-Score calculation.")
             return

    except Exception as e:
        print(f"Error downloading/processing data for {ticker}: {e}")
        return

    # 2. CALCULATE VOLATILITY
    returns = np.log(prices / prices.shift(1))
    
    # Rolling Volatility (Annualized)
    daily_sigma = returns.rolling(window=vol_window).std()
    annualized_volatility = daily_sigma * np.sqrt(252)
    
    # 3. CALCULATE ROLLING Z-SCORE
    vol_series = annualized_volatility.dropna()
    
    vol_mean = vol_series.rolling(window=z_score_window).mean()
    vol_std = vol_series.rolling(window=z_score_window).std()

    # Z = (Current Vol - Mean Vol) / Std Dev
    z_score = (vol_series - vol_mean) / vol_std
    z_score = z_score.dropna()
    
    # Re-align everything to the z_score index
    common_index = z_score.index
    vol_series = vol_series.reindex(common_index)
    vol_mean = vol_mean.reindex(common_index)
    vol_std = vol_std.reindex(common_index)
    prices_aligned = prices.reindex(common_index)

    # Calculate Bands
    vol_2_up = vol_mean + 2 * vol_std
    vol_2_down = vol_mean - 2 * vol_std

    # --- PREPARE DATA FOR PLOTTING (FLATTEN TO 1D) ---
    x_dates = common_index
    y_vol = vol_series.values.flatten() * 100
    y_mean = vol_mean.values.flatten() * 100
    y_up = vol_2_up.values.flatten() * 100
    y_down = vol_2_down.values.flatten() * 100
    y_zscore = z_score.values.flatten()
    y_price = prices_aligned.values.flatten()

    # --- PLOTTING ---
    # PROFESSIONAL LIGHT THEME
    plt.style.use('default') # Reset to standard white
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['grid.color'] = '#cccccc'
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

    # --- TOP PANEL: ANNUALIZED VOLATILITY (Blue) + PRICE (Black) ---
    
    # 1. Volatility Line (Primary - Left Axis)
    ax1.plot(x_dates, y_vol, color='#1f77b4', linewidth=1.5, label=f'Volatility ({vol_window}D)') # Standard Blue
    ax1.plot(x_dates, y_mean, color='gray', linestyle='--', alpha=0.6, linewidth=1, label='Avg Vol')
    
    # Volatility Bands
    ax1.plot(x_dates, y_up, color='red', linestyle=':', alpha=0.4)
    ax1.plot(x_dates, y_down, color='red', linestyle=':', alpha=0.4)
    ax1.fill_between(x_dates, y_down, y_up, color='red', alpha=0.05)
    
    ax1.set_ylabel('Annualized Volatility (%)', color='#1f77b4', fontsize=12, fontweight='bold')
    ax1.tick_params(axis='y', colors='#1f77b4')

    # 2. Price Line (Secondary - Right Axis)
    ax1_twin = ax1.twinx()
    ax1_twin.plot(x_dates, y_price, color='black', linewidth=2, label='Stock Price')
    ax1_twin.set_ylabel('Stock Price ($)', color='black', fontsize=12, fontweight='bold')
    ax1_twin.tick_params(axis='y', colors='black')
    
    # Combine Legends
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left', frameon=True, facecolor='white', edgecolor='gray')

    ax1.set_title(f'{ticker}: Price vs. Volatility Extremes', fontsize=16, fontweight='bold', color='black', pad=15)

    # --- BOTTOM PANEL: Z-SCORE (Purple) ---
    ax2.plot(x_dates, y_zscore, color='#9467bd', linewidth=1.5, label='Vol Z-Score') # Muted Purple

    # Horizontal Lines
    ax2.axhline(0, color='gray', linestyle='-', alpha=0.5)
    ax2.axhline(2, color='red', linestyle='--', alpha=0.7, label='+2σ (Chaotic)')
    ax2.axhline(-2, color='green', linestyle='--', alpha=0.7, label='-2σ (Quiet)')

    # Highlight Zones
    ax2.fill_between(x_dates, 2, y_zscore, where=(y_zscore >= 2), color='red', alpha=0.2, interpolate=True)
    ax2.fill_between(x_dates, -2, y_zscore, where=(y_zscore <= -2), color='green', alpha=0.2, interpolate=True)
    
    ax2.set_ylabel(f'Z-Score ({z_score_window}D)', color='#9467bd', fontsize=12, fontweight='bold')
    ax2.tick_params(axis='y', colors='#9467bd')
    ax2.legend(loc='upper left', frameon=True, facecolor='white', edgecolor='gray')
    ax2.set_xlabel('Date', color='black', fontsize=12)

    plt.tight_layout()
    plt.show()

    # REPORT
    latest_z = y_zscore[-1]
    print("\n--- VOLATILITY Z-SCORE REPORT ---")
    if latest_z > 2.0:
        print(f"ALERT: Volatility is HIGH (+{latest_z:.2f} Z-Score). Market is chaotic.")
    elif latest_z < -2.0:
        print(f"ALERT: Volatility is LOW ({latest_z:.2f} Z-Score). Market is 'asleep' (Potential Squeeze).")
    else:
        print(f"Neutral: Volatility Z-Score is {latest_z:.2f} (Normal Range).")

# --- MAIN LOOP ---
if __name__ == "__main__":
    while True:
        user_input = input("\nEnter ticker symbol (e.g., LULU, TSLA) or 'q' to quit: ").strip()
        if user_input.lower() == 'q':
            break
        if user_input:
            run_volatility_z_score_analysis(user_input)