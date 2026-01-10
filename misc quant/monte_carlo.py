import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

def run_monte_carlo(ticker, years_history=2, simulation_days=252, num_simulations=5000):
    print(f"\n--- FETCHING DATA FOR {ticker.upper()} ---")
    
    # 1. GET DATA
    try:
        start_date = (pd.Timestamp.now() - pd.DateOffset(years=years_history)).strftime('%Y-%m-%d')
        data = yf.download(ticker, start=start_date, progress=False, auto_adjust=False)
        
        # Handle the different yfinance data structures
        if data.empty:
            print(f"Error: No data found for {ticker}.")
            return

        if isinstance(data.columns, pd.MultiIndex):
            price_series = data.xs('Adj Close', axis=1, level=0)
            if isinstance(price_series, pd.DataFrame):
                price_series = price_series.iloc[:, 0]
        else:
            if 'Adj Close' in data.columns:
                price_series = data['Adj Close']
            elif 'Close' in data.columns:
                price_series = data['Close']
            else:
                price_series = data.iloc[:, 0]
                
        price_series = price_series.dropna()
        current_price = price_series.iloc[-1]
        
    except Exception as e:
        print(f"Error downloading data: {e}")
        return

    # 2. CALCULATE STATISTICS
    # Log Returns
    log_returns = np.log(1 + price_series.pct_change())

    # Drift & Volatility
    mu = log_returns.mean()
    var = log_returns.var()
    drift = mu - (0.5 * var)
    sigma = log_returns.std()
    
    # Annualized Volatility for context
    annual_vol = sigma * np.sqrt(252)

    # 3. RUN SIMULATIONS
    daily_returns = np.exp(drift + sigma * np.random.normal(0, 1, (simulation_days, num_simulations)))
    
    price_paths = np.zeros_like(daily_returns)
    price_paths[0] = current_price

    for t in range(1, simulation_days):
        price_paths[t] = price_paths[t-1] * daily_returns[t]

    # 4. CALCULATE PERCENTILES
    p05 = np.percentile(price_paths, 5, axis=1)   # Worst 5%
    p25 = np.percentile(price_paths, 25, axis=1)  # Bearish
    p50 = np.percentile(price_paths, 50, axis=1)  # Median
    p75 = np.percentile(price_paths, 75, axis=1)  # Bullish
    p95 = np.percentile(price_paths, 95, axis=1)  # Best 5%

    # 5. RISK REPORT
    risk_downside = (p05[-1] - current_price) / current_price
    upside_potential = (p95[-1] - current_price) / current_price
    risk_reward = abs(upside_potential / risk_downside) if risk_downside != 0 else 0

    print(f"\n--- QUANT RISK REPORT: {ticker.upper()} (12 Month Horizon) ---")
    print(f"Current Price:       ${current_price:.2f}")
    print(f"Annualized Volatility: {annual_vol*100:.1f}%")
    if annual_vol > 0.40: print("WARNING: This stock is Highly Volatile (High Uncertainty).")
    print(f"------------------------------------------------")
    print(f"Median Target:       ${p50[-1]:.2f}")
    print(f"Worst Case (Top 5%): ${p05[-1]:.2f} ({risk_downside*100:.1f}%)")
    print(f"Best Case (Top 5%):  ${p95[-1]:.2f} (+{upside_potential*100:.1f}%)")
    print(f"------------------------------------------------")
    print(f"Risk/Reward Ratio:   {risk_reward:.2f} (Target > 2.0 is ideal)")

    # 6. PLOT
    plt.figure(figsize=(12, 6))
    
    # Dates
    last_date = price_series.index[-1]
    future_dates = [last_date + pd.Timedelta(days=x) for x in range(simulation_days)]
    
    # Plot History (Last 100 days)
    plt.plot(price_series.index[-100:], price_series.iloc[-100:], color='black', linewidth=2, label='History')
    
    # Plot Cone
    plt.fill_between(future_dates, p05, p95, color='gray', alpha=0.2, label='90% Confidence (Wild scenarios)')
    plt.fill_between(future_dates, p25, p75, color='blue', alpha=0.3, label='50% Confidence (Likely scenarios)')
    plt.plot(future_dates, p50, color='blue', linestyle='--', linewidth=2, label='Median Path')

    plt.title(f'Monte Carlo Projection: {ticker.upper()}\n(Volatility: {annual_vol*100:.1f}%)', fontsize=14)
    plt.ylabel('Price ($)')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left')
    
    # Format Y-axis to dollars
    formatter = FuncFormatter(lambda x, pos: f'${x:.0f}')
    plt.gca().yaxis.set_major_formatter(formatter)
    
    plt.show()

# --- MAIN LOOP ---
if __name__ == "__main__":
    while True:
        user_input = input("\nEnter ticker symbol (or 'q' to quit): ").strip()
        if user_input.lower() == 'q':
            break
        if user_input:
            run_monte_carlo(user_input)