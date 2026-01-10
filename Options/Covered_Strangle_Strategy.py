import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from py_vollib.black_scholes import black_scholes

# ==========================================
# CONFIGURATION
# ==========================================
TICKER = "NVDA"
START_DATE = "2020-01-01"  # The start of the massive AI rally
END_DATE = "2025-12-15"

# --- STRATEGY SETTINGS ---
DAYS_TO_EXPIRATION = 30    # Monthly Options
TARGET_DELTA = 0.30        # 30 Delta (Standard)
FIXED_IV = 0.55            # 55% Volatility (Realistic for NVDA)
RISK_FREE_RATE = 0.045     # 4.5% Risk Free Rate

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def get_strike_from_delta(S, delta, T, r, sigma, option_type):
    if option_type == 'c':
        d1 = norm.ppf(delta)
    else:
        d1 = norm.ppf(1 - delta)
    vol_sqrt_t = sigma * np.sqrt(T)
    drift = (r + 0.5 * sigma**2) * T
    K = np.exp(np.log(S) - ((d1 * vol_sqrt_t) - drift))
    return K

# ==========================================
# 1. DATA FETCHING
# ==========================================
print(f"\nFetching data for {TICKER}...")
# Fetch extra data to ensure we have a valid start/end buffer
raw_data = yf.download(TICKER, start="2020-12-01", end="2026-01-01", progress=False)

if isinstance(raw_data.columns, pd.MultiIndex):
    raw_data.columns = raw_data.columns.get_level_values(0)

df = raw_data.loc[START_DATE:END_DATE].copy()
dates = df.index

if len(df) == 0:
    raise ValueError(f"No data found for {TICKER}. Check your date range.")

# ==========================================
# 2. BACKTEST ENGINE
# ==========================================
shares = 100
# cash starts at 0. Positive cash = Profit from premiums. Negative cash = Loss from buybacks.
cash = 0 

portfolio_values = []
buy_hold_values = []

current_call_strike = None
current_put_strike = None
expiry_idx = 0
idx = 0

print("-" * 100)
print(f"{'DATE':<12} | {'ACTION':<15} | {'PRICE':<11} | {'DETAILS'}")
print("-" * 100)

while idx < len(dates):
    current_date = dates[idx]
    current_date_str = current_date.strftime('%Y-%m-%d')
    price = df['Close'].iloc[idx]
    
    # --- A. OPEN NEW TRADES (Start of Cycle) ---
    # Logic: If we are past the previous expiry (or at start), open new positions
    if idx >= expiry_idx:
        T_start = DAYS_TO_EXPIRATION / 365.0
        
        # 1. Find Calendar Expiry Date
        target_date = current_date + pd.Timedelta(days=DAYS_TO_EXPIRATION)
        future_dates = raw_data.index[raw_data.index >= target_date]
        
        if len(future_dates) == 0: break # End of simulation
        expiry_date = future_dates[0]
        
        # Verify expiry is within our dataframe range
        if expiry_date not in df.index: break
        new_expiry_idx = df.index.get_loc(expiry_date)
        
        # 2. Determine Strikes
        current_call_strike = get_strike_from_delta(price, TARGET_DELTA, T_start, RISK_FREE_RATE, FIXED_IV, 'c')
        current_put_strike = get_strike_from_delta(price, TARGET_DELTA, T_start, RISK_FREE_RATE, FIXED_IV, 'p')
        
        # 3. Sell Options (Collect Premium)
        call_prem = black_scholes('c', price, current_call_strike, T_start, RISK_FREE_RATE, FIXED_IV)
        put_prem = black_scholes('p', price, current_put_strike, T_start, RISK_FREE_RATE, FIXED_IV)
        
        total_credit = (call_prem + put_prem) * 100
        cash += total_credit
        
        print(f"{current_date_str:<12} | OPEN TRADES     | ${price:<10.2f} | Sold Call ${current_call_strike:.2f} / Put ${current_put_strike:.2f} (Credit +${total_credit:.2f})")
        
        expiry_idx = new_expiry_idx # Update global pointer

    # --- B. DAILY VALUATION LOOP (Fill data until Expiry) ---
    # We run a mini-loop to fill values for every day between Open and Expiry
    # This ensures the graph is smooth and captures the "Cap" in real-time
    
    start_fill = idx
    end_fill = expiry_idx 
    
    for i in range(start_fill, end_fill + 1):
        if i >= len(dates): break
        
        d_price = df['Close'].iloc[i]
        d_date = dates[i]
        
        # Calculate time remaining
        days_remaining = (expiry_date - d_date).days
        T_current = max(0.001, days_remaining / 365.0)
        
        # --- LIABILITY CALCULATION ---
        # 1. Short Call Liability
        curr_call_val = black_scholes('c', d_price, current_call_strike, T_current, RISK_FREE_RATE, FIXED_IV) * 100
        
        # 2. Short Put Liability
        curr_put_val = black_scholes('p', d_price, current_put_strike, T_current, RISK_FREE_RATE, FIXED_IV) * 100
        
        total_liability = curr_call_val + curr_put_val
        
        # --- PORTFOLIO VALUE ---
        # Net Liq = (Stock Value) + (Cash collected) - (Cost to close options)
        strat_val = (shares * d_price) + cash - total_liability
        portfolio_values.append(strat_val)
        
        # Benchmark
        buy_hold_values.append(shares * d_price)
        
    # --- C. SETTLEMENT AT EXPIRATION ---
    settlement_price = df['Close'].iloc[expiry_idx]
    outcome_msg = ""
    
    # 1. Call Settlement
    if settlement_price > current_call_strike:
        # LOSS: You effectively sold shares at Strike, but must buy them back at Market Price
        loss = (settlement_price - current_call_strike) * 100
        cash -= loss # Realized Loss
        outcome_msg += f"Call ITM (Loss -${loss:.2f}) "
    else:
        outcome_msg += "Call Expired "

    # 2. Put Settlement
    if settlement_price < current_put_strike:
        # LOSS: You buy shares at Strike (above market), immediately sell at Market
        loss = (current_put_strike - settlement_price) * 100
        cash -= loss # Realized Loss
        outcome_msg += f"Put ITM (Loss -${loss:.2f})"
    else:
        outcome_msg += "Put Expired"
        
    print(f"{dates[expiry_idx].strftime('%Y-%m-%d'):<12} | SETTLEMENT      | ${settlement_price:<10.2f} | {outcome_msg}")

    # Jump main loop to the day after expiration
    idx = expiry_idx + 1

# ==========================================
# 3. RESULTS & PLOT
# ==========================================
# Trim arrays to match
limit = len(portfolio_values)
plot_dates = dates[:limit]
buy_hold_values = buy_hold_values[:limit]

final_strat = portfolio_values[-1]
final_bh = buy_hold_values[-1]

print("-" * 100)
print(f"FINAL RESULTS ({START_DATE} to {END_DATE})")
print(f"Strategy:    ${final_strat:,.2f}")
print(f"Buy & Hold:  ${final_bh:,.2f}")
print(f"Difference:  ${final_strat - final_bh:,.2f}")
print("-" * 100)

plt.figure(figsize=(12, 6))
# Strategy in Blue
plt.plot(plot_dates, portfolio_values, label='Covered Strangle (Strategy)', color='blue', linewidth=2)
# Buy & Hold in Gray
plt.plot(plot_dates, buy_hold_values, label='Buy & Hold NVDA', color='gray', linestyle='--', alpha=0.6)

plt.title(f'NVDA: The "Performance Drag" of Covered Calls (IV {FIXED_IV*100:.0f}%)')
plt.ylabel('Portfolio Value ($)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()