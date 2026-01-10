import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
from scipy.stats import norm

# ==========================
# CONFIG
# ==========================
XLSX_PATH = r"C:\Users\kerem\OneDrive\Masaüstü\Finance Analysis\Options\Amazon_10_IV.xlsx"

START_DATE = "2025-01-01"
END_DATE   = "2025-12-15"   # will stop at last date available in Excel

DTE = 30
RISK_FREE_RATE = 0.045

START_WITH_SHARES = True  # True = start with 100 shares (CC first)

COL_DATE  = "Date"
COL_CLOSE = "Amazon.com Price - Close"
COL_IV_PUT_10OTM  = "Modeled IV (90%), 30 DTE - Modeled IV (90%), 30 DTE"
COL_IV_CALL_10OTM = "Modeled IV (110%), 30 DTE - Modeled IV (110%), 30 DTE"

# ==========================
# Black–Scholes
# ==========================
def bs_price(opt_type, S, K, T, r, sigma):
    if T <= 0:
        return max(0.0, S - K) if opt_type == "c" else max(0.0, K - S)
    sigma = max(1e-9, float(sigma))
    d1 = (np.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if opt_type == "c":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

# ==========================
# LOAD DATA
# ==========================
ivdf = pd.read_excel(XLSX_PATH)
ivdf[COL_DATE] = pd.to_datetime(ivdf[COL_DATE])
ivdf = ivdf.sort_values(COL_DATE).set_index(COL_DATE)

df = pd.DataFrame({
    "Close": ivdf[COL_CLOSE].astype(float),
    "IV_put":  ivdf[COL_IV_PUT_10OTM].astype(float) / 100.0,
    "IV_call": ivdf[COL_IV_CALL_10OTM].astype(float) / 100.0,
}).loc[START_DATE:END_DATE].dropna()

if df.empty:
    raise ValueError("No data after filtering dates. Check START_DATE/END_DATE.")

dates = df.index.to_list()

# ==========================
# WHEEL SIM
# ==========================
def run_wheel(df, dte=30, r=0.045, start_with_shares=True):
    dates = df.index.to_list()
    
    # Initialize with proper capital tracking
    initial_price = float(df.loc[dates[0], "Close"])
    
    # Calculate required capital for both scenarios
    cost_of_100_shares = 100 * initial_price
    put_strike = 0.90 * initial_price
    cost_to_secure_put = put_strike * 100 * 1.1  # 10% buffer
    
    # Use the larger amount so both strategies can operate
    initial_capital = max(cost_of_100_shares, cost_to_secure_put)
    
    if start_with_shares:
        shares = 100
        cash = initial_capital - cost_of_100_shares  # Remaining cash after buying shares
    else:
        shares = 0
        cash = initial_capital  # All cash to secure puts
    
    mode = "shares" if start_with_shares else "cash"  # shares->sell calls, cash->sell puts
    
    # Benchmark: Buy 100 shares on day 1 with same initial capital
    bench_shares = 100
    bench_cash = initial_capital - cost_of_100_shares  # Remaining cash after buying shares

    port_dates, port_vals, bench_vals = [], [], []

    idx = 0
    expiry_idx = -1
    expiry_date = None
    strike = None
    opt_type = None

    while idx < len(dates):

        # Open new option if possible
        if idx > expiry_idx:
            d0 = dates[idx]
            S0 = float(df.loc[d0, "Close"])

            target = d0 + timedelta(days=dte)
            future_i = next((i for i, d in enumerate(dates) if d >= target), None)

            if future_i is None:
                # Can't open another full 30DTE trade — stop selling options, just hold to end
                for j in range(idx, len(dates)):
                    d = dates[j]
                    S = float(df.loc[d, "Close"])
                    total = cash + shares * S
                    port_dates.append(d)
                    port_vals.append(total)
                    bench_vals.append(bench_cash + bench_shares * S)
                break

            expiry_idx = future_i
            expiry_date = dates[expiry_idx]

            T = dte / 365.0

            if mode == "cash":
                strike = 0.90 * S0
                opt_type = "p"
                sigma0 = float(df.loc[d0, "IV_put"])
            else:
                strike = 1.10 * S0
                opt_type = "c"
                sigma0 = float(df.loc[d0, "IV_call"])

            premium = bs_price(opt_type, S0, strike, T, r, sigma0) * 100.0
            cash += premium

        # Mark-to-market daily until expiry (or end)
        # Process all days up to and including expiry
        end_j = min(expiry_idx, len(dates) - 1)
        for j in range(idx, end_j + 1):
            d = dates[j]
            S = float(df.loc[d, "Close"])
            
            # On expiry day, option value is intrinsic only
            if j == expiry_idx:
                if opt_type == "c":
                    liab = max(0, S - strike) * 100.0
                else:
                    liab = max(0, strike - S) * 100.0
            else:
                # Before expiry, use Black-Scholes with current IV
                sigma = float(df.loc[d, "IV_put"] if opt_type == "p" else df.loc[d, "IV_call"])
                days_rem = max(1, (expiry_date - d).days)
                T_cur = max(1 / 365.0, days_rem / 365.0)
                liab = bs_price(opt_type, S, strike, T_cur, r, sigma) * 100.0

            total = cash + shares * S - liab

            port_dates.append(d)
            port_vals.append(total)
            bench_vals.append(bench_cash + bench_shares * S)

        # Settle at expiry (handle assignment)
        if expiry_idx < len(dates):
            settle_S = float(df.loc[expiry_date, "Close"])

            # Settlement: handle assignment
            if mode == "cash":
                if settle_S < strike:
                    cash -= strike * 100.0
                    shares = 100
                    mode = "shares"
            else:
                if settle_S > strike:
                    cash += strike * 100.0
                    shares = 0
                    mode = "cash"

        idx = expiry_idx + 1

    return pd.Series(port_vals, index=port_dates), pd.Series(bench_vals, index=port_dates)

wheel, bh = run_wheel(df, dte=DTE, r=RISK_FREE_RATE, start_with_shares=START_WITH_SHARES)

print(f"Backtest ends on: {wheel.index[-1].date()} (last date available in Excel range)")
print(f"Final Wheel:    ${wheel.iloc[-1]:,.2f}")
print(f"Final Buy&Hold: ${bh.iloc[-1]:,.2f}")
print(f"Diff ($):       ${wheel.iloc[-1]-bh.iloc[-1]:,.2f}")
print(f"Diff (%):       {(wheel.iloc[-1]/bh.iloc[-1]-1)*100:.2f}%")

plt.figure(figsize=(12,6))
plt.plot(wheel.index, wheel.values, label="Wheel (10% OTM IV)", linewidth=2)
plt.plot(bh.index, bh.values, label="Buy & Hold (100 shares)", linestyle="--", alpha=0.7)
plt.title("AMZN 2025 YTD: Wheel vs Buy & Hold (10% OTM, 30DTE)")
plt.xlabel("Date")
plt.ylabel("Portfolio Value ($)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
