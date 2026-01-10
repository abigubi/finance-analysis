import yfinance as yf
import pandas as pd
import seaborn as sns  # type: ignore
import matplotlib.pyplot as plt
import numpy as np

# --- CONFIGURATION ---
# Default list, but the script will ask for your input
DEFAULT_TICKERS = ['NVDA', 'AMD', 'MSFT', 'GOOGL', 'TSLA', 'JPM', 'XOM', 'GLD', 'SPY']
PERIOD = "1y"  # Look back 1 year to see recent correlations

def get_user_tickers():
    print("Enter tickers separated by spaces (e.g., 'AAPL MSFT TSLA')")
    print("Or just press ENTER to use the default list.")
    user_input = input("Tickers: ").strip()
    if not user_input:
        return DEFAULT_TICKERS
    # Clean up input (remove commas, uppercase)
    return [t.upper().strip() for t in user_input.replace(',', ' ').split()]

def plot_correlation_heatmap():
    tickers = get_user_tickers()
    if len(tickers) < 2:
        print("Need at least 2 tickers to calculate correlation.")
        return

    print(f"\n--- DOWNLOADING DATA FOR: {', '.join(tickers)} ---")
    
    # 1. GET DATA
    try:
        data = yf.download(tickers, period=PERIOD, progress=False, auto_adjust=False)['Adj Close']
    except Exception as e:
        print(f"Error downloading: {e}")
        return

    # Handle single ticker edge case or bad downloads
    if data.empty:
        print("No data found.")
        return
    
    # Reorder columns to match input order (filter out any that failed to download)
    available_tickers = [t for t in tickers if t in data.columns]
    if len(available_tickers) < 2:
        print("Need at least 2 tickers with valid data.")
        return
    
    # Reorder data to match input order
    data = data[available_tickers]
    
    # Calculate Daily Returns
    returns = data.pct_change().dropna()
    
    # 2. CALCULATE CORRELATION MATRIX
    corr_matrix = returns.corr()
    
    # Ensure correlation matrix is in the same order as input tickers
    corr_matrix = corr_matrix.reindex(index=available_tickers, columns=available_tickers)

    # 3. PLOT HEATMAP
    plt.figure(figsize=(10, 8))
    
    # Create a mask to hide the upper triangle (optional, but makes it cleaner)
    # mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    # Custom color map: Red = Highly Correlated (Danger), Blue = Negative Correlation (Hedge)
    sns.heatmap(corr_matrix, 
                annot=True,         # Show the numbers
                fmt=".2f",          # 2 decimal places
                cmap='coolwarm',    # Red (Hot) to Blue (Cold)
                vmin=-1, vmax=1,    # Scale from -1 to 1
                center=0,           # White is 0 correlation
                square=True, 
                linewidths=.5)

    plt.title(f'Portfolio Correlation Matrix ({PERIOD} Lookback)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # 4. REDUNDANCY REPORT
    print("\n--- REDUNDANCY REPORT (Risk Check) ---")
    
    # Flatten the matrix to find pairs
    c = corr_matrix.abs()
    s = c.unstack()
    # Sort and remove self-correlations (which are always 1.0)
    so = s.sort_values(kind="quicksort", ascending=False)
    so = so[so < 1.0]

    # Remove duplicates (A-B is same as B-A)
    seen = set()
    high_corr_pairs = []
    
    for index, value in so.items():
        pair = tuple(sorted(index))
        if pair not in seen:
            seen.add(pair)
            if value > 0.80:
                high_corr_pairs.append((pair, value))
    
    if high_corr_pairs:
        print("⚠️ HIGH CORRELATION WARNING (These move together >80% of the time):")
        for pair, val in high_corr_pairs[:5]: # Top 5 risks
            print(f"  {pair[0]} <--> {pair[1]}: {val:.2f}")
        print("  -> Holding both gives you limited diversification.")
    else:
        print("✅ Good Diversification: No pairs have > 0.80 correlation.")

if __name__ == "__main__":
    plot_correlation_heatmap()