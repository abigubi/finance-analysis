import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Optional: helps prevent label overlap
try:
    from adjustText import adjust_text  # type: ignore
except ImportError:
    adjust_text = None

# --- CONFIGURATION ---
# The "Universe" we are comparing. 
# You can add/remove tickers here to compare different sectors.
TICKERS = [
    'NVDA', 'AMD', 'INTC', 'TSM', 'AVGO', 'QCOM', 'MU',   # Semis
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA',      # Big Tech
    'NFLX', 'CRM', 'ADBE', 'ORCL', 'PLTR', 'SNOW',        # Software/Growth
    'IBM', 'CSCO', 'TXN'                                  # Legacy Tech
]

print(f"--- FETCHING FUNDAMENTALS FOR {len(TICKERS)} COMPANIES ---")
print("(This may take a moment to download data for all tickers...)")

metrics_list = []

for t in TICKERS:
    try:
        stock = yf.Ticker(t)
        info = stock.info
        
        # We need two key metrics:
        # 1. Valuation: Forward P/E (Price / Expected Earnings)
        # 2. Growth: Revenue Growth (or Earnings Growth)
        
        # You can swap 'revenueGrowth' for 'earningsGrowth' if you prefer
        growth = info.get('revenueGrowth', np.nan) 
        pe = info.get('forwardPE', np.nan)
        
        if pd.notna(growth) and pd.notna(pe):
            metrics_list.append({
                'Ticker': t,
                'Growth': growth * 100, # Convert to %
                'PE': pe
            })
            print(f"Got data for {t}")
        else:
            print(f"Skipping {t} (Missing data)")
            
    except Exception as e:
        print(f"Error fetching {t}: {e}")

# Create DataFrame
df = pd.DataFrame(metrics_list)

# Filter out extreme outliers for better visualization 
# (e.g., P/E > 150 or Growth > 100% can skew the chart)
df = df[(df['PE'] < 100) & (df['PE'] > 0)]
df = df[df['Growth'] < 100]

# --- CALCULATE REGRESSION (The "Fair Value" Line) ---
X = df[['Growth']].values.reshape(-1, 1)
y = df['PE'].values

reg = LinearRegression().fit(X, y)
r_squared = reg.score(X, y)

# Predict "Fair" P/E for every stock based on its growth
df['Fair_PE'] = reg.predict(X)
df['Discount'] = (df['Fair_PE'] - df['PE']) / df['Fair_PE'] # Positive = Undervalued

# --- PLOTTING ---
plt.figure(figsize=(14, 8))
plt.style.use('default')

# 1. Scatter Plot
# Color code: Green if Undervalued (Below line), Red if Overvalued (Above line)
colors = ['green' if row['PE'] < row['Fair_PE'] else 'red' for index, row in df.iterrows()]
plt.scatter(df['Growth'], df['PE'], c=colors, s=100, alpha=0.7, edgecolors='black')

# 2. Regression Line
range_growth = np.linspace(df['Growth'].min(), df['Growth'].max(), 100).reshape(-1, 1)
plt.plot(range_growth, reg.predict(range_growth), color='blue', linestyle='--', linewidth=2, label=f'Fair Value Line (R²={r_squared:.2f})')

# 3. Labels
texts = []
for i, txt in enumerate(df['Ticker']):
    texts.append(plt.text(df['Growth'].iloc[i], df['PE'].iloc[i], txt, fontsize=9, fontweight='bold'))

# Note: If you don't have 'adjustText' installed, standard labels might overlap.
# You can install it via 'pip install adjustText' or just let them overlap.
if adjust_text is not None:
    try:
        adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))
    except:
        pass

# Formatting
plt.title('Valuation Regression: Tech Sector (Growth vs. P/E)', fontsize=16, fontweight='bold')
plt.xlabel('Projected Revenue Growth (%)', fontsize=12)
plt.ylabel('Forward P/E Ratio', fontsize=12)
plt.axhline(0, color='black', linewidth=1)
plt.axvline(0, color='black', linewidth=1)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()

plt.tight_layout()
plt.show()

# --- REPORT ---
# Sort by "Most Undervalued" (Biggest Discount relative to growth)
df_sorted = df.sort_values(by='Discount', ascending=False)

print("\n--- GARP OPPORTUNITIES (Undervalued relative to Growth) ---")
print(df_sorted[['Ticker', 'Growth', 'PE', 'Fair_PE', 'Discount']].head(5).to_string(index=False))

print("\n--- POTENTIAL OVERVALUATION (Pricey relative to Growth) ---")
print(df_sorted[['Ticker', 'Growth', 'PE', 'Fair_PE', 'Discount']].tail(5).to_string(index=False))