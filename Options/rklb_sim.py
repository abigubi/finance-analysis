import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# --- KULLANICI AYARLARI ---
FIXED_IV_VALUE = 0.8924  # Senin verdiğin değer (%89.24)
USE_FIXED_IV = True      # True yaparsan tüm yıl boyunca bu oranı kullanır

# --- Black-Scholes Formülü ---
def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    price = (S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
    return max(price, 0.01)

# --- Parametreler ---
SYMBOL = "RKLB"
START_DATE = "2024-01-01"
END_DATE = pd.Timestamp.today().strftime('%Y-%m-%d')
DTE = 35           # 35 Gün vade
OTM_PCT = 1.11     # %11 Yukarıdan (Strike belirleme)
RISK_FREE = 0.045  # Risksiz faiz

# 1. Veri İndirme
print(f"{SYMBOL} verileri indiriliyor...")
data = yf.download(SYMBOL, start="2023-10-01", end=END_DATE, progress=False)
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

# 2. Dinamik Volatilite Hesapla (Yine de arka planda dursun)
data['Log_Ret'] = np.log(data['Close'] / data['Close'].shift(1))
data['HV_Dynamic'] = data['Log_Ret'].rolling(window=30).std() * np.sqrt(252)

df = data.loc[START_DATE:].copy().dropna(subset=['HV_Dynamic'])

# --- SİMÜLASYON ---
dates = df.index
current_idx = 0
initial_price = df['Close'].iloc[0]
shares = 100
start_capital = initial_price * shares

portfolio_cc = [start_capital]
portfolio_bh = [start_capital]
trade_dates = [dates[0]]

cash_balance = 0 
total_premium_collected = 0 # Toplam ne kadar prim topladığımızı görmek için

print(f"Simülasyon Başlıyor... Mod: {'SABİT IV (%.2f)' % FIXED_IV_VALUE if USE_FIXED_IV else 'DİNAMİK HV'}")

while current_idx < len(dates) - DTE:
    # GİRİŞ
    entry_date = dates[current_idx]
    entry_price = df['Close'].iloc[current_idx]
    
    # IV Seçimi
    if USE_FIXED_IV:
        current_iv = FIXED_IV_VALUE # Senin istediğin sabit 89.24%
    else:
        # Gerçekçi mod: Tarihsel volatilite + %10 risk primi
        current_iv = df['HV_Dynamic'].iloc[current_idx] * 1.1
        
    # Strike ve Prim Hesabı
    strike = entry_price * OTM_PCT
    premium_per_share = black_scholes_call(entry_price, strike, DTE/365, RISK_FREE, current_iv)
    
    total_premium = premium_per_share * shares
    cash_balance += total_premium
    total_premium_collected += total_premium
    
    # ÇIKIŞ (Vade Sonu)
    exit_idx = current_idx + DTE
    if exit_idx >= len(dates): break
    
    exit_date = dates[exit_idx]
    exit_price = df['Close'].iloc[exit_idx]
    
    # SONUÇLAR
    bh_value = shares * exit_price # Buy & Hold Değeri
    
    # Covered Call Değeri Hesaplama
    if exit_price < strike:
        status = "HİSSE KALDI"
        # Nakit (Primler) + Hisse Değeri
        cc_value = (shares * exit_price) + cash_balance
    else:
        status = "HİSSE GİTTİ"
        # Hisse Satışı (Strike'tan) + Nakit (Primler)
        cash_from_sale = shares * strike
        total_cash_now = cash_from_sale + cash_balance
        
        # Yeniden Giriş (Re-buy)
        cc_value = total_cash_now
        cost_to_rebuy = shares * exit_price
        cash_balance = total_cash_now - cost_to_rebuy # Aradaki fark kasadan düşer
        
    portfolio_cc.append(cc_value)
    portfolio_bh.append(bh_value)
    trade_dates.append(exit_date)
    
    # Döngü İlerleme
    current_idx = exit_idx

# --- GRAFİK VE SONUÇ ---
plt.figure(figsize=(12, 6))
plt.plot(trade_dates, portfolio_bh, label='Buy & Hold (Sadece Tut)', color='orange', linestyle='--')
plt.plot(trade_dates, portfolio_cc, label=f'Covered Call (IV={current_iv:.2f})', color='green', linewidth=2.5)

plt.title(f"{SYMBOL} - Simülasyon (IV: {'Sabit %89.24' if USE_FIXED_IV else 'Dinamik'})")
plt.ylabel("Portföy Değeri ($)")
plt.legend()
plt.grid(True, alpha=0.5)

# Bilgi Kutusu
final_cc = portfolio_cc[-1]
final_bh = portfolio_bh[-1]
diff = final_cc - final_bh

result_text = (f"Bitiş Değerleri:\n"
               f"Sadece Tut: ${final_bh:.0f}\n"
               f"Covered Call: ${final_cc:.0f}\n"
               f"Fark: ${diff:.0f}\n"
               f"Toplanan Prim: ${total_premium_collected:.0f}")

plt.figtext(0.15, 0.7, result_text, bbox=dict(facecolor='white', alpha=0.9, edgecolor='black'))

plt.tight_layout()
plt.show()