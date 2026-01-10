import numpy as np
import matplotlib.pyplot as plt

# --- Strateji Parametreleri ---
STOCK_NAME = "NBIS"
S0 = 98.04  # Başlangıç Hisse Senedi Alış Fiyatı
K = 110     # Satılan Kısa Alım Opsiyonunun Kullanım Fiyatı
Cp = 2.75   # Opsiyon Satışından Elde Edilen Prim
DTE = 11    # Days to Expiration (Dec 19)

# --- Hisse Senedi Fiyat Aralığı Ayarları (Ortalanmış Aralık) ---
# Vade sonundaki hisse senedi fiyat aralığı (ST) üzerinden getiri grafiği çizilecek
# Grafik strike price (K) etrafında ortalanmıştır
range_width = 75  # Strike price'ın her iki tarafında 75 birim
ST_min = K - range_width  # 110 - 75 = 35
ST_max = K + range_width  # 110 + 75 = 185
ST = np.linspace(ST_min, ST_max, 200) # Ortalanmış aralık: $35'den $185'e

# --- Hesaplanan Temel Metrikler ---
break_even = S0 - Cp # 100 - 2.00 = 98.00
max_profit = K - S0 + Cp # 110 - 100 + 2.00 = 12.00
# Maksimum Zarar ST = 0 olduğunda oluşur (Teorik)
# ST = 0 olduğunda: Uzun hisse = -S0, Short call = Cp, Toplam = -S0 + Cp
theoretical_max_loss = -S0 + Cp  # -98.04 + 2.75 = -95.29
max_loss = theoretical_max_loss

# --- Getiri Fonksiyonları ---

# 1. Uzun Hisse Senedi Getirisi
# Getiri = ST - Başlangıç Maliyeti
long_stock_payoff = ST - S0

# 2. Kısa Alım Opsiyonu Getirisi
# Getiri = Alınan Prim - max(0, ST - K)
short_call_payoff = Cp - np.maximum(0, ST - K)

# 3. Covered Call Getirisi (Net Getiri)
# Covered Call = Uzun Hisse Senedi Getirisi + Kısa Alım Opsiyonu Getirisi
covered_call_payoff = long_stock_payoff + short_call_payoff


# --- Getiri Grafiği Çizimi ---
def plot_covered_call_payoff():
    """Covered Call getiri grafiğini oluşturur ve görüntüler."""
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    
    # 1. Bireysel bileşenleri ve net stratejiyi çiz
    
    # Uzun Hisse Senedi (Kesikli Turuncu)
    plt.plot(ST, long_stock_payoff, 
             label='Long Hisse Senedi', 
             linestyle='--', 
             color='orange', 
             linewidth=2)

    # Kısa Alım Opsiyonu (Kesikli Koyu Mavi)
    plt.plot(ST, short_call_payoff, 
             label='Short Call Opsiyonu', 
             linestyle='--', 
             color='darkblue', 
             linewidth=2)

    # Covered Call (Düz Yeşil - Net Getiri)
    plt.plot(ST, covered_call_payoff, 
             label='Covered Call (Net Getiri)', 
             linestyle='-', 
             color='green', 
             linewidth=3)

    # 2. Gölgeli Üstün Performans Bölgelerini Ekle (Covered Call > Uzun Hisse Senedi)
    
    # Covered Call'un Hisse Senedinden üstün olduğu yerleri doldur (Alt Uç)
    outperformance_mask = covered_call_payoff > long_stock_payoff
    plt.fill_between(ST, covered_call_payoff, long_stock_payoff, 
                     where=outperformance_mask, 
                     color='lightgreen', 
                     alpha=0.6, 
                     label='Covered Call, Hisse Senedinden Üstün')
    
    # Hisse Senedinin Covered Call'dan üstün olduğu yerleri doldur (Üst Uç)
    stock_outperformance_mask = long_stock_payoff > covered_call_payoff
    plt.fill_between(ST, covered_call_payoff, long_stock_payoff, 
                     where=stock_outperformance_mask, 
                     color='lightcoral', 
                     alpha=0.6, 
                     label='Hisse Senedi, Covered Call\'dan Üstün')


    # 3. Temel İşaretçileri ve Çizgileri Ekle

    # Sıfır çizgisi (X ekseni)
    plt.axhline(0, color='black', linewidth=1, linestyle='-')

    # Kullanım Fiyatı Dikey Çizgisi
    plt.axvline(K, 
                color='firebrick', 
                linestyle='--', 
                linewidth=2, 
                label=f'Kullanım Fiyatı (${K})')

    # Başa Baş Noktası (Covered Call çizgisi üzerinde)
    plt.plot(break_even, 0, 
             marker='x', 
             color='purple', 
             markersize=10, 
             markeredgewidth=2, 
             label=f'Başa Baş Noktası (${break_even:.2f})')
    
    # Maksimum Kâr Noktası (kullanım fiyatında)
    plt.plot(K, max_profit, 
             marker='X', 
             color='green', 
             markersize=10, 
             markeredgewidth=2, 
             label=f'Maksimum Kâr (${max_profit:.2f})')
    
    # Maksimum Zarar Noktası (Teorik - ST = 0'da)
    plt.plot(0, theoretical_max_loss, 
             marker='o', 
             color='red', 
             markersize=8, 
             markeredgewidth=2, 
             label=f'Maksimum Zarar (Teorik: ${theoretical_max_loss:.2f})')


    # 4. Son Rötuşlar
    plt.title(f'Covered Call Getiri Grafiği - {STOCK_NAME}', fontsize=16, fontweight='bold')
    plt.xlabel('Vade Sonu Hisse Senedi Fiyatı ($S_T$)', fontsize=12)
    plt.ylabel('Kâr / Zarar ($)', fontsize=12)
    
    # Eksen limitlerini verilere göre ayarla
    y_min = np.min([np.min(long_stock_payoff), np.min(covered_call_payoff)]) - 10
    y_max = np.max([np.max(long_stock_payoff), np.max(covered_call_payoff)]) + 5
    plt.ylim(y_min, y_max)
    plt.xlim(ST.min(), ST.max())

    # Sol ortaya strateji parametrelerini ekle (legend'in altında)
    info_text = f'Hisse Fiyatı: ${S0:.2f}\nKullanım Fiyatı: ${K}\nPremium: ${Cp:.2f}\nDTE: {DTE}'
    plt.text(0.02, 0.32, info_text,
             transform=ax.transAxes, 
             fontsize=10,
             verticalalignment='center',
             horizontalalignment='left',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Izgara ve gösterge görünümünü özelleştir
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper left', frameon=True, shadow=True)
    plt.tight_layout()
    plt.show()

# Çizim fonksiyonunu çalıştır
plot_covered_call_payoff()