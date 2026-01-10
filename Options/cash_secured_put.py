import numpy as np
import matplotlib.pyplot as plt

# --- Strateji Parametreleri ---
STOCK_NAME = "RKLB"
S0 = 63.11   # Mevcut Hisse Fiyatı (Giriş)
K = 70       # Satılan Call Opsiyonu Kullanım Fiyatı (Strike)
Premium = 4.75  # Alınan Prim (Bid fiyatı)

# --- Grafik İçin Fiyat Aralığı ---
# İsteğin üzerine aralığı 120 olarak ayarladık
ST_min = 40.0 
ST_max = 120.0
ST = np.linspace(ST_min, ST_max, 200)

# --- Hesaplanan Temel Metrikler ---
break_even = S0 - Premium  # 63.11 - 4.75 = 58.36
max_profit = (K - S0) + Premium # (70 - 63.11) + 4.75 = 11.64

# Şu anki fiyatta vade sonu kârı (Hisse hareket etmezse sadece primi kazanırız)
current_price_profit = Premium 

# --- Getiri Fonksiyonları ---
long_stock_payoff = ST - S0
covered_call_payoff = (ST - S0) + Premium - np.maximum(0, ST - K)

# --- Grafik Çizimi ---
def plot_cc_payoff():
    plt.figure(figsize=(12, 7))
    ax = plt.gca()
    
    # 1. Çizgiler
    plt.plot(ST, long_stock_payoff, label='Sadece Hisse (Long Stock)', 
             linestyle='--', color='orange', linewidth=2)

    plt.plot(ST, covered_call_payoff, label='Covered Call (Net Getiri)', 
             linestyle='-', color='green', linewidth=3)

    # 2. Alanları Doldurma
    
    # Yeşil Alan (Covered Call Üstün)
    cc_wins_mask = covered_call_payoff > long_stock_payoff
    plt.fill_between(ST, covered_call_payoff, long_stock_payoff, 
                     where=cc_wins_mask, color='lightgreen', alpha=0.5)
    
    # Kırmızı Alan (Hisse Üstün)
    stock_wins_mask = long_stock_payoff > covered_call_payoff
    plt.fill_between(ST, covered_call_payoff, long_stock_payoff, 
                     where=stock_wins_mask, color='lightcoral', alpha=0.5)

    # 3. Alan Yazıları (YENİ EKLENDİ)
    # Yeşil alanın içine yazı
    plt.text(52, -2, "Covered Call\nDaha İyi", 
             color='darkgreen', fontsize=11, fontweight='bold', ha='center', rotation=15)

    # Kırmızı alanın içine yazı
    plt.text(95, 25, "Hisse Tutmak\nDaha İyi", 
             color='darkred', fontsize=11, fontweight='bold', ha='center', rotation=25)


    # 4. Önemli Noktalar
    # Sıfır Çizgisi
    plt.axhline(0, color='black', linewidth=1)

    # Strike
    plt.axvline(K, color='gray', linestyle=':', linewidth=2, label=f'Strike (${K})')

    # MAVİ NOKTA (Çizgisiz, sadece nokta ve yazı)
    plt.plot(S0, current_price_profit, marker='o', color='blue', markersize=9)
    # Yazıyı noktanın biraz sağına ve altına koyalım ki karışmasın
    plt.text(S0 + 1, current_price_profit - 2, f"Şu Anki Yer\n(Kâr: ${Premium})", 
             color='blue', fontweight='bold', fontsize=9)

    # Başa Baş
    plt.plot(break_even, 0, marker='o', color='red', markersize=8, 
             label=f'Başa Baş (${break_even:.2f})')
    
    # Maksimum Kâr
    plt.plot(K, max_profit, marker='*', color='gold', markersize=15, markeredgecolor='black',
             label=f'Maks. Kâr (${max_profit:.2f})')

    # 5. Düzenlemeler
    plt.title(f'Covered Call Analizi: {STOCK_NAME}', fontsize=16, fontweight='bold')
    plt.xlabel('Vade Sonu Hisse Fiyatı ($)', fontsize=12)
    plt.ylabel('Kâr / Zarar ($)', fontsize=12)
    plt.xlim(ST_min, ST_max)
    
    # Bilgi Kutusu
    info_text = (f'Giriş: ${S0}\n'
                 f'Strike: ${K}\n'
                 f'Prim: ${Premium}\n'
                 f'Başa Baş: ${break_even:.2f}\n'
                 f'Maks. Kâr: ${max_profit:.2f}')
    
    plt.text(0.02, 0.95, info_text, transform=ax.transAxes, fontsize=11,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_cc_payoff()