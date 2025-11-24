import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
df = pd.read_csv('car_price.csv')

# Convert price to numeric (handles Lakh/Crore)
def convert_price(s: str):
    s = str(s).replace(',', '').strip()
    if 'Crore' in s:
        try:
            return float(s.replace('Crore', '').strip()) * 10000000
        except Exception:
            return np.nan
    if 'Lakh' in s:
        try:
            return float(s.replace('Lakh', '').strip()) * 100000
        except Exception:
            return np.nan
    try:
        return float(s)
    except Exception:
        return np.nan

# Add numeric price
df['price_numeric'] = df['car_prices_in_rupee'].apply(convert_price)

# Car age
df['car_age'] = 2025 - df['manufacture']

# Drop NaNs
plot_df = df[['car_age', 'price_numeric']].dropna().copy()

# Downsample for speed if needed
if len(plot_df) > 5000:
    plot_df = plot_df.sample(5000, random_state=42)

# Scatter: car_age vs price_numeric
plt.figure(figsize=(7,5))
plt.scatter(plot_df['car_age'], plot_df['price_numeric']/100000, alpha=0.4, s=12, color='#1f77b4', edgecolors='none')
plt.xlabel('Araba Yaşı (yıl)')
plt.ylabel('Fiyat (Lakh ₹)')
plt.title('Araba Yaşı vs Fiyat (Scatter)')
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig('age_vs_price.png', dpi=220)
print('✓ age_vs_price.png kaydedildi')
