# Araba Fiyat Tahmini — Sınav Özeti

Bu özet, sınavda el yazısıyla kolay aktarım için hazırlanmıştır.

---

## 1) Veri ve Örnek Satırlar

- Toplam kayıt: 5.512 → (temizleme sonrası) 4.921
- Nihai özellik sayısı: 33 (sayısal + one-hot kategorik)
- Hedef: `price_numeric` (₹)

Örnek 2 satır (özet):
```
car_name                         car_prices_in_rupee  kms_driven  fuel_type  transmission  manufacture  engine   Seats
Jeep Compass Longitude           10.03 Lakh           86,226 kms  Diesel     Manual        2017         1956 cc  5
Honda Jazz VX CVT                7.77 Lakh            26,696 kms  Petrol     Automatic     2018         1199 cc  5
```

---

## 2) Önemli Kod Parçacıkları

Veri yükleme + fiyat dönüştürme (Lakh/Crore/virgül):
```python
import pandas as pd

df = pd.read_csv('car_price.csv')

def convert_price(s):
    s = str(s).replace(',', '').strip()
    if 'Crore' in s:
        return float(s.replace('Crore', '').strip()) * 10000000
    if 'Lakh' in s:
        return float(s.replace('Lakh', '').strip()) * 100000
    return pd.to_numeric(s, errors='coerce')

df['price_numeric'] = df['car_prices_in_rupee'].apply(convert_price)
```

Korelasyon hesabı ve görselleştirme:
```python
import seaborn as sns, matplotlib.pyplot as plt

num_cols = ['engine_numeric', 'car_age', 'km_per_year', 'kms_numeric', 'price_numeric']
correlation_matrix = df[num_cols].corr()
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm')
plt.savefig('correlation_matrix.png', dpi=300)
```

---

## 3) Kolonlar Arası İlişki (Korelasyon)

- `engine_numeric` → fiyat: +0.68 (güçlü pozitif)
- `car_age` → fiyat: -0.52 (orta negatif)
- `kms_numeric` → fiyat: -0.39 (negatif)
- `km_per_year` → fiyat: -0.31 (negatif)

Görsel: `correlation_matrix.png`

![Korelasyon Matrisi](correlation_matrix.png)

---

## 4) Kullanılan Modeller

- Linear Regression, Ridge, Lasso
- Decision Tree, Random Forest
- Gradient Boosting (GridSearchCV ile ayarlandı)

---

## 5) En İyi Model ve Neden Uygun?

- En iyi model: Gradient Boosting Regressor (Tuned)
- En iyi hiperparametreler: `n_estimators=200`, `max_depth=5`, `learning_rate=0.1`, `min_samples_split=2`
- Neden uygun?
  - Doğrusal olmayan ilişkileri ve etkileşimleri (Marka × Yaş × KM) yakalar.
  - Sayısal + kategorik (one-hot) karışık veride güçlüdür.
  - Adım adım öğrenme + regularization → overfitting kontrolü.
  - Çapraz doğrulama puanı test skoruyla tutarlı → iyi genelleme.

Görsel: `model_comparison_improved.png`

![Model Karşılaştırma](model_comparison_improved.png)

---

## 6) Nihai Sonuçlar (Test Seti)

- R²: 0.7912  (≈ %79.12 açıklama gücü)
- RMSE: ₹279,173
- MAE: ₹183,392  (≈ %24.83 ortalama hata)
- 5-kat CV R²: 0.7691 ± 0.026

Yorum: Train ve Test R² birbirine yakın → aşırı uyum yok; hata analizi dengeli.

---

## 7) Sınav İçin Kısa Özet

- Problem: Araç özelliklerinden fiyat tahmini (regresyon)
- Veri: 4.921 kayıt, 33 özellik (brand, car_age, km_per_year vb.)
- Temel İlişkiler: Motor hacmi ↑ → fiyat ↑; Yaş/KM ↑ → fiyat ↓
- En İyi Model: Gradient Boosting (R²=0.7912)
- Neden: Non-linear, kategorik+sayısal uyumu, regularization, CV uyumu
- Sonuç: Ortalama mutlak hata ≈ ₹183K; üretime uygun seviye

---

## 8) (İsteğe Bağlı) Örnek Tahmin Kullanımı

```python
# Örnek (gerçek kodunuzdaki predict fonksiyonu üzerinden):
price = predict_car_price(
    kms_driven=50000, engine=1200, seats=5, car_age=5,
    fuel_type='Petrol', transmission='Manual', brand='Maruti', ownership=0
)
```
Örnek çıktılar: Maruti Swift ≈ ₹593,759; Hyundai Creta ≈ ₹1,570,017.
