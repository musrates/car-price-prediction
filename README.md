# Araba Fiyat Tahmin Modeli

Gradient Boosting Regressor kullanarak ikinci el araç fiyat tahmini yapan Machine Learning projesi.

## 📋 Proje Özeti

- **Amaç:** Araç özelliklerine göre (marka, yaş, km, motor vb.) ikinci el fiyat tahmini
- **Model:** Gradient Boosting Regressor (GridSearchCV ile optimize edilmiş)
- **Başarı:** R² = 0.7912 (%79.12 açıklama gücü)
- **Ortalama Hata:** ₹183,392 (≈%24.83)

## 📁 Proje Yapısı

```
car-price-prediction/
├── model.py                              # Ana ML kodu
├── car_price.csv                         # Veri seti (5,512 kayıt)
├── .gitignore                            # Git ignore kuralları
├── README.md                             # Proje dökümantasyonu
│
├── Grafikler/
│   ├── correlation_matrix.png           # Korelasyon ısı haritası
│   ├── age_vs_price.png                 # Yaş-fiyat scatter grafiği
│   ├── model_comparison_improved.png    # Model karşılaştırma grafikleri
│   ├── feature_importance_improved.png  # Özellik önem dereceleri
│   └── error_analysis.png               # Hata analizi grafikleri
│
└── Scriptler/
    └── make_extra_figures.py            # Ek görsel oluşturma scripti
```

## 🚀 Kurulum

```bash
# Repoyu klonla
git clone https://github.com/kullaniciadi/car-price-prediction.git
cd car-price-prediction

# Virtual environment oluştur
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Gereksinimleri yükle
pip install pandas numpy matplotlib seaborn scikit-learn
```

## 💻 Kullanım

```bash
# Ana modeli çalıştır (eğitim + tahmin + görselleştirme)
python model.py
```

**Çıktılar:**
- 5 adet PNG görsel dosyası
- Eğitilmiş modeller (best_model.pkl, scaler.pkl, feature_names.pkl)
- CSV ve TXT raporlar

## 📊 Özellikler

**Ham Özellikler (9):**
- car_name, car_prices_in_rupee, kms_driven, fuel_type
- transmission, ownership, manufacture, engine, Seats

**Türetilmiş Özellikler (5):**
- `brand`: Araç adından çıkarılan marka (en önemli özellik!)
- `car_age`: Araç yaşı (2025 - üretim yılı)
- `km_per_year`: Yıllık kilometre kullanımı
- `engine_per_seat`: Motor hacmi/koltuk oranı
- `high_performance`: Yüksek performans bayrağı (>2000cc)

**Nihai:** 33 özellik (one-hot encoding sonrası)

## 🎯 Model Sonuçları

| Model | Test R² | RMSE (₹) | MAE (₹) |
|-------|---------|----------|---------|
| **Gradient Boosting** | **0.7912** | **279,173** | **183,392** |
| Random Forest | 0.7531 | 303,599 | 193,262 |
| Ridge/Lasso | ≈0.709 | 329,578 | 227,426 |
| Decision Tree | 0.6579 | 357,354 | 219,217 |

**Hiperparametreler (GridSearchCV):**
- n_estimators: 200
- max_depth: 5
- learning_rate: 0.1
- min_samples_split: 2

## 📈 Korelasyon Bulguları

- `engine_numeric` ↗ fiyat: **+0.68** (güçlü pozitif)
- `car_age` ↗ fiyat: **−0.52** (orta negatif)
- `kms_numeric` ↗ fiyat: **−0.39** (negatif)
- `km_per_year` ↗ fiyat: **−0.31** (negatif)

## 🔧 Teknolojiler

- **Python 3.13**
- **pandas** - Veri manipülasyonu
- **scikit-learn** - ML modelleri ve ön işleme
- **matplotlib & seaborn** - Görselleştirme
- **numpy** - Sayısal hesaplamalar

## 👤 Yazar

[Adınız]

## 📄 Lisans

Bu proje eğitim amaçlıdır.
