# Car Price Prediction Model

Machine Learning regresyon projesi - ikinci el araç fiyat tahmini.

## 📋 Proje Özeti

- **Amaç:** Araç özelliklerinden (marka, yaş, km, motor vb.) ikinci el satış fiyatını tahmin etmek
- **Model:** Gradient Boosting Regressor (Tuned)
- **Başarı:** R² = 0.7912 (%79.12 açıklama gücü)
- **Ortalama Hata:** ₹183,392 (≈%24.83)

## 📁 Dosya Yapısı

```
mlbir/
├── model.py                              # Ana ML pipeline
├── car_price.csv                         # Veri seti (5,512 kayıt)
├── .gitignore                            # Git ignore kuralları
│
├── Grafikler/
│   ├── correlation_matrix.png           # Korelasyon ısı haritası
│   ├── age_vs_price.png                 # Yaş-fiyat scatter
│   ├── model_comparison_improved.png    # Model karşılaştırma
│   ├── feature_importance_improved.png  # Özellik önemleri
│   └── error_analysis.png               # Hata analizi
│
├── Raporlar/
│   ├── README_SINAV.md                  # Sınav özeti (kısa)
│   ├── PROJE_RAPORU.md                  # Detaylı markdown rapor
│   └── PROJE_RAPORU.docx                # Word raporu
│
└── Yardımcı Scriptler/
    ├── export_to_word.py                # Word raporu üreten script
    └── make_extra_figures.py            # Ek görsel üreten script
```

## 🚀 Kurulum

```bash
# Repository'yi klonla
git clone https://github.com/kullaniciadi/car-price-prediction.git
cd car-price-prediction

# Virtual environment oluştur
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Gereksinimleri yükle
pip install pandas numpy matplotlib seaborn scikit-learn python-docx
```

## 💻 Kullanım

```bash
# Ana modeli çalıştır (eğitim + tahmin + grafikler)
python model.py
```

**Çıktılar:**
- 5 adet PNG grafik
- Eğitilmiş model (best_model.pkl, scaler.pkl, feature_names.pkl)
- CSV ve TXT raporlar

## 📊 Özellikler

**Ham Özellikler (9):**
- car_name, car_prices_in_rupee, kms_driven, fuel_type
- transmission, ownership, manufacture, engine, Seats

**Türetilmiş Özellikler (5):**
- `brand`: Marka (en önemli özellik!)
- `car_age`: Araba yaşı (2025 - üretim yılı)
- `km_per_year`: Yıllık km kullanımı
- `engine_per_seat`: Motor/koltuk oranı
- `high_performance`: Yüksek performans bayrağı (>2000cc)

**Nihai:** 33 özellik (one-hot encoding sonrası)

## 🎯 Model Sonuçları

| Model | Test R² | RMSE (₹) | MAE (₹) |
|-------|---------|----------|---------|
| **Gradient Boosting** | **0.7912** | **279,173** | **183,392** |
| Random Forest | 0.7531 | 303,599 | 193,262 |
| Ridge/Lasso | ≈0.709 | 329,578 | 227,426 |
| Decision Tree | 0.6579 | 357,354 | 219,217 |

**Hyperparameters (GridSearchCV):**
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
- **scikit-learn** - ML modelleri
- **matplotlib & seaborn** - Görselleştirme
- **python-docx** - Word raporu

## 📝 Detaylı Raporlar

- [`README_SINAV.md`](README_SINAV.md) - Sınavda yazmalık kısa özet
- [`PROJE_RAPORU.md`](PROJE_RAPORU.md) - Kapsamlı proje raporu
- `PROJE_RAPORU.docx` - Word formatında rapor

## 👤 Yazar

[Adınız]

## 📄 Lisans

Bu proje eğitim amaçlıdır.
