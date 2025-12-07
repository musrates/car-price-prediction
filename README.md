# Car Price Prediction

Gradient Boosting Regressor kullanarak ikinci el araç fiyat tahmini yapan makine öğrenmesi projesi. Kaggle'dan aldığım veri setiyle çalışırken feature engineering ve model optimizasyonu pratiği yapmak için geliştirdim.

## Kurulum

```bash
git clone https://github.com/musrates/car-price-prediction.git
cd car-price-prediction
pip install -r requirements.txt
```

Gereksinimler: pandas, numpy, scikit-learn, matplotlib, seaborn

## Veri Seti

5512 araç ilanından başladım. IQR yöntemiyle outlier temizleme yaptıktan sonra 4946 kayıt kaldı. Veri setindeki ham özellikler:

- car_name, car_prices_in_rupee (hedef değişken)
- kms_driven, fuel_type, transmission, ownership
- manufacture (üretim yılı), engine (motor hacmi), seats

### Feature Engineering

Mevcut veriden şu yeni değişkenleri türettim:

- **brand**: Araç adından marka bilgisi çıkardım (en kritik özellik oldu)
- **car_age**: 2025 - üretim yılı
- **km_per_year**: Yıllık ortalama kilometre kullanımı
- **engine_per_seat**: Motor hacmi / koltuk sayısı (performans göstergesi)
- **high_performance**: Motor hacmi 2000cc üstü mü (binary)

## Model Performansı

6 farklı algoritma denedim ve sonuçları karşılaştırdım:

| Model | Test R² | RMSE | MAE |
|-------|---------|------|-----|
| Gradient Boosting | 0.7912 | 279,173 | 183,392 |
| Random Forest | 0.7531 | 303,599 | 193,262 |
| Ridge Regression | 0.7092 | 329,578 | 227,426 |
| Lasso Regression | 0.7090 | 329,578 | 227,426 |
| Decision Tree | 0.6579 | 357,354 | 219,217 |

Gradient Boosting en iyi sonucu verdi. GridSearchCV ile hiperparametre optimizasyonu yaptım:
- n_estimators: 200
- max_depth: 5
- learning_rate: 0.1

![Model Comparison](model_comparison_improved.png)

Cross-validation (5-fold) sonuçlarına göre overfitting problemi yok. Model test setinde R² = 0.79 başarı yakaladı, ortalama hata ±183k Rupi.

## Feature Importance

![Feature Importance](feature_importance_improved.png)

En önemli 10 özellik:

1. brand_grouped_Maruti (0.234) - En kritik faktör
2. engine_numeric (0.187)
3. car_age (0.156)
4. brand_grouped_Hyundai (0.098)
5. km_per_year (0.067)
6. kms_numeric (0.054)
7. engine_per_seat (0.043)
8. ownership_encoded (0.038)
9. brand_grouped_Honda (0.031)
10. seats_numeric (0.025)

Marka bilgisi beklendiği gibi en önemli faktör oldu. Motor hacmi ve araç yaşı da güçlü etkiye sahip.

## Korelasyon Analizi

![Correlation Matrix](correlation_matrix.png)

Fiyat ile korelasyonlar:
- Motor hacmi: +0.68 (güçlü pozitif)
- Araç yaşı: -0.52 (orta negatif)
- Kilometre: -0.39 (negatif)

Beklendiği gibi büyük motorlu arabalar daha pahalı, eski ve yüksek kilometreli arabalar daha ucuz.

## Hata Analizi

![Error Analysis](error_analysis.png)

Model genel olarak başarılı ama bazı noktalarda sapma var:
- Çok düşük fiyatlı araçlarda (50k-100k) hafif overestimate yapıyor
- Premium segmentte (2M+) veri az olduğu için tahminler dalgalı

MAPE (Mean Absolute Percentage Error) %24.83. Ortalama olarak tahminler gerçek fiyatın %25'i kadar sapıyor.

## Kullanım

Model eğitimi:

```bash
python model.py
```

Bu komut tüm işlemleri yapıyor: veri temizleme, feature engineering, model eğitimi, görselleştirme ve model kaydetme.

Tahmin yapma:

```python
from model import predict_car_price

# Örnek: Maruti Swift
price = predict_car_price(
    kms_driven=50000,
    engine=1200,
    seats=5,
    car_age=5,
    fuel_type='Petrol',
    transmission='Manual',
    brand='Maruti',
    ownership=0  # 0: 1st Owner, 1: 2nd, 2: 3rd, 3: 4th+
)
print(f"Tahmini Fiyat: ₹{price:,.0f}")
```

## Dosya Yapısı

```
car-price-prediction/
├── model.py                              # Ana ML kodu
├── car_price.csv                         # Veri seti
├── requirements.txt
├── README.md
├── age_vs_price.png
├── correlation_matrix.png
├── error_analysis.png
├── feature_importance_improved.png
├── model_comparison_improved.png
└── make_extra_figures.py                 # Ek görsel scripti
```

## Öğrendiklerim

- Feature engineering'in ne kadar kritik olduğunu gördüm. Marka bilgisini çıkarmak model performansını ciddi şekilde artırdı
- IQR ile outlier temizleme gerçekten fark yarattı. Ham veriyle R² 0.65 civarındayken temizleme sonrası 0.79'a çıktı
- GridSearchCV sabır istiyor (3-4 saat sürdü) ama manuel tuning'den çok daha iyi sonuç veriyor
- Gradient Boosting, Random Forest'tan %5 daha iyi ama eğitim süresi 3-4 kat daha uzun
- Premium markalarda (BMW, Mercedes) veri az olduğu için tahminler tutarsız



