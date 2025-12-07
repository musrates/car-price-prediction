# Araba Fiyat Tahmin Modeli

Gradient Boosting kullanarak ikinci el araç fiyatlarını tahmin eden bir makine öğrenmesi projesi. Kaggle'dan aldığım veri setiyle çalışırken öğrendiklerimi uygulamak için geliştirdim.

## Sonuçlar

Model test setinde **R² = 0.79** başarı yakaladı. Ortalama olarak tahmini fiyatlar gerçek fiyatlardan ±183k Rupi sapıyor.

En önemli faktörler:
- Marka (%23)
- Motor hacmi (%19)  
- Araç yaşı (%16)

## Kurulum

```bash
git clone https://github.com/musrates/car-price-prediction.git
cd car-price-prediction
pip install -r requirements.txt
python model.py
```

## Veri Seti

5512 araç ilanından başladım, outlier temizleme sonrası 4946 kayıt kaldı. Temel özellikler:
- Kilometre
- Motor hacmi
- Yakıt tipi (Petrol/Diesel/CNG/LPG/Electric)
- Vites (Manuel/Otomatik)
- Kaçıncı el
- Koltuk sayısı

Bunlardan `car_age` ve `brand` gibi yeni değişkenler türettim. Marka bilgisi en kritik özellik oldu.

## Model

6 farklı algoritma denedim:

| Model | Test R² | MAE (₹) |
|-------|---------|---------|
| **Gradient Boosting** | **0.7912** | **183,392** |
| Random Forest | 0.7531 | 193,262 |
| Ridge/Lasso | ~0.709 | 227,426 |
| Decision Tree | 0.6579 | 219,217 |

Gradient Boosting'i GridSearchCV ile optimize ettim:
- n_estimators: 200
- max_depth: 5  
- learning_rate: 0.1

Cross-validation (5-fold) ile doğruladım, overfitting sorunu yok gibi görünüyor.

## Kullanım

```python
from model import predict_car_price

# Örnek tahmin
price = predict_car_price(
    kms_driven=50000,
    engine=1200,
    seats=5,
    car_age=5,
    fuel_type='Petrol',
    transmission='Manual',
    brand='Maruti',
    ownership=0
)
print(f"Tahmini: ₹{price:,.0f}")
```

## Dosya Yapısı

```
├── model.py              # Ana kod
├── car_price.csv         # Veri seti
├── requirements.txt
├── Grafikler/
│   ├── correlation_matrix.png
│   ├── feature_importance_improved.png
│   └── ...
└── Scriptler/
    └── make_extra_figures.py
```

## Ne Öğrendim

- Feature engineering'in ne kadar kritik olduğunu gördüm (özellikle brand çıkarma)
- IQR ile outlier temizleme gerçekten fark yarattı
- GridSearchCV sabır istiyor ama değiyor
- Gradient Boosting, Random Forest'tan biraz daha iyi ama çok daha yavaş

## TODO

- [ ] Flask API yapıp deployment denemeliyim
- [ ] Veri setine renk, bölge gibi özellikler eklenebilir
- [ ] Ensemble model (voting) deneyebilirim
- [ ] Streamlit ile basit bir UI yapılabilir

## Sorunlar

Bazı premium markalarda (BMW, Mercedes) veri az olduğu için tahminler pek iyi değil. Daha fazla veri lazım veya bu markalar için ayrı model eğitilebilir.

---

Sorular için issue açabilirsiniz. Projeye katkıda bulunmak isterseniz PR gönderin.

**Not:** Bu bir öğrenme projesi, production'da kullanmadan önce daha fazla test gerekir.
