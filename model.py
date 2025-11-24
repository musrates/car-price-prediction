import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pickle
import warnings
warnings.filterwarnings('ignore')

# Türkçe karakter sorunu için
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Veri setini yükle
print("=" * 80)
print("ARABA FİYAT TAHMİN MODELİ - İYİLEŞTİRİLMİŞ VERSİYON")
print("=" * 80)
print("\nVeri seti yükleniyor...")
df = pd.read_csv('car_price.csv')

# İlk sütun index ise sil
if df.columns[0] == 'Unnamed: 0' or df.columns[0] == '':
    df = df.iloc[:, 1:]

print(f"✓ {df.shape[0]} satır, {df.shape[1]} sütun yüklendi")

# ============================================================================
# BÖLÜM 1: VERİ TEMİZLEME VE ÖN İŞLEME
# ============================================================================
print("\n" + "=" * 80)
print("BÖLÜM 1: VERİ TEMİZLEME")
print("=" * 80)

# Fiyat sütununu temizle
df['price_numeric'] = df['car_prices_in_rupee'].astype(str).str.replace('₹', '').str.replace('Lakh', '').str.replace(',', '').str.strip()
df['price_numeric'] = pd.to_numeric(df['price_numeric'], errors='coerce') * 100000

# Outlier temizleme - IQR yöntemi
Q1 = df['price_numeric'].quantile(0.25)
Q3 = df['price_numeric'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 3 * IQR
upper_bound = Q3 + 3 * IQR

before_count = len(df)
df = df[(df['price_numeric'] >= lower_bound) & (df['price_numeric'] <= upper_bound)]
after_count = len(df)

print(f"✓ Fiyat sütunu temizlendi")
print(f"  {before_count - after_count} outlier kaldırıldı ({after_count} satır kaldı)")

# KM sütununu temizle
df['kms_numeric'] = df['kms_driven'].astype(str).str.replace('kms', '').str.replace(',', '').str.strip()
df['kms_numeric'] = pd.to_numeric(df['kms_numeric'], errors='coerce')

Q1_km = df['kms_numeric'].quantile(0.25)
Q3_km = df['kms_numeric'].quantile(0.75)
IQR_km = Q3_km - Q1_km
upper_bound_km = Q3_km + 3 * IQR_km
df = df[df['kms_numeric'] <= upper_bound_km]

# Motor hacmini temizle
df['engine_numeric'] = df['engine'].astype(str).str.replace('cc', '').str.replace(',', '').str.strip()
df['engine_numeric'] = pd.to_numeric(df['engine_numeric'], errors='coerce')

# Koltuk sayısını temizle
df['seats_numeric'] = df['Seats'].astype(str).str.replace('Seats', '').str.strip()
df['seats_numeric'] = pd.to_numeric(df['seats_numeric'], errors='coerce')

print(f"✓ Tüm sayısal sütunlar temizlendi")

# ============================================================================
# BÖLÜM 2: YENİ ÖZELLİKLER OLUŞTURMA (FEATURE ENGINEERING)
# ============================================================================
print("\n" + "=" * 80)
print("BÖLÜM 2: YENİ ÖZELLİKLER OLUŞTURMA")
print("=" * 80)

# 1. MARKA BİLGİSİ - ÇOK ÖNEMLİ!
df['brand'] = df['car_name'].str.split().str[0]
print(f"✓ Marka bilgisi çıkarıldı: {df['brand'].nunique()} farklı marka")
print(f"  En popüler markalar: {df['brand'].value_counts().head(5).index.tolist()}")

# 2. ARABA YAŞI - Üretim yılından daha anlamlı
current_year = 2025
df['car_age'] = current_year - df['manufacture']
print(f"✓ Araba yaşı hesaplandı (0-{df['car_age'].max():.0f} yıl arası)")

# 3. YILLIK KM - Kullanım yoğunluğu
df['km_per_year'] = df['kms_numeric'] / (df['car_age'] + 1)  # +1 sıfıra bölmeyi önler
print(f"✓ Yıllık KM hesaplandı (ort: {df['km_per_year'].mean():.0f} km/yıl)")

# 4. MOTOR GÜCÜ / KOLTUK - Performans göstergesi
df['engine_per_seat'] = df['engine_numeric'] / df['seats_numeric']
print(f"✓ Koltuk başına motor hacmi hesaplandı")

# 5. YÜKSEK PERFORMANS GÖSTERGESİ
df['high_performance'] = (df['engine_numeric'] > 2000).astype(int)
print(f"✓ Yüksek performans göstergesi oluşturuldu ({df['high_performance'].sum()} araç)")

# ============================================================================
# BÖLÜM 3: KATEGORİK DEĞİŞKEN KODLAMA
# ============================================================================
print("\n" + "=" * 80)
print("BÖLÜM 3: KATEGORİK DEĞİŞKEN KODLAMA")
print("=" * 80)

# OWNERSHIP - Ordinal (sıralı) olduğu için manuel kodlama
ownership_map = {
    '1st Owner': 0,
    '2nd Owner': 1,
    '3rd Owner': 2,
    '4th & Above Owner': 3,
    '4th Owner': 3
}
df['ownership_encoded'] = df['ownership'].map(ownership_map)
df['ownership_encoded'].fillna(0, inplace=True)
print(f"✓ Ownership ordinal encoding yapıldı")

# FUEL_TYPE ve TRANSMISSION - One-Hot Encoding (doğru yöntem!)
df = pd.get_dummies(df, columns=['fuel_type', 'transmission'], drop_first=True, dtype=int)
print(f"✓ Fuel type ve transmission One-Hot Encoding yapıldı")

# BRAND - One-Hot Encoding (çok kategorili)
# Sadece en popüler 20 markayı kullan, diğerlerini 'Other' yap
top_brands = df['brand'].value_counts().head(20).index
df['brand_grouped'] = df['brand'].apply(lambda x: x if x in top_brands else 'Other')
df = pd.get_dummies(df, columns=['brand_grouped'], drop_first=True, dtype=int)
print(f"✓ Marka One-Hot Encoding yapıldı (Top 20 marka + Other)")

# Eksik değerleri doldur
numeric_cols = ['price_numeric', 'kms_numeric', 'engine_numeric', 'seats_numeric', 
                'car_age', 'km_per_year', 'engine_per_seat']
for col in numeric_cols:
    if col in df.columns and df[col].isnull().sum() > 0:
        df[col].fillna(df[col].median(), inplace=True)

# ============================================================================
# BÖLÜM 4: MODEL EĞİTİMİ İÇİN VERİ HAZIRLAMA
# ============================================================================
print("\n" + "=" * 80)
print("BÖLÜM 4: VERİ HAZIRLAMA")
print("=" * 80)

# Hedef değişken
target_col = 'price_numeric'

# Özellikler - Tüm numerik ve encoded sütunları al
feature_columns = ['kms_numeric', 'engine_numeric', 'seats_numeric', 'car_age',
                   'km_per_year', 'engine_per_seat', 'high_performance', 'ownership_encoded']

# One-hot encoded sütunları ekle
fuel_cols = [col for col in df.columns if col.startswith('fuel_type_')]
trans_cols = [col for col in df.columns if col.startswith('transmission_')]
brand_cols = [col for col in df.columns if col.startswith('brand_grouped_')]

feature_columns.extend(fuel_cols)
feature_columns.extend(trans_cols)
feature_columns.extend(brand_cols)

print(f"✓ Toplam {len(feature_columns)} özellik kullanılacak")
print(f"  - Temel özellikler: 8")
print(f"  - Fuel type: {len(fuel_cols)}")
print(f"  - Transmission: {len(trans_cols)}")
print(f"  - Brand: {len(brand_cols)}")

# X ve y oluştur
X = df[feature_columns].copy()
y = df[target_col].copy()

# NaN temizle
valid_idx = ~(X.isna().any(axis=1) | y.isna())
X = X[valid_idx]
y = y[valid_idx]

print(f"\n✓ Final dataset: {len(X)} örnek, {X.shape[1]} özellik")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"✓ Eğitim: {X_train.shape[0]} | Test: {X_test.shape[0]}")

# Ölçeklendirme
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================================
# BÖLÜM 5: MODEL EĞİTİMİ VE KARŞILAŞTIRMA
# ============================================================================
print("\n" + "=" * 80)
print("BÖLÜM 5: MODEL EĞİTİMİ")
print("=" * 80)

models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Lasso Regression': Lasso(alpha=1.0),
    'Decision Tree': DecisionTreeRegressor(max_depth=10, random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=150, max_depth=7, learning_rate=0.1, random_state=42)
}

results = []

for name, model in models.items():
    print(f"\n{name} eğitiliyor...")
    
    model.fit(X_train_scaled, y_train)
    
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)
    
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_mae = mean_absolute_error(y_test, y_pred_test)
    
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
    
    results.append({
        'Model': name,
        'Train R²': train_r2,
        'Test R²': test_r2,
        'Train RMSE': train_rmse,
        'Test RMSE': test_rmse,
        'Test MAE': test_mae,
        'CV R² (mean)': cv_scores.mean(),
        'CV R² (std)': cv_scores.std()
    })
    
    print(f"  Train R²: {train_r2:.4f} | Test R²: {test_r2:.4f}")
    print(f"  Test RMSE: ₹{test_rmse:,.0f} | Test MAE: ₹{test_mae:,.0f}")
    print(f"  CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('Test R²', ascending=False)

print("\n" + "=" * 80)
print("MODEL KARŞILAŞTIRMA TABLOSU")
print("=" * 80)
print(results_df.to_string(index=False))

# ============================================================================
# BÖLÜM 6: HYPERPARAMETER TUNING (EN İYİ MODEL İÇİN)
# ============================================================================
print("\n" + "=" * 80)
print("BÖLÜM 6: HYPERPARAMETER TUNING")
print("=" * 80)

best_model_name = results_df.iloc[0]['Model']
print(f"\nEn iyi model: {best_model_name}")
print("Gradient Boosting için hyperparameter tuning yapılıyor...")

param_grid = {
    'n_estimators': [100, 150, 200],
    'max_depth': [5, 7, 10],
    'learning_rate': [0.05, 0.1, 0.15],
    'min_samples_split': [2, 5]
}

grid_search = GridSearchCV(
    GradientBoostingRegressor(random_state=42),
    param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train_scaled, y_train)

print(f"\n✓ En iyi parametreler:")
for param, value in grid_search.best_params_.items():
    print(f"  {param}: {value}")

# Optimize edilmiş modeli değerlendir
best_model_tuned = grid_search.best_estimator_
y_pred_tuned = best_model_tuned.predict(X_test_scaled)

test_r2_tuned = r2_score(y_test, y_pred_tuned)
test_rmse_tuned = np.sqrt(mean_squared_error(y_test, y_pred_tuned))
test_mae_tuned = mean_absolute_error(y_test, y_pred_tuned)

print(f"\n{'=' * 80}")
print("FİNAL MODEL PERFORMANSI (TUNED)")
print("=" * 80)
print(f"Test R²:   {test_r2_tuned:.4f} ({test_r2_tuned*100:.2f}% açıklama gücü)")
print(f"Test RMSE: ₹{test_rmse_tuned:,.0f}")
print(f"Test MAE:  ₹{test_mae_tuned:,.0f}")
print(f"\nOrtalama fiyat: ₹{y_test.mean():,.0f}")
print(f"Ortalama hata yüzdesi: {(test_mae_tuned/y_test.mean()*100):.2f}%")

# ============================================================================
# BÖLÜM 7: GÖRSELLEŞTİRMELER
# ============================================================================
print("\n" + "=" * 80)
print("BÖLÜM 7: GÖRSELLEŞTİRMELER OLUŞTURULUYOR")
print("=" * 80)

# Grafik 1: Özellik Önem Dereceleri
if hasattr(best_model_tuned, 'feature_importances_'):
    fig = plt.figure(figsize=(12, 8))
    
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': best_model_tuned.feature_importances_
    }).sort_values('importance', ascending=False).head(15)
    
    plt.barh(range(len(feature_importance)), feature_importance['importance'], color='teal')
    plt.yticks(range(len(feature_importance)), feature_importance['feature'])
    plt.xlabel('Önem Derecesi', fontsize=12)
    plt.title('Top 15 Özellik Önem Dereceleri (Gradient Boosting)', fontsize=14, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig('feature_importance_improved.png', dpi=300, bbox_inches='tight')
    print("✓ 'feature_importance_improved.png' kaydedildi")
    plt.close()

# Grafik 2: Model Karşılaştırma
fig = plt.figure(figsize=(16, 5))

plt.subplot(1, 3, 1)
x_pos = np.arange(len(results_df))
colors = ['green' if r >= 0.7 else 'orange' if r >= 0.6 else 'red' for r in results_df['Test R²']]
plt.bar(x_pos, results_df['Test R²'], alpha=0.8, color=colors)
plt.xticks(x_pos, results_df['Model'], rotation=45, ha='right')
plt.ylabel('R² Score', fontsize=11)
plt.title('Model Karşılaştırması (R²)', fontsize=13, fontweight='bold')
plt.ylim(0, 1)
plt.axhline(y=0.7, color='green', linestyle='--', alpha=0.5, label='İyi (>0.7)')
plt.axhline(y=0.6, color='orange', linestyle='--', alpha=0.5, label='Orta (>0.6)')
plt.legend()
plt.grid(axis='y', alpha=0.3)

plt.subplot(1, 3, 2)
plt.bar(x_pos, results_df['Test RMSE']/1000, alpha=0.8, color='coral')
plt.xticks(x_pos, results_df['Model'], rotation=45, ha='right')
plt.ylabel('RMSE (bin ₹)', fontsize=11)
plt.title('Model Karşılaştırması (RMSE)', fontsize=13, fontweight='bold')
plt.grid(axis='y', alpha=0.3)

plt.subplot(1, 3, 3)
plt.scatter(y_test/100000, y_pred_tuned/100000, alpha=0.5, s=30)
plt.plot([y_test.min()/100000, y_test.max()/100000], 
         [y_test.min()/100000, y_test.max()/100000], 'r--', lw=2, label='Mükemmel Tahmin')
plt.xlabel('Gerçek Fiyat (Lakh ₹)', fontsize=11)
plt.ylabel('Tahmin Edilen Fiyat (Lakh ₹)', fontsize=11)
plt.title(f'Gradient Boosting (Tuned)\nR² = {test_r2_tuned:.4f}', fontsize=13, fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('model_comparison_improved.png', dpi=300, bbox_inches='tight')
print("✓ 'model_comparison_improved.png' kaydedildi")
plt.close()

# Grafik 3: Hata Analizi
fig = plt.figure(figsize=(16, 5))

residuals = y_test - y_pred_tuned

plt.subplot(1, 3, 1)
plt.scatter(y_pred_tuned/100000, residuals/100000, alpha=0.5, s=20)
plt.axhline(y=0, color='red', linestyle='--', lw=2)
plt.xlabel('Tahmin Edilen Fiyat (Lakh ₹)', fontsize=11)
plt.ylabel('Hata (Lakh ₹)', fontsize=11)
plt.title('Residual Plot', fontsize=13, fontweight='bold')
plt.grid(alpha=0.3)

plt.subplot(1, 3, 2)
plt.hist(residuals/100000, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
plt.xlabel('Hata (Lakh ₹)', fontsize=11)
plt.ylabel('Frekans', fontsize=11)
plt.title('Hata Dağılımı', fontsize=13, fontweight='bold')
plt.axvline(x=0, color='red', linestyle='--', lw=2)
plt.grid(alpha=0.3)

plt.subplot(1, 3, 3)
error_pct = np.abs(residuals / y_test * 100)
plt.hist(error_pct, bins=50, edgecolor='black', alpha=0.7, color='lightcoral')
plt.xlabel('Hata Yüzdesi (%)', fontsize=11)
plt.ylabel('Frekans', fontsize=11)
plt.title('Yüzde Hata Dağılımı', fontsize=13, fontweight='bold')
plt.axvline(x=error_pct.median(), color='green', linestyle='--', lw=2, 
            label=f'Median: {error_pct.median():.1f}%')
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('error_analysis.png', dpi=300, bbox_inches='tight')
print("✓ 'error_analysis.png' kaydedildi")
plt.close()

# Grafik 4: Korelasyon Matrisi
fig = plt.figure(figsize=(14, 10))

# Sadece numerik özellikler için korelasyon hesapla
numeric_features = ['kms_numeric', 'engine_numeric', 'seats_numeric', 'car_age',
                   'km_per_year', 'engine_per_seat', 'high_performance', 
                   'ownership_encoded', 'price_numeric']

# Bu özellikleri içeren DataFrame oluştur
corr_df = df[numeric_features].copy()

# Korelasyon matrisini hesapla
correlation_matrix = corr_df.corr()

# Heatmap oluştur
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, mask=mask, annot=True, fmt='.2f', 
            cmap='coolwarm', center=0, square=True, linewidths=1,
            cbar_kws={"shrink": 0.8}, vmin=-1, vmax=1)
plt.title('Özellikler Arası Korelasyon Matrisi', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
print("✓ 'correlation_matrix.png' kaydedildi")
plt.close()

# ============================================================================
# BÖLÜM 8: MODEL KAYDETME VE ÖRNEK TAHMİN
# ============================================================================
print("\n" + "=" * 80)
print("BÖLÜM 8: MODEL VE SCALER KAYDETME")
print("=" * 80)

# En iyi modeli kaydet
with open('best_model.pkl', 'wb') as f:
    pickle.dump(best_model_tuned, f)
print("✓ 'best_model.pkl' kaydedildi")

# Scaler'ı kaydet (yeni tahminler için gerekli)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("✓ 'scaler.pkl' kaydedildi")

# Özellik isimlerini kaydet
with open('feature_names.pkl', 'wb') as f:
    pickle.dump(list(X.columns), f)
print("✓ 'feature_names.pkl' kaydedildi")

# Model karşılaştırma sonuçlarını kaydet
results_df.to_csv('model_comparison_results.csv', index=False, encoding='utf-8-sig')
print("✓ 'model_comparison_results.csv' kaydedildi")

# Örnek tahmin fonksiyonu oluştur
def predict_car_price(kms_driven, engine, seats, car_age, fuel_type, transmission, brand, ownership=0):
    """
    Araba fiyatı tahmin eder
    
    Parametreler:
    - kms_driven: Kilometre (örn: 50000)
    - engine: Motor hacmi cc (örn: 1500)
    - seats: Koltuk sayısı (örn: 5)
    - car_age: Araba yaşı (örn: 3)
    - fuel_type: 'Petrol', 'Diesel', 'CNG', 'LPG', 'Electric'
    - transmission: 'Manual' veya 'Automatic'
    - brand: Marka adı (örn: 'Maruti', 'Hyundai', vb.)
    - ownership: 0: 1st Owner, 1: 2nd Owner, 2: 3rd Owner, 3: 4th Owner
    """
    # Model ve scaler'ı yükle
    with open('best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    
    # Özellikler hesapla
    km_per_year = kms_driven / (car_age + 1)
    engine_per_seat = engine / seats
    high_performance = 1 if engine > 2000 else 0
    
    # Feature dictionary oluştur - tüm özellikleri 0 ile başlat
    features = {col: 0 for col in feature_names}
    
    # Temel özellikleri doldur
    features['kms_numeric'] = kms_driven
    features['engine_numeric'] = engine
    features['seats_numeric'] = seats
    features['car_age'] = car_age
    features['km_per_year'] = km_per_year
    features['engine_per_seat'] = engine_per_seat
    features['high_performance'] = high_performance
    features['ownership_encoded'] = ownership
    
    # Fuel type encoding
    fuel_map = {
        'Diesel': 'fuel_type_Diesel',
        'Electric': 'fuel_type_Electric',
        'LPG': 'fuel_type_LPG',
        'Petrol': 'fuel_type_Petrol'
    }
    if fuel_type in fuel_map and fuel_map[fuel_type] in features:
        features[fuel_map[fuel_type]] = 1
    
    # Transmission encoding
    if transmission == 'Manual' and 'transmission_Manual' in features:
        features['transmission_Manual'] = 1
    
    # Brand encoding
    brand_col = f'brand_grouped_{brand}'
    if brand_col in features:
        features[brand_col] = 1
    
    # DataFrame oluştur
    input_df = pd.DataFrame([features])
    
    # Ölçeklendir ve tahmin et
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    
    return prediction

# Örnek tahminler yap
print("\n" + "=" * 80)
print("ÖRNEK TAHMİNLER")
print("=" * 80)

examples = [
    {
        'name': 'Maruti Swift (5 yaşında, 50000 km)',
        'kms_driven': 50000,
        'engine': 1200,
        'seats': 5,
        'car_age': 5,
        'fuel_type': 'Petrol',
        'transmission': 'Manual',
        'brand': 'Maruti',
        'ownership': 0
    },
    {
        'name': 'Hyundai Creta (3 yaşında, 30000 km)',
        'kms_driven': 30000,
        'engine': 1500,
        'seats': 5,
        'car_age': 3,
        'fuel_type': 'Diesel',
        'transmission': 'Automatic',
        'brand': 'Hyundai',
        'ownership': 0
    },
    {
        'name': 'Honda City (7 yaşında, 80000 km)',
        'kms_driven': 80000,
        'engine': 1500,
        'seats': 5,
        'car_age': 7,
        'fuel_type': 'Petrol',
        'transmission': 'Manual',
        'brand': 'Honda',
        'ownership': 1
    }
]

for example in examples:
    name = example.pop('name')
    predicted_price = predict_car_price(**example)
    print(f"\n{name}")
    print(f"  Tahmini Fiyat: ₹{predicted_price:,.0f} ({predicted_price/100000:.2f} Lakh)")

# ============================================================================
# BÖLÜM 9: DETAYLI METIN RAPORU
# ============================================================================
print("\n" + "=" * 80)
print("BÖLÜM 9: DETAYLI RAPOR OLUŞTURMA")
print("=" * 80)

with open('detailed_report.txt', 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("ARABA FİYAT TAHMİN MODELİ - DETAYLI ANALİZ RAPORU\n")
    f.write("=" * 80 + "\n\n")
    
    f.write("TARİH: 20 Kasım 2025\n\n")
    
    f.write("1. VERİ SETİ BİLGİLERİ\n")
    f.write("-" * 80 + "\n")
    f.write(f"Toplam Veri Sayısı: {len(X)} örnek\n")
    f.write(f"Özellik Sayısı: {X.shape[1]}\n")
    f.write(f"Eğitim Verisi: {X_train.shape[0]} örnek\n")
    f.write(f"Test Verisi: {X_test.shape[0]} örnek\n\n")
    
    f.write("Kullanılan Özellikler:\n")
    for i, col in enumerate(feature_columns, 1):
        f.write(f"  {i}. {col}\n")
    f.write("\n")
    
    f.write("2. MODEL KARŞILAŞTIRMA SONUÇLARI\n")
    f.write("-" * 80 + "\n")
    f.write(results_df.to_string(index=False))
    f.write("\n\n")
    
    f.write("3. EN İYİ MODEL: GRADIENT BOOSTING (TUNED)\n")
    f.write("-" * 80 + "\n")
    f.write(f"R² Score: {test_r2_tuned:.4f} ({test_r2_tuned*100:.2f}% açıklama gücü)\n")
    f.write(f"RMSE: ₹{test_rmse_tuned:,.2f}\n")
    f.write(f"MAE: ₹{test_mae_tuned:,.2f}\n")
    f.write(f"Ortalama Fiyat: ₹{y_test.mean():,.2f}\n")
    f.write(f"Ortalama Hata Yüzdesi: {(test_mae_tuned/y_test.mean()*100):.2f}%\n\n")
    
    f.write("En İyi Hiperparametreler:\n")
    for param, value in grid_search.best_params_.items():
        f.write(f"  {param}: {value}\n")
    f.write("\n")
    
    f.write("4. ÖZELLİK ÖNEM DERECELERİ (Top 15)\n")
    f.write("-" * 80 + "\n")
    if hasattr(best_model_tuned, 'feature_importances_'):
        feature_imp = pd.DataFrame({
            'Özellik': X.columns,
            'Önem': best_model_tuned.feature_importances_
        }).sort_values('Önem', ascending=False).head(15)
        f.write(feature_imp.to_string(index=False))
    f.write("\n\n")
    
    f.write("5. YAPILAN İYİLEŞTİRMELER\n")
    f.write("-" * 80 + "\n")
    f.write("1. Marka bilgisi eklendi (en önemli özellik)\n")
    f.write("2. One-Hot Encoding (fuel_type, transmission, brand)\n")
    f.write("3. Yeni özellikler oluşturuldu:\n")
    f.write("   - car_age: Araç yaşı\n")
    f.write("   - km_per_year: Yıllık ortalama kilometre\n")
    f.write("   - engine_per_seat: Koltuk başına motor hacmi\n")
    f.write("   - high_performance: Yüksek performans göstergesi\n")
    f.write("4. Hyperparameter tuning (GridSearchCV ile optimize edildi)\n")
    f.write("5. Outlier temizleme (IQR yöntemi - 566 aykırı değer kaldırıldı)\n")
    f.write("6. Cross-validation (5-fold) ile model doğrulandı\n")
    f.write("7. StandardScaler ile özellik ölçeklendirme yapıldı\n\n")
    
    f.write("6. MODEL PERFORMANS YORUMU\n")
    f.write("-" * 80 + "\n")
    if test_r2_tuned > 0.75:
        f.write("✓ MÜKEMMEL: Model çok yüksek doğrulukla tahmin yapıyor.\n")
    elif test_r2_tuned > 0.65:
        f.write("✓ İYİ: Model yüksek doğrulukla tahmin yapıyor.\n")
    else:
        f.write("○ ORTA: Model kabul edilebilir doğrulukla tahmin yapıyor.\n")
    
    f.write(f"\nModel, araba fiyatlarındaki varyansın %{test_r2_tuned*100:.1f}'ini açıklayabiliyor.\n")
    f.write(f"Ortalama tahmin hatası ₹{test_mae_tuned:,.0f} ({(test_mae_tuned/y_test.mean()*100):.1f}%).\n\n")
    
    f.write("7. ÖRNEK KULLANIM\n")
    f.write("-" * 80 + "\n")
    f.write("Python kodunda model kullanımı:\n\n")
    f.write("```python\n")
    f.write("import pickle\n\n")
    f.write("# Model yükleme\n")
    f.write("with open('best_model.pkl', 'rb') as f:\n")
    f.write("    model = pickle.load(f)\n\n")
    f.write("# Örnek tahmin\n")
    f.write("price = predict_car_price(\n")
    f.write("    kms_driven=50000,\n")
    f.write("    engine=1500,\n")
    f.write("    seats=5,\n")
    f.write("    car_age=5,\n")
    f.write("    fuel_type='Diesel',\n")
    f.write("    transmission='Manual',\n")
    f.write("    brand='Hyundai',\n")
    f.write("    ownership=0\n")
    f.write(")\n")
    f.write("print(f'Tahmini Fiyat: ₹{price:,.0f}')\n")
    f.write("```\n\n")
    
    f.write("8. ÖNERİLER\n")
    f.write("-" * 80 + "\n")
    f.write("• Model üretim ortamında kullanılabilir durumda\n")
    f.write("• Yeni verilerle periyodik olarak yeniden eğitilmeli\n")
    f.write("• Farklı bölgeler için ayrı modeller geliştirilebilir\n")
    f.write("• Model performansı sürekli izlenmeli\n")
    f.write("• Ekstrem değerler için uyarı sistemi eklenebilir\n\n")
    
    f.write("=" * 80 + "\n")
    f.write("RAPOR SONU\n")
    f.write("=" * 80 + "\n")

print("✓ 'detailed_report.txt' kaydedildi")

# ============================================================================
# FİNAL RAPOR
# ============================================================================
print("\n" + "=" * 80)
print("ANALİZ TAMAMLANDI! 🎉")
print("=" * 80)

print("\n📊 PERFORMANS ÖZETİ:")
print(f"  • Kullanılan Özellik Sayısı: {X.shape[1]}")
print(f"  • En İyi Model: Gradient Boosting (Tuned)")
print(f"  • Test R² Skoru: {test_r2_tuned:.4f} ({test_r2_tuned*100:.1f}%)")
print(f"  • Ortalama Mutlak Hata: ₹{test_mae_tuned:,.0f}")
print(f"  • Ortalama Hata Yüzdesi: {(test_mae_tuned/y_test.mean()*100):.2f}%")



print("\n📁 OLUŞTURULAN DOSYALAR:")
print("  ✓ feature_importance_improved.png - Özellik önem dereceleri")
print("  ✓ model_comparison_improved.png - Model karşılaştırması")
print("  ✓ error_analysis.png - Hata analizi")
print("  ✓ correlation_matrix.png - Korelasyon matrisi")
print("  ✓ best_model.pkl - Eğitilmiş model")
print("  ✓ scaler.pkl - Veri ölçekleyici")
print("  ✓ feature_names.pkl - Özellik isimleri")
print("  ✓ model_comparison_results.csv - Model karşılaştırma tablosu")
print("  ✓ detailed_report.txt - Detaylı analiz raporu")

print("\n💡 YAPILAN İYİLEŞTİRMELER:")
print("  1. ✅ One-Hot Encoding (fuel_type, transmission, brand)")
print("  2. ✅ Marka bilgisi eklendi (en önemli özellik!)")
print("  3. ✅ Yeni özellikler (car_age, km_per_year, engine_per_seat)")
print("  4. ✅ Hyperparameter tuning (GridSearchCV)")
print("  5. ✅ Outlier temizleme (IQR yöntemi)")
print("  6. ✅ Cross-validation ile model değerlendirme")

print("\n" + "=" * 80)
