# Car Price Prediction Model

Machine Learning project for predicting used car prices using Gradient Boosting Regressor.

## 📋 Project Overview

- **Goal:** Predict used car prices based on features (brand, age, mileage, engine, etc.)
- **Model:** Gradient Boosting Regressor (Tuned with GridSearchCV)
- **Performance:** R² = 0.7912 (79.12% variance explained)
- **Average Error:** ₹183,392 (≈24.83%)

## 📁 Project Structure

```
car-price-prediction/
├── model.py                              # Main ML pipeline
├── car_price.csv                         # Dataset (5,512 records)
├── .gitignore                            # Git ignore rules
├── README.md                             # Project documentation
│
├── Visualizations/
│   ├── correlation_matrix.png           # Correlation heatmap
│   ├── age_vs_price.png                 # Age vs price scatter plot
│   ├── model_comparison_improved.png    # Model comparison charts
│   ├── feature_importance_improved.png  # Feature importance
│   └── error_analysis.png               # Error analysis plots
│
└── Scripts/
    └── make_extra_figures.py            # Generate additional visualizations
```

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/car-price-prediction.git
cd car-price-prediction

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Install requirements
pip install pandas numpy matplotlib seaborn scikit-learn
```

## 💻 Usage

```bash
# Run the main model (training + prediction + visualizations)
python model.py
```

**Outputs:**
- 5 PNG visualization files
- Trained models (best_model.pkl, scaler.pkl, feature_names.pkl)
- CSV and TXT reports

## 📊 Features

**Raw Features (9):**
- car_name, car_prices_in_rupee, kms_driven, fuel_type
- transmission, ownership, manufacture, engine, Seats

**Engineered Features (5):**
- `brand`: Brand extracted from car name (most important feature!)
- `car_age`: Vehicle age (2025 - manufacture year)
- `km_per_year`: Annual mileage usage
- `engine_per_seat`: Engine-to-seat ratio
- `high_performance`: High performance flag (>2000cc)

**Final:** 33 features (after one-hot encoding)

## 🎯 Model Results

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

## 📈 Correlation Insights

- `engine_numeric` ↗ price: **+0.68** (strong positive)
- `car_age` ↗ price: **−0.52** (moderate negative)
- `kms_numeric` ↗ price: **−0.39** (negative)
- `km_per_year` ↗ price: **−0.31** (negative)

## 🔧 Technologies

- **Python 3.13**
- **pandas** - Data manipulation
- **scikit-learn** - ML models & preprocessing
- **matplotlib & seaborn** - Visualizations
- **numpy** - Numerical computations

## 👤 Author

Your Name

## 📄 License

This project is for educational purposes.
