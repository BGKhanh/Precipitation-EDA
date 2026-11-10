# Ho Chi Minh City Daily Rainfall Forecasting System

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![Models](https://img.shields.io/badge/models-XGBoost%20%7C%20LightGBM%20%7C%20Hurdle%20%7C%20Neural-green.svg)]()
[![Forecast Horizon](https://img.shields.io/badge/forecast%20horizon-1--7%20days-orange.svg)]()
[![Tests Passing](https://img.shields.io/badge/tests-passing-brightgreen.svg)](tests/)

An end-to-end machine learning application for multi-day daily rainfall forecasting ($H = 1 \dots 7$ days) in Ho Chi Minh City, Vietnam. The system processes meteorological data from NASA POWER (2001–2026) and provides daily rainfall amount predictions (mm/day) along with rain/no-rain event detection.

---

## 🌟 Highlights & Key Capabilities

- 🌧️ **Multi-Day Forecasting (1–7 Days Ahead)**: Delivers daily precipitation forecasts for the upcoming week from any given forecast date.
- 🎯 **High Prediction Accuracy**:
  - **82.5% accuracy** on rainy vs. dry day classification (WMO $\ge 1.0\text{ mm}$ standard).
  - **95.2% Probability of Detection (POD / Recall)** on actual rainy days.
  - **3.73 mm** Mean Absolute Error (MAE) on 7-day continuous rainfall amounts with the champion model (`XGBoost Tweedie`).
- ⚡ **Fast & Lightweight Inference**: Less than **0.05 seconds** per 7-day forecast cycle, ideal for operational alerts and scheduled pipelines.
- 🛡️ **Zero-Inflation & Extreme Weather Handling**: Specifically tuned for tropical monsoon climates where 50–60% of days have zero rainfall, while rainy days can feature sudden torrential convective downpours (>50–100 mm/day).
- 🧩 **Modular & Extensible Architecture**: Reusable modules for automated data collection, leak-proof feature engineering, model training, and performance auditing.

---

## 📊 Performance at a Glance

The table below summarizes performance across different model architectures on the out-of-sample Test dataset (2020–2026, 2,184 forecast origins, 15,288 forecast instances):

| Model Family | Representative Model | Feature Set | MAE (mm) | RMSE (mm) | Rain Detection Accuracy | Inference Latency |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 🥇 **Tweedie Regressors (Best)** | `XGBoost Tweedie` | $F_3$ (Lags + Rolling + Season) | **3.73** | **8.10** | **82.5%** | **~0.04s** |
| 🥈 **Tweedie Regressors** | `LightGBM Tweedie` | $F_3$ (Lags + Rolling + Season) | 3.75 | 8.09 | 82.3% | ~0.05s |
| 🥉 **Direct ML Regressors** | `Random Forest Direct` | $F_3$ (Lags + Rolling + Season) | 3.94 | 8.11 | 80.1% | ~0.03s |
| **Two-Stage Hurdle (Mixture)** | `Logistic + Linear Regressor` | $F_3$ (Lags + Rolling + Season) | 3.96 | 8.01 | 81.2% | ~0.02s |
| **Deep Sequence Models** | `PatchTST (Transformer)` | Rainfall Target | 4.03 | 8.66 | 77.4% | ~1.76s |
| **Statistical Time Series** | `AutoARIMA` | Rainfall Target | 4.33 | 8.20 | 73.1% | ~49.8s |
| **Baseline Persistence** | `Naive (Yesterday's Rain)` | Rainfall Target | 5.14 | 10.76 | 68.2% | <0.01s |

### Forecast Error by Lead Day (Champion Model: XGBoost Tweedie)

Rainfall predictability naturally decreases with longer lead times:

| Lead Day | Day 1 ($t+1$) | Day 2 ($t+2$) | Day 3 ($t+3$) | Day 4 ($t+4$) | Day 5 ($t+5$) | Day 6 ($t+6$) | Day 7 ($t+7$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **MAE (mm)** | **3.22** | **3.52** | **3.67** | **3.81** | **3.91** | **3.99** | **4.02** |
| **RMSE (mm)** | 7.39 | 7.91 | 8.11 | 8.26 | 8.36 | 8.37 | 8.27 |

---

## 🛠️ Data & Feature Engineering

The system ingests daily surface meteorological data for Ho Chi Minh City (10.78°N, 106.70°E) from the NASA POWER API:

- **Raw Weather Variables (36 features, $F_0$)**: Temperature (min/max/dew point), relative & specific humidity, surface pressure, solar insolation, wind speed ($U/V$ vectors at 2m, 10m, 50m), and multi-layer soil wetness.
- **Historical Lags ($F_1$, 53 features)**: 1, 2, 3, 7, 14, and 30-day historical values of rainfall, humidity, and dew point.
- **Rolling Statistics ($F_2$, 65 features)**: 3, 7, 14, and 30-day backward moving averages and standard deviations.
- **Calendar & Monsoon Signals ($F_3$, 70 features)**: Solar harmonic components ($\sin/\cos$ day of year), wet season indicators (May–November), and monsoon transition phases.

All feature transformations strictly use past observations ($t \le 0$) to ensure zero lookahead in operational forecasts.

---

## 💡 Quick Start & Usage

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/your-username/DS108_project.git
cd DS108_project

# Set up Conda environment (Python 3.12)
conda create -n ts_forecasting python=3.12 -y
conda activate ts_forecasting

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Verification Tests

```bash
# Run unit and integration tests
pytest tests/ -v
```

### 3. Making a 7-Day Forecast (Python Example)

```python
import pandas as pd
from src.data.loader import load_clean_data
from src.featurengineering.pipeline import build_feature_set
from src.models.tree_models import XGBoostTweedieModel

# 1. Load historical weather data
df = load_clean_data("data/raw/hcmc_weather_daily.csv")

# 2. Build F3 features (lags, rolling averages, season indicators)
X, y = build_feature_set(df, level="F3")

# 3. Initialize and train model (or load saved weights)
model = XGBoostTweedieModel(n_estimators=300, max_depth=6)
model.fit(X, y)

# 4. Generate 7-day multi-horizon forecast from latest date
latest_features = X.iloc[[-1]]
seven_day_forecast = model.predict(latest_features)

print("Predicted rainfall for next 7 days (mm):")
for day, rainfall in enumerate(seven_day_forecast[0], start=1):
    print(f"  Day +{day}: {rainfall:.2f} mm")
```

---

## 🏗️ Project Structure

```
DS108_project/
├── data/
│   ├── raw/                           # Raw NASA POWER daily meteorological exports
│   └── processed/results/             # Evaluation outputs and forecast predictions
├── src/
│   ├── config/
│   │   └── constants.py               # Coordinates, split dates, feature lists
│   ├── data/
│   │   ├── crawler.py                 # NASA POWER API client for fetching daily data
│   │   ├── loader.py                  # Data loading and time series splitting
│   │   └── dataquality.py             # Data integrity, continuity, and quality checks
│   ├── featurengineering/
│   │   ├── pipeline.py                # F0 to F3 feature transformation pipelines
│   │   └── utils.py                   # Causal lag and rolling window operators
│   ├── models/
│   │   ├── tree_models.py             # XGBoost, LightGBM, Random Forest forecasters
│   │   ├── hurdle.py                  # Two-Stage Hurdle models (Classification + Amount)
│   │   ├── linear_models.py           # Ridge, Lasso, ElasticNet forecasters
│   │   └── nixtla/                    # Wrappers for StatsForecast and NeuralForecast
│   ├── evaluation/
│   │   ├── metrics.py                 # Core evaluation metrics (MAE, RMSE, MSE, MASE)
│   │   ├── ranking.py                 # Multi-model ranking and comparison utilities
│   │   └── diagnostics.py             # Rain event classification & confusion metrics
│   └── analysis/                      # Model diagnostics and comparison tools
│       ├── loaders.py                 # Multi-partition forecast results loader
│       ├── rankings.py                # Leaderboard builders
│       ├── horizon.py                 # Multi-step error degradation analysis
│       ├── extremes.py                # Heavy rainfall (>50mm, >100mm) analysis
│       └── classification.py          # Rain/no-rain detection metrics
├── notebooks/
│   ├── modeling/                      # Step-by-step model training and evaluation
│   │   ├── 03.0_phase1_statistical_baselines.ipynb
│   │   ├── 03.1_phase2_ml_direct_forecasting.ipynb
│   │   ├── 03.2_phase3_ml_tweedie_forecasting.ipynb
│   │   ├── 03.3_phase4_neural_forecasting.ipynb
│   │   ├── 03.4_phase5_hurdle_forecasting.ipynb
│   │   └── 03.5_stage3_test_evaluation.ipynb
│   └── result_analysis/               # Comparative performance dashboards
│       ├── 01_benchmark_leaderboard.ipynb
│       ├── 02_horizon_analysis.ipynb
│       ├── 03_extreme_events.ipynb
│       └── 05_computational_efficiency.ipynb
├── reports/
│   └── test_benchmark/                # Full model leaderboards & summary reports
│       ├── unified_72_scenarios_test_leaderboard.csv
│       ├── unified_72_test_horizon_breakdown.csv
│       └── executive_test_summary.md
└── tests/                             # Automated test suite
    ├── analysis/                      # Tests for analysis and metrics modules
    ├── test_point_forecast_metrics.py # Point forecast metric tests
    └── test_hurdle_baseline.py        # Hurdle model pipeline tests
```

---

## 🏃 Reproducing Model Training & Reports

To run the complete training and evaluation pipeline from scratch:

```bash
# 1. Run model family pipelines
python scripts/run_phase1_statistical.py   # Statistical baselines
python scripts/run_phase2_ml_direct.py      # Direct ML regressors
python scripts/run_phase3_ml_tweedie.py    # Tweedie gradient boosters
python scripts/run_phase4_neural.py        # Deep learning models (GPU)
python scripts/run_phase5_hurdle.py        # Two-Stage Hurdle forecasters

# 2. Run multi-horizon test evaluation
python scripts/run_stage3_test_evaluation.py

# 3. Generate consolidated leaderboard and summary tables
python scripts/build_test_benchmark_leaderboard.py
```

All summary tables and performance metrics will be exported to `reports/test_benchmark/`.

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
