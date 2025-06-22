# DS108 Weather Prediction Project - Rainfall Forecasting for Ho Chi Minh City

A comprehensive machine learning project for predicting daily rainfall in Ho Chi Minh City using NASA POWER meteorological data.

## 🎯 Project Overview

This project implements multiple machine learning approaches to forecast daily rainfall amounts for Ho Chi Minh City, Vietnam. The system uses 25+ years of historical weather data (2000-2025) from NASA POWER API and employs various modeling techniques including tree-based models, linear models, neural networks, and time series analysis.

### Key Features
- **Multi-model approach**: Tree-based, Linear, Neural Network, and Time Series models
- **Two-stage prediction**: Rain/no-rain classification + rainfall amount regression
- **Advanced feature engineering**: MSTL decomposition, lag features, rolling statistics
- **Comprehensive evaluation**: Multiple metrics and visualization tools
- **Production-ready pipeline**: Modular, scalable, and well-documented codebase

## 📊 Dataset

- **Source**: NASA POWER API
- **Location**: Ho Chi Minh City (10.78°N, 106.7°E)
- **Time Period**: January 1, 2000 - April 30, 2025
- **Samples**: 9,252+ daily observations
- **Features**: 23 meteorological variables
- **Target**: Daily precipitation (PRECTOTCORR) in mm/day

### Meteorological Variables
- Temperature (min, max, average)
- Humidity (relative, specific)
- Wind speed and direction
- Solar radiation
- Pressure
- Soil moisture
- And more...

## 🏗️ Project Structure:
'''
DS108_project/
├── src/ # Source code modules
│ ├── config/ # Configuration and constants
│ │ └── constants.py
│ ├── data/ # Data handling modules
│ │ ├── crawler.py # NASA API data collection
│ │ ├── loader.py # Data loading utilities
│ │ └── dataquality.py # Data quality checks
│ ├── analysis/ # Exploratory data analysis
│ │ ├── CorrelationAnalysis.py
│ │ ├── DistributionAnalysis.py
│ │ ├── ExtremeEventAnalysis.py
│ │ ├── Stationarity.py
│ │ └── TemporalAnalysis.py
│ ├── featurengineering/ # Feature engineering tools
│ │ └── utils.py # Feature creation utilities
│ ├── models/ # Machine learning models
│ │ ├── base.py # Base model classes
│ │ ├── tree_models.py # Random Forest, XGBoost, LightGBM
│ │ ├── linear_models.py # Ridge, Lasso, Linear Regression
│ │ ├── neural_models.py # RNN, LSTM, GRU, BiLSTM
│ │ └── time_series.py # ARIMA, SARIMA models
│ └── training/ # Training infrastructure
│ ├── trainer.py # Training manager
│ ├── optimizer.py # Hyperparameter optimization
│ └── validator.py # Cross-validation utilities
├── nasa_power_hcmc_data/ # Raw and processed data
├── DS.ipynb # Main analysis notebook
├── requirements.txt # Python dependencies
└── README.md # This file
'''

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- Required packages listed in `requirements.txt`

### Installation
```bash
# Clone the repository
git clone https://github.com/your-username/DS108_project.git
cd DS108_project

# Install dependencies
pip install -r requirements.txt
```

### Quick Start
```python
# Load the main notebook
jupyter notebook DS.ipynb

# Or use individual modules
from src.models import RandomForestRainfallModel
from src.data.loader import load_weather_data

# Load data
df = load_weather_data('nasa_power_hcmc_data/hcmc_weather_data_20000101_20250430.csv')

# Create and train model
model = RandomForestRainfallModel(use_two_stage=True)
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)
```

## 🤖 Available Models

### Tree-Based Models
- **Random Forest**: Ensemble of decision trees with bagging
- **XGBoost**: Gradient boosting with advanced regularization
- **LightGBM**: Fast gradient boosting with leaf-wise growth

### Linear Models
- **Ridge Regression**: L2 regularized linear regression
- **Lasso Regression**: L1 regularized with feature selection
- **Linear Regression**: Standard ordinary least squares

### Neural Networks (pending)
- **RNN**: Basic recurrent neural network
- **LSTM**: Long Short-Term Memory networks
- **GRU**: Gated Recurrent Units
- **BiLSTM**: Bidirectional LSTM 

### Time Series Models
- **ARIMA**: AutoRegressive Integrated Moving Average
- **SARIMA**: Seasonal ARIMA with seasonal components
- **ARIMAX**: AutoRegressive Integrated Moving Average (pending)
- **SARIMAX**: Seasonal ARIMAX with seasonal components (pending)

### Two-Stage Modeling
Each model (except time series) supports configurable two-stage approach:
1. **Classification Stage**: Predict rain/no-rain (binary classification)
2. **Regression Stage**: Predict rainfall amount for rainy days

### Advanced Feature Engineering
- **MSTL Decomposition**: Trend, seasonal, and residual components
- **Lag Features**: Historical values (1-7 days)
- **Rolling Statistics**: Moving averages, standard deviations
- **Temporal Features**: Day of year, month, season indicators

### Comprehensive Evaluation
- **Classification Metrics**: ROC-AUC, Precision, Recall
- **Regression Metrics**: MAE, RMSE, R²
- **Cross-Validation**: Time series aware validation
- **Visualization**: Performance plots and forecast charts

## 📈 Performance Metrics

The project evaluates models using multiple metrics:

- **Mean Absolute Error (MAE)**: Average absolute prediction error
- **Root Mean Square Error (RMSE)**: Penalizes larger errors more heavily
- **R² Score**: Coefficient of determination (explained variance)
- **ROC-AUC**: Area under ROC curve for classification

## 🛠️ Usage Examples

### Training Multiple Models
```python
from src.training import RainfallTrainer

# Initialize trainer
trainer = RainfallTrainer(classification_threshold=0.1)

# Define model configurations
model_configs = {
    'RandomForest_TwoStage': {
        'class': RandomForestRainfallModel,
        'use_two_stage': True,
        'param_spaces': {...}
    }
}

# Train all models
results = trainer.full_training_pipeline(
    df=data,
    target_column='Lượng mưa',
    model_configs=model_configs
)
```

### Feature Engineering
```python
from src.featurengineering.utils import create_lag_features, create_rolling_features

# Create lag features
df_with_lags = create_lag_features(df, ['Temperature', 'Humidity'], lag_periods=[1, 3, 7])

# Create rolling features
df_with_rolling = create_rolling_features(df, ['Precipitation'], windows=[7, 30])
```

### Time Series Analysis
```python
from src.models.time_series import SARIMAModel, StationarityTester

# Test stationarity
stationarity = StationarityTester.test_stationarity(rainfall_series)

# Train SARIMA model
model = SARIMAModel(order=(3,1,3), seasonal_order=(1,1,1,365))
model.fit(rainfall_series)

# Generate forecasts
forecasts = model.forecast(steps=30)
```
