"""
Models package for DS108 Weather Prediction Project.

This package contains 4 model families for rainfall prediction:
1. Tree-based models: Random Forest, XGBoost, LightGBM (with two-stage option)
2. Neural Network models: RNN, GRU, LSTM, BiLSTM (with two-stage option) 
3. Linear models: Logistic Regression + Ridge/Lasso (with two-stage option)
4. Time series models: ARIMA, SARIMA, ARIMAX, SARIMAX (single-stage only)

Each model supports configurable two-stage approach:
- Stage 1: Classification (rain/no-rain) with ROC-AUC metric
- Stage 2: Regression (rainfall amount) with MAE, RMSE, R2 metrics
- Option to disable two-stage for classical single-stage approach comparison
"""

from .base import (
    BaseRainfallModel,
    BaseTimeSeriesModel,
    calculate_metrics,
    evaluate_rainfall_model
)

from .tree_models import (
    RandomForestRainfallModel,
    LightGBMRainfallModel, 
    XGBoostRainfallModel
)

# Import neural models with fallback
try:
    from .neural_models import (
        RNNRainfallModel,
        LSTMRainfallModel,
        GRURainfallModel,
        BiLSTMRainfallModel
    )
    NEURAL_MODELS_AVAILABLE = True
except ImportError:
    NEURAL_MODELS_AVAILABLE = False
    print("⚠️ Neural network models not available. Install PyTorch to use them.")

# Import linear models with fallback
try:
    from .linear_models import (
        LinearRainfallModel
    )
    LINEAR_MODELS_AVAILABLE = True
except ImportError:
    LINEAR_MODELS_AVAILABLE = False
    print("⚠️ Linear models not available.")

from .time_series import (
    ARIMAModel,
    SARIMAModel,
    ARIMAXModel,
    SARIMAXModel,
    StationarityTester,
    evaluate_time_series_model,
    prepare_time_series_data
)

# Base exports (always available)
__all__ = [
    # Base classes
    'BaseRainfallModel',
    'BaseTimeSeriesModel',
    
    # Tree-based models (always available)
    'RandomForestRainfallModel',
    'LightGBMRainfallModel',
    'XGBoostRainfallModel',
    
    # Time series models (always available)
    'ARIMAModel',
    'SARIMAModel', 
    'ARIMAXModel',
    'SARIMAXModel',
    'StationarityTester',
    
    # Utilities
    'calculate_metrics',
    'evaluate_rainfall_model',
    'evaluate_time_series_model',
    'prepare_time_series_data'
]

# Add neural models if available
if NEURAL_MODELS_AVAILABLE:
    __all__.extend([
        'RNNRainfallModel',
        'LSTMRainfallModel',
        'GRURainfallModel', 
        'BiLSTMRainfallModel'
    ])

# Add linear models if available
if LINEAR_MODELS_AVAILABLE:
    __all__.extend([
        'LinearRainfallModel'
    ])


def get_available_models() -> Dict[str, List[str]]:
    """
    Get list of available model families and their models.
    
    Returns:
        Dictionary with model families as keys and model lists as values
    """
    available = {
        'tree_based': [
            'RandomForestRainfallModel',
            'LightGBMRainfallModel',
            'XGBoostRainfallModel'
        ],
        'time_series': [
            'ARIMAModel',
            'SARIMAModel',
            'ARIMAXModel', 
            'SARIMAXModel'
        ]
    }
    
    if NEURAL_MODELS_AVAILABLE:
        available['neural_network'] = [
            'RNNRainfallModel',
            'LSTMRainfallModel',
            'GRURainfallModel',
            'BiLSTMRainfallModel'
        ]
    
    if LINEAR_MODELS_AVAILABLE:
        available['linear'] = [
            'LinearRainfallModel'
        ]
    
    return available


def create_model(model_type: str, model_name: str, **kwargs) -> BaseRainfallModel:
    """
    Factory function to create model instances.
    
    Args:
        model_type: Model family ('tree_based', 'neural_network', 'linear', 'time_series')
        model_name: Specific model name
        **kwargs: Model parameters
        
    Returns:
        Model instance
        
    Example:
        # Create two-stage Random Forest
        rf_model = create_model('tree_based', 'RandomForestRainfallModel', 
                               use_two_stage=True, n_estimators=100)
        
        # Create single-stage LSTM
        lstm_model = create_model('neural_network', 'LSTMRainfallModel',
                                 use_two_stage=False, hidden_size=64)
    """
    available = get_available_models()
    
    if model_type not in available:
        raise ValueError(f"Model type '{model_type}' not available. Available: {list(available.keys())}")
    
    if model_name not in available[model_type]:
        raise ValueError(f"Model '{model_name}' not available in '{model_type}'. Available: {available[model_type]}")
    
    # Get model class
    model_class = globals()[model_name]
    
    # Create instance
    return model_class(**kwargs)


# Model comparison utilities
def compare_approaches(model: BaseRainfallModel, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
    """
    Compare two-stage vs single-stage approach for the same model.
    
    Args:
        model: Model class (not instance)
        X_test: Test features
        y_test: Test targets
        
    Returns:
        Comparison results
        
    Note:
        This function requires both models to be pre-fitted
    """
    if not hasattr(model, 'use_two_stage'):
        raise ValueError("Model must support two-stage configuration")
    
    results = {
        'model_type': model.__class__.__name__,
        'test_samples': len(y_test)
    }
    
    # Two-stage results
    if model.use_two_stage:
        two_stage_results = evaluate_rainfall_model(model, X_test, y_test)
        results['two_stage'] = two_stage_results
    
    return results


def print_model_summary():
    """
    Print summary of available models and their capabilities.
    """
    print("🏗️ DS108 RAINFALL PREDICTION MODELS SUMMARY")
    print("=" * 60)
    
    available = get_available_models()
    
    for family, models in available.items():
        print(f"\n📊 {family.upper().replace('_', ' ')} MODELS:")
        for model in models:
            print(f"   • {model}")
            
        if family != 'time_series':
            print(f"   ✅ Two-stage support: YES (configurable)")
            print(f"   📈 Metrics: Classification (ROC-AUC) + Regression (MAE, RMSE, R2)")
        else:
            print(f"   ⚠️ Two-stage support: NO (single-stage only)")
            print(f"   📈 Metrics: Regression (MAE, RMSE, R2)")
    
    print(f"\n🔧 Usage:")
    print(f"   from src.models import create_model")
    print(f"   model = create_model('tree_based', 'RandomForestRainfallModel', use_two_stage=True)")
    print(f"   model.fit(X_train, y_train)")
    print(f"   predictions = model.predict(X_test)") 