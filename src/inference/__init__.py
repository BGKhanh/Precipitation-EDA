"""
DS108 Inference Module
======================
Exposes standalone predictor and MLflow Pyfunc wrapper.
"""
from src.inference.predictor import StandaloneRainfallPredictor
from src.inference.pyfunc import RainfallForecastingPyfuncWrapper

__all__ = ["StandaloneRainfallPredictor", "RainfallForecastingPyfuncWrapper"]

