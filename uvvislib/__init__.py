"""
UV-Vis and NIR Data Analysis Library

A modular, professional Python library for UV-Vis and NIR data analysis, 
machine learning, and model interpretation.
"""

__version__ = "0.1.0"
__author__ = "UVVisResearch Team"

# Core imports
from .utils.config import Config
from .utils.logging import setup_logging
from .data.loader import DataLoader
from .data.preprocessing import Preprocessor
from .models.base import BaseModel
from .models.mlp import MLPModel
from .models.cnn import CNNModel
from .models.cnn_mlp import CNNMLPRegressor
from .models.random_forest import RandomForestModel

# Evaluation imports
from .evaluation.metrics import (
    r2_score, r2_score_single, polyfit, mape, rmse, rmse_exp,
    neg_rmse_exp, compute_evaluation, correlation_analysis, bias_analysis
)
from .evaluation.cross_validation import (
    DoubleKFoldCV, NestedCrossValidation, create_cv_scorers, 
    stratified_regression_cv
)

# Visualization imports
from .visualization.plots import Plotter

__all__ = [
    # Core utilities
    "Config",
    "setup_logging", 
    
    # Data handling
    "DataLoader",
    "Preprocessor",
    
    # Models
    "BaseModel",
    "MLPModel",
    "CNNModel", 
    "CNNMLPRegressor",
    "RandomForestModel",
    
    # Evaluation metrics
    "r2_score",
    "r2_score_single",
    "polyfit", 
    "mape",
    "rmse",
    "rmse_exp",
    "neg_rmse_exp",
    "compute_evaluation",
    "correlation_analysis",
    "bias_analysis",
    
    # Cross-validation
    "DoubleKFoldCV",
    "NestedCrossValidation",
    "create_cv_scorers",
    "stratified_regression_cv",
    
    # Visualization
    "Plotter"
]
