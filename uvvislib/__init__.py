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
from .models.mlp import MLPRegressor
from .models.cnn import CNNRegressor
from .models.cnn_mlp import CNNMLPRegressor
from .models.random_forest import RandomForestRegressor
from .models.clustering import SpectralClusterer, ClusteringAnalyzer

# Evaluation imports
from .evaluation.metrics import (
    r2_score, r2_score_single, polyfit, mape, rmse, rmse_exp,
    neg_rmse_exp, compute_evaluation, correlation_analysis, bias_analysis
)
from .evaluation.cross_validation import (
    DoubleKFoldCV, NestedCrossValidation, create_cv_scorers, 
    stratified_regression_cv
)

# Generative imports
from .generative import (
    BaseGenerativeModel,
    CGAN,
    SmoteRegression,
    KnnInterpolation,
    NoiseAugmenter,
)
from .generative.evaluation import (
    spectral_fidelity_metrics,
    pca_metrics,
    downstream_utility,
    reliability_score,
    compare_generators,
)

# Visualization imports
from .visualization.plots import Plotter

# Persistence imports
from .persistence.model_manager import ModelManager
from .persistence.experiment_manager import ExperimentManager
from .persistence.data_persistence import DataPersistence

__all__ = [
    # Core utilities
    "Config",
    "setup_logging", 
    
    # Data handling
    "DataLoader",
    "Preprocessor",
    
    # Models
    "BaseModel",
    "MLPRegressor",
    "CNNRegressor", 
    "CNNMLPRegressor",
    "RandomForestRegressor",
    "SpectralClusterer",
    "ClusteringAnalyzer",
    
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
    
    # Generative models
    "BaseGenerativeModel",
    "CGAN",
    "SmoteRegression",
    "KnnInterpolation",
    "NoiseAugmenter",

    # Generative evaluation
    "spectral_fidelity_metrics",
    "pca_metrics",
    "downstream_utility",
    "reliability_score",
    "compare_generators",

    # Visualization
    "Plotter",
    
    # Persistence
    "ModelManager",
    "ExperimentManager",
    "DataPersistence"
]
