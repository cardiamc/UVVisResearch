"""
Evaluation module for UV-Vis spectral analysis.

This module provides comprehensive evaluation tools including:
- Metrics: R2, RMSE, MAPE, correlation analysis, bias analysis
- Cross-validation: Double k-fold, nested CV, stratified CV
"""

from .metrics import (
    r2_score,
    r2_score_single,
    polyfit,
    mape,
    rmse,
    rmse_exp,
    neg_rmse_exp,
    compute_evaluation,
    correlation_analysis,
    bias_analysis
)

from .cross_validation import (
    DoubleKFoldCV,
    NestedCrossValidation,
    create_cv_scorers,
    stratified_regression_cv
)

__all__ = [
    # Metrics
    'r2_score',
    'r2_score_single', 
    'polyfit',
    'mape',
    'rmse',
    'rmse_exp',
    'neg_rmse_exp',
    'compute_evaluation',
    'correlation_analysis',
    'bias_analysis',
    
    # Cross-validation
    'DoubleKFoldCV',
    'NestedCrossValidation',
    'create_cv_scorers',
    'stratified_regression_cv'
]
