"""
Visualization module for UV-Vis spectral analysis.

This module provides comprehensive plotting functions for spectral data analysis,
model evaluation, and results visualization.
"""

from .plots import (
    Plotter,
    plot_spectra,
    plot_prediction_vs_actual,
    plot_feature_importance,
    plot_training_history,
    plot_cv_results
)

__all__ = [
    'Plotter',
    'plot_spectra',
    'plot_prediction_vs_actual',
    'plot_feature_importance',
    'plot_training_history',
    'plot_cv_results'
]
