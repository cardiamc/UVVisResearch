"""
Models module for UV-Vis spectral analysis.

This module provides various machine learning models including:
- Base model class
- MLP (Multi-Layer Perceptron)
- CNN (Convolutional Neural Network)
- CNN-MLP (Hybrid model)
- Random Forest
- Clustering algorithms
"""

from .base import BaseModel
from .mlp import MLPRegressor
from .cnn import CNNRegressor
from .cnn_mlp import CNNMLPRegressor
from .random_forest import RandomForestRegressor
from .clustering import SpectralClusterer, ClusteringAnalyzer

__all__ = [
    'BaseModel',
    'MLPRegressor', 
    'CNNRegressor',
    'CNNMLPRegressor',
    'RandomForestRegressor',
    'SpectralClusterer',
    'ClusteringAnalyzer'
]
