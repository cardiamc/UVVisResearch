"""
Models module for UV-Vis spectral analysis.

This module provides various machine learning models including:
- Base model class
- MLP (Multi-Layer Perceptron)
- CNN (Convolutional Neural Network)
- CNN-MLP (Hybrid model)
- Random Forest
"""

from .base import BaseModel
from .mlp import MLPModel
from .cnn import CNNModel
from .cnn_mlp import CNNMLPRegressor
from .random_forest import RandomForestModel

__all__ = [
    'BaseModel',
    'MLPModel', 
    'CNNModel',
    'CNNMLPRegressor',
    'RandomForestModel'
]
