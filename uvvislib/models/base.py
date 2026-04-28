"""
Base model class for UV-Vis analysis models.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, Union
import numpy as np
import pandas as pd
import logging
from pathlib import Path

from ..utils.config import Config


class BaseModel(ABC):
    """
    Abstract base class for all models in the UV-Vis analysis library.
    
    This class defines the common interface that all models must implement,
    including training, prediction, evaluation, and persistence methods.
    """
    
    def __init__(self, config: Config, model_name: str = "base_model"):
        """
        Initialize base model.
        
        Args:
            config: Configuration object
            model_name: Name of the model
        """
        self.config = config
        self.model_name = model_name
        self.logger = logging.getLogger(f"{__name__}.{model_name}")
        
        # Model state
        self.is_fitted = False
        self.training_history: Dict[str, list] = {}
        self.feature_importance: Optional[np.ndarray] = None
        
        # Model parameters
        self.model_params: Dict[str, Any] = {}
        
        # Data storage
        self.feature_columns: Optional[list] = None
        self.target_columns: Optional[list] = None
        self.n_features: Optional[int] = None
        self.n_targets: Optional[int] = None
    
    @abstractmethod
    def fit(
        self, 
        X: Union[np.ndarray, pd.DataFrame], 
        y: Union[np.ndarray, pd.DataFrame],
        **kwargs
    ) -> 'BaseModel':
        """
        Fit the model to the training data.
        
        Args:
            X: Training features
            y: Training targets
            **kwargs: Additional training parameters
            
        Returns:
            Self for method chaining
        """
        pass
    
    @abstractmethod
    def predict(
        self, 
        X: Union[np.ndarray, pd.DataFrame]
    ) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Features to predict on
            
        Returns:
            Predictions
        """
        pass
    
    def fit_predict(
        self, 
        X: Union[np.ndarray, pd.DataFrame], 
        y: Union[np.ndarray, pd.DataFrame],
        **kwargs
    ) -> np.ndarray:
        """
        Fit the model and make predictions on the same data.
        
        Args:
            X: Training features
            y: Training targets
            **kwargs: Additional training parameters
            
        Returns:
            Predictions
        """
        self.fit(X, y, **kwargs)
        return self.predict(X)
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """
        Get feature importance scores if available.
        
        Returns:
            Feature importance array or None if not available
        """
        return self.feature_importance
    
    def get_training_history(self) -> Dict[str, list]:
        """
        Get training history if available.
        
        Returns:
            Dictionary containing training metrics over time
        """
        return self.training_history.copy()
    
    def get_model_params(self) -> Dict[str, Any]:
        """
        Get model parameters.
        
        Returns:
            Dictionary of model parameters
        """
        return self.model_params.copy()
    
    def set_model_params(self, params: Dict[str, Any]) -> None:
        """
        Set model parameters.
        
        Args:
            params: Dictionary of model parameters
        """
        self.model_params.update(params)
    
    def validate_data(
        self, 
        X: Union[np.ndarray, pd.DataFrame], 
        y: Optional[Union[np.ndarray, pd.DataFrame]] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Validate and preprocess input data.
        
        Args:
            X: Features
            y: Targets (optional)
            
        Returns:
            Tuple of validated (X, y)
        """
        # Convert to numpy arrays
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.DataFrame):
            y = y.values
        
        # Check shapes
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        if y is not None and y.ndim == 1:
            y = y.reshape(-1, 1)
        
        # Check for NaN values
        if np.isnan(X).any():
            raise ValueError("Features contain NaN values")
        
        if y is not None and np.isnan(y).any():
            raise ValueError("Targets contain NaN values")
        
        # Check that X and y have same number of samples
        if y is not None and len(X) != len(y):
            raise ValueError("X and y must have the same number of samples")
        
        return X, y
    
    def _update_model_info(self, X: np.ndarray, y: Optional[np.ndarray]) -> None:
        """
        Update model information based on data.

        Args:
            X: Features
            y: Targets, or None for unsupervised models (e.g. clustering).
        """
        self.n_features = X.shape[1]
        if y is None:
            self.n_targets = 0
        else:
            self.n_targets = y.shape[1] if y.ndim > 1 else 1

        self.logger.info(f"Model info: {self.n_features} features, {self.n_targets} targets")
    
    def save_model(self, filepath: Union[str, Path]) -> None:
        """
        Save the model to disk.
        
        Args:
            filepath: Path where to save the model
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model-specific data
        self._save_model_impl(filepath)
        
        # Save metadata
        metadata = {
            'model_name': self.model_name,
            'model_params': self.model_params,
            'feature_columns': self.feature_columns,
            'target_columns': self.target_columns,
            'n_features': self.n_features,
            'n_targets': self.n_targets,
            'is_fitted': self.is_fitted,
            'training_history': self.training_history
        }
        
        metadata_file = filepath.with_suffix('.json')
        import json
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        self.logger.info(f"Model saved to {filepath}")
    
    @abstractmethod
    def _save_model_impl(self, filepath: Path) -> None:
        """
        Model-specific save implementation.
        
        Args:
            filepath: Path where to save the model
        """
        pass
    
    def load_model(self, filepath: Union[str, Path]) -> None:
        """
        Load the model from disk.
        
        Args:
            filepath: Path to the saved model
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        # Load metadata
        metadata_file = filepath.with_suffix('.json')
        if metadata_file.exists():
            import json
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            self.model_name = metadata.get('model_name', self.model_name)
            self.model_params = metadata.get('model_params', {})
            self.feature_columns = metadata.get('feature_columns')
            self.target_columns = metadata.get('target_columns')
            self.n_features = metadata.get('n_features')
            self.n_targets = metadata.get('n_targets')
            self.is_fitted = metadata.get('is_fitted', False)
            self.training_history = metadata.get('training_history', {})
        
        # Load model-specific data
        self._load_model_impl(filepath)
        
        self.logger.info(f"Model loaded from {filepath}")
    
    @abstractmethod
    def _load_model_impl(self, filepath: Path) -> None:
        """
        Model-specific load implementation.
        
        Args:
            filepath: Path to the saved model
        """
        pass
    
    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the model.
        
        Returns:
            Dictionary containing model summary information
        """
        summary = {
            'model_name': self.model_name,
            'is_fitted': self.is_fitted,
            'n_features': self.n_features,
            'n_targets': self.n_targets,
            'feature_columns': self.feature_columns,
            'target_columns': self.target_columns,
            'model_params': self.model_params,
            'has_feature_importance': self.feature_importance is not None,
            'has_training_history': len(self.training_history) > 0
        }
        
        return summary
    
    def __repr__(self) -> str:
        """String representation of the model."""
        fitted_str = "fitted" if self.is_fitted else "not fitted"
        return f"{self.__class__.__name__}({self.model_name}, {fitted_str})"
    
    def __str__(self) -> str:
        """String representation of the model."""
        return self.__repr__() 