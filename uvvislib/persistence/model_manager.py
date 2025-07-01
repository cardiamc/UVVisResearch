"""
Model persistence manager for UV-Vis analysis library.

This module provides utilities for saving, loading, and managing trained models.
"""

import json
import pickle
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import logging
from datetime import datetime

from ..utils.config import Config
from ..models.base import BaseModel


class ModelManager:
    """
    Manager for model persistence operations.
    
    This class handles saving, loading, and managing trained models
    with their associated metadata and configurations.
    """
    
    def __init__(self, config: Config, base_path: str = "./models"):
        """
        Initialize model manager.
        
        Args:
            config: Configuration object
            base_path: Base directory for model storage
        """
        self.config = config
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        # Model registry
        self.model_registry: Dict[str, Dict[str, Any]] = {}
        self._load_registry()
    
    def save_model(
        self, 
        model: BaseModel, 
        model_name: str,
        experiment_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save a trained model with metadata.
        
        Args:
            model: Trained model to save
            model_name: Name for the model
            experiment_name: Name of the experiment (optional)
            metadata: Additional metadata to save
            
        Returns:
            Path to saved model
        """
        if not model.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        # Create model directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if experiment_name:
            model_dir = self.base_path / experiment_name / model_name / timestamp
        else:
            model_dir = self.base_path / model_name / timestamp
        
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = model_dir / "model.pkl"
        model.save_model(model_path)
        
        # Prepare metadata
        model_metadata = {
            'model_name': model_name,
            'model_type': model.__class__.__name__,
            'timestamp': timestamp,
            'experiment_name': experiment_name,
            'model_params': model.get_model_params(),
            'model_summary': model.get_model_summary(),
            'config': self.config.to_dict(),
            'additional_metadata': metadata or {}
        }
        
        # Save metadata
        metadata_path = model_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(model_metadata, f, indent=2, default=str)
        
        # Update registry
        model_id = f"{model_name}_{timestamp}"
        self.model_registry[model_id] = {
            'path': str(model_dir),
            'metadata': model_metadata
        }
        self._save_registry()
        
        self.logger.info(f"Model saved to: {model_dir}")
        return str(model_dir)
    
    def load_model(
        self, 
        model_path: Union[str, Path],
        model_class: Optional[type] = None
    ) -> BaseModel:
        """
        Load a saved model.
        
        Args:
            model_path: Path to the saved model
            model_class: Model class to use for loading (optional)
            
        Returns:
            Loaded model
        """
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model path does not exist: {model_path}")
        
        # Load metadata
        metadata_path = model_path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {}
        
        # Determine model class
        if model_class is None:
            model_type = metadata.get('model_type', 'BaseModel')
            # Import model classes dynamically
            from ..models import (
                MLPRegressor, CNNRegressor, CNNMLPRegressor, 
                RandomForestRegressor, SpectralClusterer
            )
            
            model_classes = {
                'MLPRegressor': MLPRegressor,
                'CNNRegressor': CNNRegressor,
                'CNNMLPRegressor': CNNMLPRegressor,
                'RandomForestRegressor': RandomForestRegressor,
                'SpectralClusterer': SpectralClusterer
            }
            
            model_class = model_classes.get(model_type)
            if model_class is None:
                raise ValueError(f"Unknown model type: {model_type}")
        
        # Create model instance
        model = model_class(self.config)
        
        # Load model
        model_file = model_path / "model.pkl"
        if model_file.exists():
            model.load_model(model_file)
        else:
            raise FileNotFoundError(f"Model file not found: {model_file}")
        
        self.logger.info(f"Model loaded from: {model_path}")
        return model
    
    def list_models(
        self, 
        experiment_name: Optional[str] = None,
        model_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        List available models.
        
        Args:
            experiment_name: Filter by experiment name
            model_name: Filter by model name
            
        Returns:
            List of model information dictionaries
        """
        models = []
        
        for model_id, info in self.model_registry.items():
            metadata = info['metadata']
            
            # Apply filters
            if experiment_name and metadata.get('experiment_name') != experiment_name:
                continue
            if model_name and metadata.get('model_name') != model_name:
                continue
            
            models.append({
                'model_id': model_id,
                'path': info['path'],
                'metadata': metadata
            })
        
        return models
    
    def delete_model(self, model_path: Union[str, Path]) -> bool:
        """
        Delete a saved model.
        
        Args:
            model_path: Path to the model to delete
            
        Returns:
            True if successful, False otherwise
        """
        model_path = Path(model_path)
        
        if not model_path.exists():
            self.logger.warning(f"Model path does not exist: {model_path}")
            return False
        
        try:
            import shutil
            shutil.rmtree(model_path)
            
            # Remove from registry
            for model_id, info in list(self.model_registry.items()):
                if info['path'] == str(model_path):
                    del self.model_registry[model_id]
                    break
            
            self._save_registry()
            self.logger.info(f"Model deleted: {model_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete model {model_path}: {e}")
            return False
    
    def get_model_info(self, model_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Get information about a saved model.
        
        Args:
            model_path: Path to the model
            
        Returns:
            Model information dictionary
        """
        model_path = Path(model_path)
        metadata_path = model_path / "metadata.json"
        
        if not metadata_path.exists():
            raise FileNotFoundError(f"Model metadata not found: {metadata_path}")
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        return metadata
    
    def _load_registry(self) -> None:
        """Load model registry from file."""
        registry_path = self.base_path / "model_registry.json"
        
        if registry_path.exists():
            try:
                with open(registry_path, 'r') as f:
                    self.model_registry = json.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load model registry: {e}")
                self.model_registry = {}
    
    def _save_registry(self) -> None:
        """Save model registry to file."""
        registry_path = self.base_path / "model_registry.json"
        
        try:
            with open(registry_path, 'w') as f:
                json.dump(self.model_registry, f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Failed to save model registry: {e}") 