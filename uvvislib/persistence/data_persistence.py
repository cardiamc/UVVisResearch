"""
Data persistence utilities for UV-Vis analysis library.

This module provides utilities for saving and loading datasets,
preprocessed data, and analysis results.
"""

import json
import pickle
from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple
import logging
import pandas as pd
import numpy as np

from ..utils.config import Config


class DataPersistence:
    """
    Data persistence manager for UV-Vis analysis.
    
    This class handles saving and loading datasets, preprocessed data,
    and analysis results in various formats.
    """
    
    def __init__(self, config: Config, base_path: str = "./data"):
        """
        Initialize data persistence manager.
        
        Args:
            config: Configuration object
            base_path: Base directory for data storage
        """
        self.config = config
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        # Create subdirectories
        (self.base_path / "raw").mkdir(exist_ok=True)
        (self.base_path / "processed").mkdir(exist_ok=True)
        (self.base_path / "features").mkdir(exist_ok=True)
        (self.base_path / "results").mkdir(exist_ok=True)
    
    def save_dataset(
        self,
        features: pd.DataFrame,
        targets: pd.DataFrame,
        dataset_name: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save a complete dataset (features and targets).
        
        Args:
            features: Feature DataFrame
            targets: Target DataFrame
            dataset_name: Name for the dataset
            metadata: Additional metadata
            
        Returns:
            Path to saved dataset
        """
        dataset_path = self.base_path / "processed" / dataset_name
        dataset_path.mkdir(parents=True, exist_ok=True)
        
        # Save features and targets
        features_path = dataset_path / "features.csv"
        targets_path = dataset_path / "targets.csv"
        
        features.to_csv(features_path, index=False)
        targets.to_csv(targets_path, index=False)
        
        # Save metadata
        dataset_metadata = {
            'dataset_name': dataset_name,
            'features_shape': features.shape,
            'targets_shape': targets.shape,
            'feature_columns': list(features.columns),
            'target_columns': list(targets.columns),
            'config': self.config.to_dict(),
            'additional_metadata': metadata or {}
        }
        
        metadata_path = dataset_path / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(dataset_metadata, f, indent=2, default=str)
        
        self.logger.info(f"Dataset saved to: {dataset_path}")
        return str(dataset_path)
    
    def load_dataset(
        self, 
        dataset_name: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        """
        Load a complete dataset.
        
        Args:
            dataset_name: Name of the dataset to load
            
        Returns:
            Tuple of (features, targets, metadata)
        """
        dataset_path = self.base_path / "processed" / dataset_name
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        
        # Load features and targets
        features_path = dataset_path / "features.csv"
        targets_path = dataset_path / "targets.csv"
        
        if not features_path.exists() or not targets_path.exists():
            raise FileNotFoundError(f"Dataset files not found in: {dataset_path}")
        
        features = pd.read_csv(features_path)
        targets = pd.read_csv(targets_path)
        
        # Load metadata
        metadata_path = dataset_path / "metadata.json"
        metadata = {}
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        
        self.logger.info(f"Dataset loaded from: {dataset_path}")
        return features, targets, metadata
    
    def save_features(
        self,
        features: pd.DataFrame,
        feature_name: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save extracted features.
        
        Args:
            features: Feature DataFrame
            feature_name: Name for the features
            metadata: Additional metadata
            
        Returns:
            Path to saved features
        """
        features_path = self.base_path / "features" / f"{feature_name}.csv"
        features_path.parent.mkdir(parents=True, exist_ok=True)
        
        features.to_csv(features_path, index=False)
        
        # Save metadata
        if metadata:
            metadata_path = features_path.with_suffix('.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
        
        self.logger.info(f"Features saved to: {features_path}")
        return str(features_path)
    
    def load_features(
        self, 
        feature_name: str
    ) -> Tuple[pd.DataFrame, Optional[Dict[str, Any]]]:
        """
        Load extracted features.
        
        Args:
            feature_name: Name of the features to load
            
        Returns:
            Tuple of (features, metadata)
        """
        features_path = self.base_path / "features" / f"{feature_name}.csv"
        
        if not features_path.exists():
            raise FileNotFoundError(f"Features not found: {features_path}")
        
        features = pd.read_csv(features_path)
        
        # Load metadata if available
        metadata_path = features_path.with_suffix('.json')
        metadata = None
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        
        self.logger.info(f"Features loaded from: {features_path}")
        return features, metadata
    
    def save_preprocessing_pipeline(
        self,
        preprocessor,
        pipeline_name: str
    ) -> str:
        """
        Save a fitted preprocessing pipeline.
        
        Args:
            preprocessor: Fitted preprocessor object
            pipeline_name: Name for the pipeline
            
        Returns:
            Path to saved pipeline
        """
        pipeline_path = self.base_path / "processed" / f"{pipeline_name}_pipeline.pkl"
        pipeline_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(pipeline_path, 'wb') as f:
            pickle.dump(preprocessor, f)
        
        self.logger.info(f"Preprocessing pipeline saved to: {pipeline_path}")
        return str(pipeline_path)
    
    def load_preprocessing_pipeline(
        self, 
        pipeline_name: str
    ):
        """
        Load a saved preprocessing pipeline.
        
        Args:
            pipeline_name: Name of the pipeline to load
            
        Returns:
            Loaded preprocessor object
        """
        pipeline_path = self.base_path / "processed" / f"{pipeline_name}_pipeline.pkl"
        
        if not pipeline_path.exists():
            raise FileNotFoundError(f"Pipeline not found: {pipeline_path}")
        
        with open(pipeline_path, 'rb') as f:
            preprocessor = pickle.load(f)
        
        self.logger.info(f"Preprocessing pipeline loaded from: {pipeline_path}")
        return preprocessor
    
    def save_analysis_results(
        self,
        results: Dict[str, Any],
        analysis_name: str,
        format: str = "json"
    ) -> str:
        """
        Save analysis results.
        
        Args:
            results: Results dictionary
            analysis_name: Name for the analysis
            format: Format to save in ('json', 'pkl')
            
        Returns:
            Path to saved results
        """
        results_path = self.base_path / "results" / f"{analysis_name}.{format}"
        results_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "json":
            # Convert numpy arrays to lists for JSON serialization
            serializable_results = self._make_serializable(results)
            with open(results_path, 'w') as f:
                json.dump(serializable_results, f, indent=2, default=str)
        elif format == "pkl":
            with open(results_path, 'wb') as f:
                pickle.dump(results, f)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        self.logger.info(f"Analysis results saved to: {results_path}")
        return str(results_path)
    
    def load_analysis_results(
        self, 
        analysis_name: str,
        format: str = "json"
    ) -> Dict[str, Any]:
        """
        Load analysis results.
        
        Args:
            analysis_name: Name of the analysis to load
            format: Format of the saved results ('json', 'pkl')
            
        Returns:
            Loaded results dictionary
        """
        results_path = self.base_path / "results" / f"{analysis_name}.{format}"
        
        if not results_path.exists():
            raise FileNotFoundError(f"Analysis results not found: {results_path}")
        
        if format == "json":
            with open(results_path, 'r') as f:
                results = json.load(f)
        elif format == "pkl":
            with open(results_path, 'rb') as f:
                results = pickle.load(f)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        self.logger.info(f"Analysis results loaded from: {results_path}")
        return results
    
    def list_datasets(self) -> List[str]:
        """
        List available datasets.
        
        Returns:
            List of dataset names
        """
        datasets_path = self.base_path / "processed"
        
        if not datasets_path.exists():
            return []
        
        datasets = []
        for item in datasets_path.iterdir():
            if item.is_dir() and (item / "features.csv").exists():
                datasets.append(item.name)
        
        return datasets
    
    def list_features(self) -> List[str]:
        """
        List available feature files.
        
        Returns:
            List of feature names
        """
        features_path = self.base_path / "features"
        
        if not features_path.exists():
            return []
        
        features = []
        for item in features_path.iterdir():
            if item.is_file() and item.suffix == '.csv':
                features.append(item.stem)
        
        return features
    
    def _make_serializable(self, obj: Any) -> Any:
        """Convert object to JSON serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        else:
            return obj 