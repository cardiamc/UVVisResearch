"""
Experiment management for UV-Vis analysis library.

This module provides utilities for tracking experiments, saving results,
and managing experiment metadata.
"""

import json
import pickle
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import logging
from datetime import datetime
import pandas as pd
import numpy as np

from ..utils.config import Config


class ExperimentManager:
    """
    Manager for experiment tracking and results persistence.
    
    This class handles saving experiment results, configurations,
    and metadata for reproducible research.
    """
    
    def __init__(self, config: Config, base_path: str = "./experiments"):
        """
        Initialize experiment manager.
        
        Args:
            config: Configuration object
            base_path: Base directory for experiment storage
        """
        self.config = config
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        # Current experiment
        self.current_experiment: Optional[str] = None
        self.current_experiment_path: Optional[Path] = None
        
        # Experiment registry
        self.experiment_registry: Dict[str, Dict[str, Any]] = {}
        self._load_registry()
    
    def start_experiment(
        self, 
        experiment_name: str,
        description: str = "",
        tags: Optional[List[str]] = None
    ) -> str:
        """
        Start a new experiment.
        
        Args:
            experiment_name: Name of the experiment
            description: Experiment description
            tags: List of tags for the experiment
            
        Returns:
            Experiment ID
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_id = f"{experiment_name}_{timestamp}"
        
        # Create experiment directory
        experiment_path = self.base_path / experiment_id
        experiment_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (experiment_path / "models").mkdir(exist_ok=True)
        (experiment_path / "results").mkdir(exist_ok=True)
        (experiment_path / "plots").mkdir(exist_ok=True)
        (experiment_path / "data").mkdir(exist_ok=True)
        
        # Save experiment metadata
        metadata = {
            'experiment_id': experiment_id,
            'experiment_name': experiment_name,
            'description': description,
            'tags': tags or [],
            'start_time': timestamp,
            'config': self.config.to_dict(),
            'status': 'running'
        }
        
        metadata_path = experiment_path / "experiment_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        # Update registry
        self.experiment_registry[experiment_id] = {
            'path': str(experiment_path),
            'metadata': metadata
        }
        self._save_registry()
        
        # Set as current experiment
        self.current_experiment = experiment_id
        self.current_experiment_path = experiment_path
        
        self.logger.info(f"Started experiment: {experiment_id}")
        return experiment_id
    
    def end_experiment(self, status: str = "completed") -> None:
        """
        End the current experiment.
        
        Args:
            status: Final status of the experiment
        """
        if self.current_experiment is None:
            self.logger.warning("No active experiment to end")
            return
        
        # Update metadata
        if self.current_experiment_path:
            metadata_path = self.current_experiment_path / "experiment_metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                metadata['end_time'] = datetime.now().strftime("%Y%m%d_%H%M%S")
                metadata['status'] = status
                
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2, default=str)
        
        # Update registry
        if self.current_experiment in self.experiment_registry:
            self.experiment_registry[self.current_experiment]['metadata']['status'] = status
            self.experiment_registry[self.current_experiment]['metadata']['end_time'] = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._save_registry()
        
        self.logger.info(f"Ended experiment: {self.current_experiment} with status: {status}")
        
        # Clear current experiment
        self.current_experiment = None
        self.current_experiment_path = None
    
    def save_results(
        self, 
        results: Dict[str, Any],
        filename: str = "results.json"
    ) -> str:
        """
        Save experiment results.
        
        Args:
            results: Results dictionary to save
            filename: Name of the results file
            
        Returns:
            Path to saved results
        """
        if self.current_experiment_path is None:
            raise ValueError("No active experiment. Call start_experiment() first.")
        
        results_path = self.current_experiment_path / "results" / filename
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = self._make_serializable(results)
        
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        self.logger.info(f"Results saved to: {results_path}")
        return str(results_path)
    
    def save_predictions(
        self, 
        predictions: np.ndarray,
        true_values: np.ndarray,
        target_names: Optional[List[str]] = None,
        filename: str = "predictions.csv"
    ) -> str:
        """
        Save model predictions.
        
        Args:
            predictions: Model predictions
            true_values: True target values
            target_names: Names of target variables
            filename: Name of the predictions file
            
        Returns:
            Path to saved predictions
        """
        if self.current_experiment_path is None:
            raise ValueError("No active experiment. Call start_experiment() first.")
        
        predictions_path = self.current_experiment_path / "results" / filename
        
        # Create DataFrame
        if target_names is None:
            target_names = [f"target_{i}" for i in range(predictions.shape[1])]
        
        df = pd.DataFrame()
        for i, name in enumerate(target_names):
            df[f"{name}_pred"] = predictions[:, i]
            df[f"{name}_true"] = true_values[:, i]
        
        df.to_csv(predictions_path, index=False)
        
        self.logger.info(f"Predictions saved to: {predictions_path}")
        return str(predictions_path)
    
    def save_plot(
        self, 
        figure,
        filename: str,
        dpi: int = 300
    ) -> str:
        """
        Save a plot from the current experiment.
        
        Args:
            figure: Matplotlib figure to save
            filename: Name of the plot file
            dpi: DPI for the saved image
            
        Returns:
            Path to saved plot
        """
        if self.current_experiment_path is None:
            raise ValueError("No active experiment. Call start_experiment() first.")
        
        plot_path = self.current_experiment_path / "plots" / filename
        
        figure.savefig(plot_path, dpi=dpi, bbox_inches='tight')
        
        self.logger.info(f"Plot saved to: {plot_path}")
        return str(plot_path)
    
    def save_data(
        self, 
        data: Union[pd.DataFrame, np.ndarray],
        filename: str,
        format: str = "csv"
    ) -> str:
        """
        Save data from the current experiment.
        
        Args:
            data: Data to save
            filename: Name of the data file
            format: Format to save in ('csv', 'pkl', 'npy')
            
        Returns:
            Path to saved data
        """
        if self.current_experiment_path is None:
            raise ValueError("No active experiment. Call start_experiment() first.")
        
        data_path = self.current_experiment_path / "data" / filename
        
        if format == "csv":
            if isinstance(data, np.ndarray):
                data = pd.DataFrame(data)
            data.to_csv(data_path, index=False)
        elif format == "pkl":
            with open(data_path, 'wb') as f:
                pickle.dump(data, f)
        elif format == "npy":
            np.save(data_path, data)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        self.logger.info(f"Data saved to: {data_path}")
        return str(data_path)
    
    def list_experiments(
        self, 
        status: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        List available experiments.
        
        Args:
            status: Filter by experiment status
            tags: Filter by experiment tags
            
        Returns:
            List of experiment information dictionaries
        """
        experiments = []
        
        for experiment_id, info in self.experiment_registry.items():
            metadata = info['metadata']
            
            # Apply filters
            if status and metadata.get('status') != status:
                continue
            if tags:
                experiment_tags = metadata.get('tags', [])
                if not any(tag in experiment_tags for tag in tags):
                    continue
            
            experiments.append({
                'experiment_id': experiment_id,
                'path': info['path'],
                'metadata': metadata
            })
        
        return experiments
    
    def load_experiment_results(
        self, 
        experiment_id: str,
        filename: str = "results.json"
    ) -> Dict[str, Any]:
        """
        Load results from a specific experiment.
        
        Args:
            experiment_id: ID of the experiment
            filename: Name of the results file
            
        Returns:
            Experiment results
        """
        if experiment_id not in self.experiment_registry:
            raise ValueError(f"Experiment not found: {experiment_id}")
        
        experiment_path = Path(self.experiment_registry[experiment_id]['path'])
        results_path = experiment_path / "results" / filename
        
        if not results_path.exists():
            raise FileNotFoundError(f"Results file not found: {results_path}")
        
        with open(results_path, 'r') as f:
            results = json.load(f)
        
        return results
    
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
    
    def _load_registry(self) -> None:
        """Load experiment registry from file."""
        registry_path = self.base_path / "experiment_registry.json"
        
        if registry_path.exists():
            try:
                with open(registry_path, 'r') as f:
                    self.experiment_registry = json.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load experiment registry: {e}")
                self.experiment_registry = {}
    
    def _save_registry(self) -> None:
        """Save experiment registry to file."""
        registry_path = self.base_path / "experiment_registry.json"
        
        try:
            with open(registry_path, 'w') as f:
                json.dump(self.experiment_registry, f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Failed to save experiment registry: {e}") 