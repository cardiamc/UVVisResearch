"""
Configuration management for UV-Vis analysis library.
"""

import json
import os
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union
import numpy as np


@dataclass
class Config:
    """
    Configuration class for UV-Vis analysis parameters.
    
    This class manages all configuration parameters used throughout the library,
    including data paths, model parameters, training settings, and evaluation metrics.
    """
    
    # Data configuration
    data_path: str = "./Data/"
    uv_vis_data_file: str = "abs_spectra_100mm.csv"
    chemical_data_file: str = "Full_chemical_analysis.csv"
    
    # Feature configuration
    # Default grid: 200–727.5 nm at 2.5 nm step → 212 features
    # (np.arange uses a half-open interval, so end=730.0 includes 727.5)
    wavelength_start: float = 200.0
    wavelength_end: float = 730.0
    wavelength_step: float = 2.5
    categorical_features: List[str] = field(default_factory=lambda: ['FASE PROCESSO'])
    
    # Target variables (water quality indicators)
    target_variables: List[str] = field(default_factory=lambda: [
        "CONDUCIBILITA'", "POTENZIALE REDOX", "NITRITI", "NITRATI", "FLUORURI", 
        "SODIO", "CALCIO", "MAGNESIO", "SOLFATI", "CLORURI", "DUREZZA (da calcolo)", 
        "pH", "AMMONIO", "CLORATI", "ARSENICO", "ANTIMONIO", "ALLUMINIO", 
        "CADMIO", "CROMO TOTALE", "FERRO", "MANGANESE", "NICHEL", "PIOMBO",
        "RAME", "SELENIO", "MERCURIO", "BORO", "BROMODICLOROMETANO", 
        "BROMOFORMIO", "CLOROFORMIO", "cis-1,2-DICLOROETILENE", 
        "DIBROMOCLOROMETANO", "TETRACLOROETILENE", "trans-1,2-DICLOROETILENE", 
        "TRICLOROETILENE", "1,1-DICLOROETANO", "1,1-DICLOROETILENE",
        "1,1,1-TRICLOROETANO", "1,2-DIBROMOETANO", "1,2-DICLOROETANO",
        "1,2-DICLOROPROPANO", "ESACLOROBUTADIENE", "CLORURO DI VINILE", 
        "METILTERZIARBUTILETERE (MTBE)", "RESIDUO FISSO A 180°C", 
        "1,2,4-TRICLOROBENZENE", "1,2,3-TRICLOROBENZENE", "1,3-DICLOROBENZENE",
        "1,3,5-TRIMETILBENZENE", "n-PROPILBENZENE", "iso-PROPILBENZENE", 
        "STIRENE", "o-XILENE", "(m+p)-XILENE", "ETILBENZENE", "TOLUENE",
        "CLORITI", "TOC", "CONTA DI MICRORGANISMI VITALI A 36°C", 
        "CONTA DI MICRORGANISMI VITALI A 22°C"
    ])
    
    # Preprocessing configuration
    log_target: bool = False
    use_categorical_features: bool = False
    apply_pca: bool = False
    pca_components: int = 15
    apply_smoothing: bool = True
    gaussian_sigma: float = 1.5
    
    # Cross-validation configuration
    k_fold_splits: int = 5
    k_fold_inner_splits: int = 4
    random_state: int = 20
    
    # Early stopping configuration
    early_stopping_patience: int = 100
    early_stopping_min_delta: float = 0.35
    
    # Model configuration
    mlp_config: Dict[str, Any] = field(default_factory=lambda: {
        'learning_rate_range': (0.0001, 0.001),
        'hidden_size_range': (350, 1300),
        'weight_decay_range': (0.001, 0.002),
        'epochs': 8000,
        'activation_functions': ['sigmoid']
    })
    
    cnn_config: Dict[str, Any] = field(default_factory=lambda: {
        'learning_rate_range': (0.0001, 0.001),
        'hidden_size_range': (400, 1100),
        'kernel_size_options': [2, 3, 4, 5, 6, 8],
        'stride_size_options': [2, 3, 4, 5, 6, 8],
        'weight_decay_range': (0.001, 0.002),
        'epochs': 5000
    })
    
    # Random Forest configuration
    rf_config: Dict[str, Any] = field(default_factory=lambda: {
        'criterion': ['poisson', 'friedman_mse', 'squared_error', 'absolute_error'],
        'n_estimators': [5, 10, 15, 20, 23, 25, 30],
        'max_depth': [2, 5, 6, 7, 8, 12, 15, 19, None],
        'min_samples_leaf': [1, 3, 5, 10],
        'min_samples_split': [2, 3, 5, 8, 10, 15, 20, 30]
    })

    # Generative model configuration (CGAN + baselines)
    # n_z=200 matches the research code checkpoints; SKILL.md example uses 62 —
    # change this field if you want to train a new model from scratch with a
    # different noise dimension (pre-trained weights are NOT cross-compatible).
    cgan_config: Dict[str, Any] = field(default_factory=lambda: {
        # Architecture
        'input_size':    212,
        'n_z':           200,
        'class_num':     1,
        # Optimiser
        'lrG':           1e-4,
        'lrD':           1e-4,
        'beta1':         0.5,
        'beta2':         0.999,
        'weight_decay':  1e-4,
        # Training
        'epochs':        200,
        'batch_size':    64,
        'lambda_reg':    0.01,
        'random_state':  42,
        # Sampling
        'n_samples_gen': 500,
        # SMOTE
        'smote_n_bins':      10,
        'smote_k_neighbors': 5,
        # kNN interpolation
        'knn_k_neighbors': 5,
        # Noise augmentation
        'noise_sigma': 0.1,
    })
    
    # Output configuration
    output_dir: str = "./LOG/"
    save_models: bool = True
    save_predictions: bool = True
    save_plots: bool = True
    
    def _format_wavelength(self, wav: float) -> str:
        """Format a wavelength as the column name used in the CSV files.

        Integer-valued wavelengths render without a trailing ``.0`` ('200'),
        non-integers keep a decimal ('202.5'). This matches both common
        conventions in the source data files.
        """
        return str(int(wav)) if float(wav).is_integer() else str(wav)

    @property
    def feature_wavelengths(self) -> List[str]:
        """Generate feature wavelength strings."""
        return [self._format_wavelength(wav) for wav in np.arange(
            self.wavelength_start,
            self.wavelength_end,
            self.wavelength_step,
        )]

    @property
    def spectrum_wavelengths(self) -> List[str]:
        """Generate spectrum wavelength strings."""
        return [self._format_wavelength(wav) for wav in np.arange(
            self.wavelength_start,
            self.wavelength_end,
            self.wavelength_step,
        )]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'data_path': self.data_path,
            'uv_vis_data_file': self.uv_vis_data_file,
            'chemical_data_file': self.chemical_data_file,
            'wavelength_start': self.wavelength_start,
            'wavelength_end': self.wavelength_end,
            'wavelength_step': self.wavelength_step,
            'target_variables': self.target_variables,
            'log_target': self.log_target,
            'use_categorical_features': self.use_categorical_features,
            'apply_smoothing': self.apply_smoothing,
            'gaussian_sigma': self.gaussian_sigma,
            'k_fold_splits': self.k_fold_splits,
            'k_fold_inner_splits': self.k_fold_inner_splits,
            'random_state': self.random_state,
            'early_stopping_patience': self.early_stopping_patience,
            'early_stopping_min_delta': self.early_stopping_min_delta,
            'mlp_config': self.mlp_config,
            'cnn_config': self.cnn_config,
            'rf_config': self.rf_config,
            'cgan_config': self.cgan_config,
            'output_dir': self.output_dir
        }
    
    def save(self, filepath: str) -> None:
        """Save configuration to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'Config':
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        # Create new instance with loaded data
        config = cls()
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        return config
    
    def validate(self) -> bool:
        """Validate configuration parameters."""
        if self.wavelength_start >= self.wavelength_end:
            raise ValueError("wavelength_start must be less than wavelength_end")
        
        if self.wavelength_step <= 0:
            raise ValueError("wavelength_step must be positive")
        
        if self.k_fold_splits < 2:
            raise ValueError("k_fold_splits must be at least 2")
        
        if self.early_stopping_patience < 1:
            raise ValueError("early_stopping_patience must be at least 1")
        
        return True 