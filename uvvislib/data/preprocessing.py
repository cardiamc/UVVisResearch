"""
Data preprocessing utilities for UV-Vis analysis.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional, Dict, Any, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
import logging

from ..utils.config import Config


class Preprocessor:
    """
    Data preprocessing pipeline for UV-Vis analysis.
    
    Handles data cleaning, normalization, feature engineering, and other preprocessing steps.
    """
    
    def __init__(self, config: Config):
        """
        Initialize preprocessor with configuration.
        
        Args:
            config: Configuration object containing preprocessing parameters
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Preprocessing transformers
        self.scaler = None
        self.pca = None
        self.imputer = None
        
        # Data storage
        self.features_df: Optional[pd.DataFrame] = None
        self.targets_df: Optional[pd.DataFrame] = None
        self.processed_features: Optional[pd.DataFrame] = None
        self.processed_targets: Optional[pd.DataFrame] = None
        
        # Column information
        self.feature_columns: List[str] = []
        self.target_columns: List[str] = []
        self.categorical_columns: List[str] = []
    
    def clean_data(
        self, 
        features_df: pd.DataFrame, 
        targets_df: pd.DataFrame,
        remove_nan_targets: bool = True,
        remove_outliers: bool = False,
        outlier_threshold: float = 3.0
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Clean the dataset by removing NaN values and outliers.
        
        Args:
            features_df: Features DataFrame
            targets_df: Targets DataFrame
            remove_nan_targets: Whether to remove rows with NaN targets
            remove_outliers: Whether to remove outliers
            outlier_threshold: Z-score threshold for outlier detection
            
        Returns:
            Tuple of cleaned (features_df, targets_df)
        """
        self.logger.info("Cleaning dataset")
        
        # Store original data
        self.features_df = features_df.copy()
        self.targets_df = targets_df.copy()
        
        # Remove rows with NaN targets
        if remove_nan_targets:
            initial_rows = len(targets_df)
            valid_indices = targets_df.notna().all(axis=1)
            targets_df = targets_df[valid_indices]
            features_df = features_df[valid_indices]
            
            removed_rows = initial_rows - len(targets_df)
            self.logger.info(f"Removed {removed_rows} rows with NaN targets")
        
        # Remove outliers if requested
        if remove_outliers:
            features_df, targets_df = self._remove_outliers(
                features_df, targets_df, outlier_threshold
            )
        
        # Handle missing values in features
        if features_df.isna().any().any():
            self.logger.info("Handling missing values in features")
            features_df = self._handle_missing_values(features_df)
        
        self.logger.info(f"Cleaned data shape: {features_df.shape}")
        return features_df, targets_df
    
    def _remove_outliers(
        self, 
        features_df: pd.DataFrame, 
        targets_df: pd.DataFrame, 
        threshold: float
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Remove outliers based on Z-score."""
        initial_rows = len(features_df)
        
        # Calculate Z-scores for features
        z_scores = np.abs((features_df - features_df.mean()) / features_df.std())
        
        # Find rows with outliers
        outlier_mask = (z_scores > threshold).any(axis=1)
        
        # Remove outliers
        features_df = features_df[~outlier_mask]
        targets_df = targets_df[~outlier_mask]
        
        removed_rows = initial_rows - len(features_df)
        self.logger.info(f"Removed {removed_rows} outlier rows")
        
        return features_df, targets_df
    
    def _handle_missing_values(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in features."""
        if self.imputer is None:
            self.imputer = SimpleImputer(strategy='mean')
            features_df = pd.DataFrame(
                self.imputer.fit_transform(features_df),
                columns=features_df.columns,
                index=features_df.index
            )
        else:
            features_df = pd.DataFrame(
                self.imputer.transform(features_df),
                columns=features_df.columns,
                index=features_df.index
            )
        
        return features_df
    
    def apply_target_transformation(
        self, 
        targets_df: pd.DataFrame, 
        log_transform: Optional[bool] = None
    ) -> pd.DataFrame:
        """
        Apply transformations to target variables.
        
        Args:
            targets_df: Targets DataFrame
            log_transform: Whether to apply log transformation. If None, uses config.
            
        Returns:
            Transformed targets DataFrame
        """
        if log_transform is None:
            log_transform = self.config.log_target
        
        if log_transform:
            self.logger.info("Applying log transformation to targets")
            targets_df = np.log(targets_df)
        
        self.processed_targets = targets_df.copy()
        return targets_df
    
    def normalize_features(
        self, 
        features_df: pd.DataFrame, 
        scaler_type: str = 'standard',
        fit: bool = True
    ) -> pd.DataFrame:
        """
        Normalize feature data.
        
        Args:
            features_df: Features DataFrame
            scaler_type: Type of scaler ('standard', 'minmax', 'robust')
            fit: Whether to fit the scaler (True for training, False for testing)
            
        Returns:
            Normalized features DataFrame
        """
        self.logger.info(f"Normalizing features using {scaler_type} scaler")
        
        # Initialize scaler if not exists
        if self.scaler is None:
            if scaler_type == 'standard':
                self.scaler = StandardScaler()
            elif scaler_type == 'minmax':
                self.scaler = MinMaxScaler()
            elif scaler_type == 'robust':
                self.scaler = RobustScaler()
            else:
                raise ValueError(f"Unknown scaler type: {scaler_type}")
        
        # Apply scaling
        if fit:
            scaled_features = self.scaler.fit_transform(features_df)
        else:
            scaled_features = self.scaler.transform(features_df)
        
        # Convert back to DataFrame
        normalized_df = pd.DataFrame(
            scaled_features,
            columns=features_df.columns,
            index=features_df.index
        )
        
        self.processed_features = normalized_df.copy()
        return normalized_df
    
    def apply_pca(
        self, 
        features_df: pd.DataFrame, 
        n_components: Optional[int] = None,
        fit: bool = True
    ) -> pd.DataFrame:
        """
        Apply PCA dimensionality reduction.
        
        Args:
            features_df: Features DataFrame
            n_components: Number of components. If None, uses config.
            fit: Whether to fit the PCA (True for training, False for testing)
            
        Returns:
            PCA-transformed features DataFrame
        """
        if n_components is None:
            n_components = self.config.pca_components
        
        if not self.config.apply_pca:
            self.logger.info("PCA not enabled in config, returning original features")
            return features_df
        
        self.logger.info(f"Applying PCA with {n_components} components")
        
        # Initialize PCA if not exists
        if self.pca is None:
            self.pca = PCA(n_components=n_components)
        
        # Apply PCA
        if fit:
            pca_features = self.pca.fit_transform(features_df)
            explained_variance = self.pca.explained_variance_ratio_
            self.logger.info(f"Explained variance: {explained_variance.sum():.3f}")
        else:
            pca_features = self.pca.transform(features_df)
        
        # Convert to DataFrame
        pca_df = pd.DataFrame(
            pca_features,
            columns=[f'PC_{i+1}' for i in range(pca_features.shape[1])],
            index=features_df.index
        )
        
        return pca_df
    
    def engineer_features(
        self, 
        features_df: pd.DataFrame,
        add_statistical_features: bool = True,
        add_derivative_features: bool = False
    ) -> pd.DataFrame:
        """
        Engineer additional features from spectral data.
        
        Args:
            features_df: Features DataFrame
            add_statistical_features: Whether to add statistical features
            add_derivative_features: Whether to add derivative features
            
        Returns:
            DataFrame with engineered features
        """
        self.logger.info("Engineering additional features")
        
        engineered_df = features_df.copy()
        
        # Add statistical features
        if add_statistical_features:
            engineered_df['signal_mean'] = features_df.mean(axis=1)
            engineered_df['signal_std'] = features_df.std(axis=1)
            engineered_df['signal_sum'] = features_df.sum(axis=1)
            engineered_df['signal_max'] = features_df.max(axis=1)
            engineered_df['signal_min'] = features_df.min(axis=1)
            engineered_df['signal_range'] = features_df.max(axis=1) - features_df.min(axis=1)
            
            # Add skewness and kurtosis if scipy is available
            try:
                from scipy.stats import skew, kurtosis
                engineered_df['signal_skewness'] = features_df.apply(skew, axis=1)
                engineered_df['signal_kurtosis'] = features_df.apply(kurtosis, axis=1)
            except ImportError:
                self.logger.warning("scipy not available, skipping skewness and kurtosis")
        
        # Add derivative features
        if add_derivative_features:
            # First derivative
            derivative_1 = np.diff(features_df.values, axis=1)
            derivative_1_df = pd.DataFrame(
                derivative_1,
                columns=[f'd1_{col}' for col in features_df.columns[:-1]],
                index=features_df.index
            )
            engineered_df = pd.concat([engineered_df, derivative_1_df], axis=1)
            
            # Second derivative
            derivative_2 = np.diff(derivative_1, axis=1)
            derivative_2_df = pd.DataFrame(
                derivative_2,
                columns=[f'd2_{col}' for col in features_df.columns[:-2]],
                index=features_df.index
            )
            engineered_df = pd.concat([engineered_df, derivative_2_df], axis=1)
        
        self.logger.info(f"Engineered features shape: {engineered_df.shape}")
        return engineered_df
    
    def preprocess_pipeline(
        self,
        features_df: pd.DataFrame,
        targets_df: pd.DataFrame,
        clean_data: bool = True,
        normalize_features: bool = True,
        apply_pca: Optional[bool] = None,
        engineer_features: bool = False,
        log_targets: Optional[bool] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Complete preprocessing pipeline.
        
        Args:
            features_df: Features DataFrame
            targets_df: Targets DataFrame
            clean_data: Whether to clean the data
            normalize_features: Whether to normalize features
            apply_pca: Whether to apply PCA
            engineer_features: Whether to engineer features
            log_targets: Whether to log-transform targets
            
        Returns:
            Tuple of preprocessed (features_df, targets_df)
        """
        self.logger.info("Starting preprocessing pipeline")
        
        # Clean data
        if clean_data:
            features_df, targets_df = self.clean_data(features_df, targets_df)
        
        # Engineer features
        if engineer_features:
            features_df = self.engineer_features(features_df)
        
        # Normalize features
        if normalize_features:
            features_df = self.normalize_features(features_df, fit=True)
        
        # Apply PCA
        if apply_pca is None:
            apply_pca = self.config.apply_pca
        
        if apply_pca:
            features_df = self.apply_pca(features_df, fit=True)
        
        # Transform targets
        targets_df = self.apply_target_transformation(targets_df, log_targets)
        
        self.logger.info("Preprocessing pipeline completed")
        return features_df, targets_df
    
    def transform_test_data(
        self,
        features_df: pd.DataFrame,
        targets_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Transform test data using fitted preprocessors.
        
        Args:
            features_df: Test features DataFrame
            targets_df: Test targets DataFrame
            
        Returns:
            Tuple of transformed (features_df, targets_df)
        """
        self.logger.info("Transforming test data")
        
        # Handle missing values
        if self.imputer is not None:
            features_df = self._handle_missing_values(features_df)
        
        # Normalize features
        if self.scaler is not None:
            features_df = self.normalize_features(features_df, fit=False)
        
        # Apply PCA
        if self.pca is not None:
            features_df = self.apply_pca(features_df, fit=False)
        
        # Transform targets
        targets_df = self.apply_target_transformation(targets_df)
        
        return features_df, targets_df
    
    def get_preprocessing_info(self) -> Dict[str, Any]:
        """
        Get information about the preprocessing steps applied.
        
        Returns:
            Dictionary containing preprocessing information
        """
        info = {
            'scaler_type': type(self.scaler).__name__ if self.scaler else None,
            'pca_components': self.pca.n_components_ if self.pca else None,
            'pca_explained_variance': self.pca.explained_variance_ratio_.sum() if self.pca else None,
            'imputer_strategy': self.imputer.strategy if self.imputer else None,
            'features_shape': self.processed_features.shape if self.processed_features is not None else None,
            'targets_shape': self.processed_targets.shape if self.processed_targets is not None else None
        }
        
        return info 