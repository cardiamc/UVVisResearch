"""
Data preprocessing utilities for UV-Vis analysis.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional, Dict, Any, Union
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
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

        # Coerce all non-numeric target columns to float.
        # Handles: pandas object dtype (old-style), pandas 3.x StringDtype, and
        # values with comma decimal separators or leading/trailing whitespace.
        for col in targets_df.columns:
            if not pd.api.types.is_numeric_dtype(targets_df[col]):
                targets_df[col] = pd.to_numeric(
                    targets_df[col].astype(str).str.strip().str.replace(',', '.', regex=False),
                    errors='coerce'
                )

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
    
    def smooth_spectra(
        self,
        features_df: pd.DataFrame,
        sigma: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Apply a 1-D Gaussian filter row-wise to spectral features.

        This is the canonical UV-Vis smoothing step (sigma=1.5 by default,
        matching the original training pipeline).

        Args:
            features_df: Spectral features DataFrame (rows = samples,
                columns = wavelengths).
            sigma: Standard deviation of the Gaussian kernel. If None,
                uses ``self.config.gaussian_sigma``.

        Returns:
            Smoothed features DataFrame with the same shape and columns.
        """
        if sigma is None:
            sigma = self.config.gaussian_sigma

        self.logger.info(f"Smoothing spectra with Gaussian filter (sigma={sigma})")

        smoothed = gaussian_filter1d(features_df.values, sigma=sigma, axis=1)
        return pd.DataFrame(
            smoothed,
            columns=features_df.columns,
            index=features_df.index,
        )

    def apply_savitzky_golay(
        self,
        features_df: pd.DataFrame,
        window_length: int = 11,
        polyorder: int = 3,
        deriv: int = 0,
        delta: float = 1.0,
    ) -> pd.DataFrame:
        """
        Apply Savitzky-Golay filter row-wise to spectral features.

        Suitable for NIR spectra: use ``deriv=0`` for smoothing, ``deriv=1``
        for the first derivative, or ``deriv=2`` for the second derivative.
        Derivatives remove additive and multiplicative baselines and can
        sharpen spectral features.

        Parameters
        ----------
        features_df : pd.DataFrame
            Spectral features (rows = samples, columns = wavenumbers /
            wavelengths).
        window_length : int
            Length of the filter window. Must be a positive odd integer
            greater than ``polyorder``. Automatically bumped to the next odd
            integer if an even value is supplied.
        polyorder : int
            Polynomial order used for the least-squares fit within each
            window.
        deriv : int
            Derivative order. 0 = smoothing only, 1 = first derivative,
            2 = second derivative.
        delta : float
            Spacing between consecutive samples on the spectral axis
            (e.g. 16.0 cm⁻¹ for a downsampled NIR grid). Affects the
            physical scale of derivatives but not ML performance when
            followed by normalisation.

        Returns
        -------
        pd.DataFrame
            Filtered (or differentiated) features with the same shape and
            column names as the input.
        """
        if window_length % 2 == 0:
            window_length += 1
        if window_length <= polyorder:
            window_length = polyorder + 2 if (polyorder + 2) % 2 == 1 else polyorder + 3

        self.logger.info(
            f"Applying Savitzky-Golay filter "
            f"(window={window_length}, poly={polyorder}, deriv={deriv})"
        )
        sg_data = savgol_filter(
            features_df.values.astype(float),
            window_length=window_length,
            polyorder=polyorder,
            deriv=deriv,
            delta=delta,
            axis=1,
            mode="interp",
        )
        return pd.DataFrame(sg_data, columns=features_df.columns, index=features_df.index)

    def apply_snv(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply Standard Normal Variate (SNV) normalisation row-wise.

        Each spectrum is centred by its mean and scaled by its standard
        deviation. SNV is the standard first step for NIR reflectance /
        transmittance data because it removes multiplicative scatter
        differences between samples without requiring a reference.

        Parameters
        ----------
        features_df : pd.DataFrame
            Spectral features DataFrame (rows = samples, columns = channels).

        Returns
        -------
        pd.DataFrame
            SNV-normalised DataFrame with the same shape and column names.
        """
        self.logger.info("Applying SNV normalisation")
        X = features_df.values.astype(float)
        row_mean = X.mean(axis=1, keepdims=True)
        row_std = X.std(axis=1, keepdims=True)
        snv = (X - row_mean) / np.where(row_std == 0, 1.0, row_std)
        return pd.DataFrame(snv, columns=features_df.columns, index=features_df.index)

    def apply_water_subtraction(
        self,
        features_df: pd.DataFrame,
        water_reference: np.ndarray,
    ) -> pd.DataFrame:
        """
        Subtract a water background spectrum from NIR features.

        Removes the dominant water absorption baseline so the residual signal
        reflects dissolved analyte contributions. Supports both a single
        reference (broadcast to all samples) and a per-sample reference
        (week-matched correction).

        Parameters
        ----------
        features_df : pd.DataFrame
            NIR absorbance spectra (n_samples × n_wavenumbers).
        water_reference : np.ndarray, shape (n_wavenumbers,) or (n_samples, n_wavenumbers)
            Water background to subtract. A 1-D array is broadcast across all
            rows; a 2-D array is subtracted row-wise (week-matched correction).

        Returns
        -------
        pd.DataFrame
            Background-corrected spectra with the same index and columns.
        """
        self.logger.info("Applying water background subtraction")
        ref = np.asarray(water_reference, dtype=float)
        corrected = features_df.values.astype(float) - ref
        return pd.DataFrame(corrected, columns=features_df.columns, index=features_df.index)

    def apply_msc(
        self,
        features_df: pd.DataFrame,
        reference: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """
        Apply Multiplicative Scatter Correction (MSC) to spectral features.

        For each spectrum, a linear model is fitted against a reference
        spectrum and the estimated additive offset and slope are used to
        correct multiplicative and additive scatter effects.

        Parameters
        ----------
        features_df : pd.DataFrame
            Spectral features DataFrame (rows = samples).
        reference : np.ndarray, optional
            1-D reference spectrum (length = n_features). Defaults to the
            column-wise mean of ``features_df``.

        Returns
        -------
        pd.DataFrame
            MSC-corrected DataFrame with the same shape and column names.
        """
        X = features_df.values.astype(float)
        if reference is None:
            reference = X.mean(axis=0)

        self.logger.info("Applying MSC correction")
        corrected = np.empty_like(X)
        for i, spectrum in enumerate(X):
            coeffs = np.polyfit(reference, spectrum, 1)
            a, b = coeffs[0], coeffs[1]
            corrected[i] = (spectrum - b) / (a if abs(a) > 1e-10 else 1.0)

        return pd.DataFrame(corrected, columns=features_df.columns, index=features_df.index)

    def normalize_features(
        self,
        features_df: pd.DataFrame,
        scaler_type: str = 'minmax',
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
        smooth_spectra: Optional[bool] = None,
        normalize_features: bool = True,
        scaler_type: str = 'minmax',
        apply_pca: Optional[bool] = None,
        engineer_features: bool = False,
        log_targets: Optional[bool] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Complete preprocessing pipeline.

        The default order is: clean → Gaussian smoothing (sigma=1.5) →
        engineered features → MinMax scaling → optional PCA → optional log
        target transform. This matches the canonical training pipeline.

        Args:
            features_df: Features DataFrame
            targets_df: Targets DataFrame
            clean_data: Whether to clean the data
            smooth_spectra: Whether to apply Gaussian smoothing. If None,
                uses ``self.config.apply_smoothing``.
            normalize_features: Whether to normalize features
            scaler_type: Scaler to use when ``normalize_features`` is True
                ('minmax', 'standard', or 'robust'). Defaults to 'minmax'.
            apply_pca: Whether to apply PCA
            engineer_features: Whether to engineer features
            log_targets: Whether to log-transform targets

        Returns:
            Tuple of preprocessed (features_df, targets_df)
        """
        self.logger.info("Starting preprocessing pipeline")

        if clean_data:
            features_df, targets_df = self.clean_data(features_df, targets_df)

        if smooth_spectra is None:
            smooth_spectra = self.config.apply_smoothing
        if smooth_spectra:
            features_df = self.smooth_spectra(features_df)

        if engineer_features:
            features_df = self.engineer_features(features_df)

        if normalize_features:
            features_df = self.normalize_features(
                features_df, scaler_type=scaler_type, fit=True
            )

        if apply_pca is None:
            apply_pca = self.config.apply_pca
        if apply_pca:
            features_df = self.apply_pca(features_df, fit=True)

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

        # Apply Gaussian smoothing (matches the training pipeline; stateless)
        if self.config.apply_smoothing:
            features_df = self.smooth_spectra(features_df)

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