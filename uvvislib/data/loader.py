"""
Data loading utilities for UV-Vis analysis.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional, Dict, Any
from pathlib import Path
import logging

from ..utils.config import Config


class DataLoader:
    """
    Data loader for UV-Vis and chemical analysis data.
    
    Handles loading, combining, and basic preprocessing of spectral and chemical data.
    """
    
    def __init__(self, config: Config):
        """
        Initialize data loader with configuration.
        
        Args:
            config: Configuration object containing data paths and parameters
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Data storage
        self.uv_vis_data: Optional[pd.DataFrame] = None
        self.chemical_data: Optional[pd.DataFrame] = None
        self.combined_data: Optional[pd.DataFrame] = None
        
        # Feature and target columns
        self.feature_columns: List[str] = []
        self.target_columns: List[str] = []
        self.categorical_columns: List[str] = []
    
    def load_uv_vis_data(self, filepath: Optional[str] = None) -> pd.DataFrame:
        """
        Load UV-Vis spectral data.
        
        Args:
            filepath: Path to UV-Vis data file. If None, uses config default.
            
        Returns:
            DataFrame containing UV-Vis spectral data
        """
        if filepath is None:
            filepath = Path(self.config.data_path) / self.config.uv_vis_data_file
        
        self.logger.info(f"Loading UV-Vis data from: {filepath}")
        
        try:
            self.uv_vis_data = pd.read_csv(filepath, sep=";", decimal=",")
            self.logger.info(f"UV-Vis data loaded: {self.uv_vis_data.shape}")
            return self.uv_vis_data
        except FileNotFoundError:
            self.logger.error(f"UV-Vis data file not found: {filepath}")
            raise
        except Exception as e:
            self.logger.error(f"Error loading UV-Vis data: {str(e)}")
            raise
    
    def load_chemical_data(self, filepath: Optional[str] = None) -> pd.DataFrame:
        """
        Load chemical analysis data.
        
        Args:
            filepath: Path to chemical data file. If None, uses config default.
            
        Returns:
            DataFrame containing chemical analysis data
        """
        if filepath is None:
            filepath = Path(self.config.data_path) / self.config.chemical_data_file
        
        self.logger.info(f"Loading chemical data from: {filepath}")
        
        try:
            self.chemical_data = pd.read_csv(filepath, sep=";", decimal=",")
            self.logger.info(f"Chemical data loaded: {self.chemical_data.shape}")
            return self.chemical_data
        except FileNotFoundError:
            self.logger.error(f"Chemical data file not found: {filepath}")
            raise
        except Exception as e:
            self.logger.error(f"Error loading chemical data: {str(e)}")
            raise
    
    def combine_data(
        self, 
        join_on: List[str] = ['Week', 'Gathering_Point'],
        uv_vis_data: Optional[pd.DataFrame] = None,
        chemical_data: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Combine UV-Vis and chemical data based on common identifiers.
        
        Args:
            join_on: Column names to use for joining the datasets
            uv_vis_data: UV-Vis data DataFrame. If None, uses loaded data.
            chemical_data: Chemical data DataFrame. If None, uses loaded data.
            
        Returns:
            Combined DataFrame
        """
        if uv_vis_data is None:
            uv_vis_data = self.uv_vis_data
        if chemical_data is None:
            chemical_data = self.chemical_data
        
        if uv_vis_data is None or chemical_data is None:
            raise ValueError("Both UV-Vis and chemical data must be loaded first")
        
        self.logger.info("Combining UV-Vis and chemical data")
        
        # Set index for joining
        uv_vis_indexed = uv_vis_data.set_index(join_on)
        chemical_indexed = chemical_data.set_index(join_on)
        
        # Join datasets
        self.combined_data = chemical_data.join(uv_vis_indexed, on=join_on)
        
        self.logger.info(f"Combined data shape: {self.combined_data.shape}")
        return self.combined_data
    
    def extract_features_and_targets(
        self, 
        data: Optional[pd.DataFrame] = None,
        feature_columns: Optional[List[str]] = None,
        target_columns: Optional[List[str]] = None,
        categorical_columns: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Extract feature and target columns from the dataset.
        
        Args:
            data: DataFrame to extract from. If None, uses combined data.
            feature_columns: List of feature column names. If None, uses wavelength features.
            target_columns: List of target column names. If None, uses config targets.
            categorical_columns: List of categorical column names.
            
        Returns:
            Tuple of (features_df, targets_df)
        """
        if data is None:
            data = self.combined_data
        
        if data is None:
            raise ValueError("No data available. Load and combine data first.")
        
        # Use default columns if not specified
        if feature_columns is None:
            feature_columns = self.config.feature_wavelengths
        
        if target_columns is None:
            target_columns = self.config.target_variables
        
        if categorical_columns is None:
            categorical_columns = self.config.categorical_features
        
        # Check if columns exist in data
        missing_features = [col for col in feature_columns if col not in data.columns]
        missing_targets = [col for col in target_columns if col not in data.columns]
        
        if missing_features:
            self.logger.warning(f"Missing feature columns: {missing_features}")
        
        if missing_targets:
            self.logger.warning(f"Missing target columns: {missing_targets}")
        
        # Extract available columns
        available_features = [col for col in feature_columns if col in data.columns]
        available_targets = [col for col in target_columns if col in data.columns]
        
        # Extract features and targets
        features_df = data[available_features].copy()
        targets_df = data[available_targets].copy()
        
        # Store column information
        self.feature_columns = available_features
        self.target_columns = available_targets
        self.categorical_columns = [col for col in categorical_columns if col in data.columns]
        
        self.logger.info(f"Extracted {len(available_features)} features and {len(available_targets)} targets")
        
        return features_df, targets_df
    
    def load_and_prepare_data(
        self,
        uv_vis_file: Optional[str] = None,
        chemical_file: Optional[str] = None,
        join_on: List[str] = ['Week', 'Gathering_Point']
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Complete data loading and preparation pipeline.
        
        Args:
            uv_vis_file: Path to UV-Vis data file
            chemical_file: Path to chemical data file
            join_on: Column names for joining datasets
            
        Returns:
            Tuple of (features_df, targets_df)
        """
        # Load data
        self.load_uv_vis_data(uv_vis_file)
        self.load_chemical_data(chemical_file)
        
        # Combine data
        self.combine_data(join_on)
        
        # Extract features and targets
        features_df, targets_df = self.extract_features_and_targets()
        
        return features_df, targets_df
    
    def get_data_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded data.
        
        Returns:
            Dictionary containing data information
        """
        info = {
            'uv_vis_data_shape': self.uv_vis_data.shape if self.uv_vis_data is not None else None,
            'chemical_data_shape': self.chemical_data.shape if self.chemical_data is not None else None,
            'combined_data_shape': self.combined_data.shape if self.combined_data is not None else None,
            'feature_columns_count': len(self.feature_columns),
            'target_columns_count': len(self.target_columns),
            'categorical_columns_count': len(self.categorical_columns),
            'feature_columns': self.feature_columns,
            'target_columns': self.target_columns,
            'categorical_columns': self.categorical_columns
        }
        
        return info
    
    def save_combined_data(self, filepath: str) -> None:
        """
        Save the combined dataset to a file.
        
        Args:
            filepath: Path where to save the combined data
        """
        if self.combined_data is None:
            raise ValueError("No combined data available. Load and combine data first.")
        
        self.combined_data.to_csv(filepath, index=False)
        self.logger.info(f"Combined data saved to: {filepath}")
    
    def load_combined_data(self, filepath: str) -> pd.DataFrame:
        """
        Load previously combined data from file.
        
        Args:
            filepath: Path to the combined data file
            
        Returns:
            Combined DataFrame
        """
        self.logger.info(f"Loading combined data from: {filepath}")
        
        try:
            self.combined_data = pd.read_csv(filepath)
            self.logger.info(f"Combined data loaded: {self.combined_data.shape}")
            return self.combined_data
        except FileNotFoundError:
            self.logger.error(f"Combined data file not found: {filepath}")
            raise
        except Exception as e:
            self.logger.error(f"Error loading combined data: {str(e)}")
            raise 