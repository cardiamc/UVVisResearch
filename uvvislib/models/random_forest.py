"""
Random Forest model for UV-Vis analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Union, List
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor as SklearnRandomForestRegressor

from .base import BaseModel
from ..utils.config import Config


class RandomForestRegressor(BaseModel):
    """
    Random Forest regressor for UV-Vis analysis.
    """
    
    def __init__(
        self, 
        config: Config,
        n_estimators: int = 100,
        criterion: str = 'squared_error',
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_weight_fraction_leaf: float = 0.0,
        max_features: Union[str, int, float] = 'sqrt',
        max_leaf_nodes: Optional[int] = None,
        min_impurity_decrease: float = 0.0,
        bootstrap: bool = True,
        oob_score: bool = False,
        n_jobs: Optional[int] = None,
        random_state: Optional[int] = None,
        verbose: int = 0,
        warm_start: bool = False,
        ccp_alpha: float = 0.0,
        max_samples: Optional[Union[int, float]] = None
    ):
        """
        Initialize Random Forest regressor.
        
        Args:
            config: Configuration object
            n_estimators: Number of trees in the forest
            criterion: Function to measure quality of split
            max_depth: Maximum depth of trees
            min_samples_split: Minimum samples required to split
            min_samples_leaf: Minimum samples required at leaf node
            min_weight_fraction_leaf: Minimum weighted fraction at leaf node
            max_features: Number of features to consider for best split
            max_leaf_nodes: Maximum number of leaf nodes
            min_impurity_decrease: Minimum impurity decrease for split
            bootstrap: Whether to use bootstrap samples
            oob_score: Whether to use out-of-bag samples for validation
            n_jobs: Number of jobs for parallel processing
            random_state: Random state for reproducibility
            verbose: Verbosity level
            warm_start: Whether to reuse solution from previous fit
            ccp_alpha: Complexity parameter for pruning
            max_samples: Number of samples to draw for training
        """
        super().__init__(config, model_name="RandomForest")
        
        # Model parameters
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.warm_start = warm_start
        self.ccp_alpha = ccp_alpha
        self.max_samples = max_samples
        
        # Create sklearn model
        self.model = SklearnRandomForestRegressor(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            ccp_alpha=ccp_alpha,
            max_samples=max_samples
        )
        
        # Store parameters
        self.model_params = {
            'n_estimators': n_estimators,
            'criterion': criterion,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'min_weight_fraction_leaf': min_weight_fraction_leaf,
            'max_features': max_features,
            'max_leaf_nodes': max_leaf_nodes,
            'min_impurity_decrease': min_impurity_decrease,
            'bootstrap': bootstrap,
            'oob_score': oob_score,
            'n_jobs': n_jobs,
            'random_state': random_state,
            'verbose': verbose,
            'warm_start': warm_start,
            'ccp_alpha': ccp_alpha,
            'max_samples': max_samples
        }
        
        self.logger.info("Random Forest initialized")
    
    def fit(
        self, 
        X: Union[np.ndarray, pd.DataFrame], 
        y: Union[np.ndarray, pd.DataFrame],
        **kwargs
    ) -> 'RandomForestRegressor':
        """
        Fit the Random Forest model.
        
        Args:
            X: Training features
            y: Training targets
            **kwargs: Additional parameters passed to sklearn
            
        Returns:
            Self for method chaining
        """
        # Validate and preprocess data
        X, y = self.validate_data(X, y)
        
        # Update model info
        self._update_model_info(X, y)
        
        # Store feature and target column names if available
        if isinstance(X, pd.DataFrame):
            self.feature_columns = list(X.columns)
        if isinstance(y, pd.DataFrame):
            self.target_columns = list(y.columns)
        
        self.logger.info(f"Training Random Forest with {self.n_estimators} trees")
        
        # Fit the model
        self.model.fit(X, y, **kwargs)
        
        # Store feature importance
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = self.model.feature_importances_
        
        # Store training history
        self.training_history = {
            'n_estimators': self.n_estimators,
            'oob_score': getattr(self.model, 'oob_score_', None),
            'n_features': self.model.n_features_in_ if hasattr(self.model, 'n_features_in_') else X.shape[1]
        }
        
        self.is_fitted = True
        self.logger.info("Training completed")
        
        return self
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Features to predict on
            
        Returns:
            Predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Validate data
        X, _ = self.validate_data(X)
        
        # Make predictions
        predictions = self.model.predict(X)
        
        return predictions
    
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Predict class probabilities (if available).
        
        Args:
            X: Features to predict on
            
        Returns:
            Class probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        if not hasattr(self.model, 'predict_proba'):
            raise ValueError("This model does not support probability predictions")
        
        # Validate data
        X, _ = self.validate_data(X)
        
        # Make predictions
        probabilities = self.model.predict_proba(X)
        
        return probabilities
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """
        Get feature importance scores.
        
        Returns:
            Feature importance array
        """
        if not self.is_fitted:
            return None
        
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        
        return None
    
    def get_feature_importance_per_tree(self) -> Optional[np.ndarray]:
        """
        Get feature importance scores for each tree.
        
        Returns:
            Feature importance array for each tree
        """
        if not self.is_fitted:
            return None
        
        if hasattr(self.model, 'estimators_'):
            importances = []
            for tree in self.model.estimators_:
                if hasattr(tree, 'feature_importances_'):
                    importances.append(tree.feature_importances_)
            return np.array(importances)
        
        return None
    
    def get_oob_score(self) -> Optional[float]:
        """
        Get out-of-bag score if available.
        
        Returns:
            OOB score or None
        """
        if not self.is_fitted:
            return None
        
        return getattr(self.model, 'oob_score_', None)
    
    def get_tree_info(self) -> Dict[str, Any]:
        """
        Get information about the trees in the forest.
        
        Returns:
            Dictionary containing tree information
        """
        if not self.is_fitted or not hasattr(self.model, 'estimators_'):
            return {}
        
        trees = self.model.estimators_
        depths = [tree.get_depth() for tree in trees]
        n_leaves = [tree.get_n_leaves() for tree in trees]
        
        return {
            'n_trees': len(trees),
            'mean_depth': np.mean(depths),
            'std_depth': np.std(depths),
            'min_depth': np.min(depths),
            'max_depth': np.max(depths),
            'mean_leaves': np.mean(n_leaves),
            'std_leaves': np.std(n_leaves),
            'min_leaves': np.min(n_leaves),
            'max_leaves': np.max(n_leaves)
        }
    
    def _save_model_impl(self, filepath: Path) -> None:
        """Save the sklearn model."""
        if self.model is not None:
            joblib.dump(self.model, filepath)
    
    def _load_model_impl(self, filepath: Path) -> None:
        """Load the sklearn model."""
        if filepath.exists():
            self.model = joblib.load(filepath)
            self.is_fitted = True
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get detailed model summary."""
        summary = super().get_model_summary()
        
        if self.is_fitted and self.model is not None:
            tree_info = self.get_tree_info()
            summary.update({
                'n_estimators': self.n_estimators,
                'criterion': self.criterion,
                'max_depth': self.max_depth,
                'oob_score': self.get_oob_score(),
                'tree_info': tree_info,
                'feature_importance_available': self.feature_importance is not None
            })
        
        return summary
    
    def partial_fit(self, X: Union[np.ndarray, pd.DataFrame], 
                   y: Union[np.ndarray, pd.DataFrame]) -> 'RandomForestRegressor':
        """
        Incremental fit on a batch of samples.
        
        Args:
            X: Training features
            y: Training targets
            
        Returns:
            Self for method chaining
        """
        if not self.warm_start:
            raise ValueError("Partial fit requires warm_start=True")
        
        # Validate and preprocess data
        X, y = self.validate_data(X, y)
        
        # Update model info if not already set
        if not self.is_fitted:
            self._update_model_info(X, y)
        
        # Partial fit
        self.model.partial_fit(X, y)
        
        # Update feature importance
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = self.model.feature_importances_
        
        self.is_fitted = True
        
        return self 

RandomForestModel = RandomForestRegressor 