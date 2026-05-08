"""
Cross-validation utilities for UV-Vis spectral analysis.

This module provides various cross-validation strategies including
double k-fold cross-validation, stratified k-fold, and nested cross-validation
with hyperparameter tuning.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
from sklearn.base import clone
from sklearn.model_selection import (
    KFold, StratifiedKFold, RandomizedSearchCV, GridSearchCV,
    ParameterGrid, ParameterSampler, cross_val_score
)
from sklearn.metrics import make_scorer
import logging

from ..evaluation.metrics import (
    r2_score_single, neg_rmse_exp, rmse_exp, compute_evaluation
)


class DoubleKFoldCV:
    """
    Double K-Fold Cross-Validation with hyperparameter tuning.
    
    This class implements a nested cross-validation strategy where:
    - Outer loop: Evaluates model performance on test sets
    - Inner loop: Tunes hyperparameters on validation sets
    """
    
    def __init__(self, 
                 outer_splits: int = 5,
                 inner_splits: int = 4,
                 random_state: int = 42,
                 stratify: bool = True,
                 shuffle: bool = True):
        """
        Initialize Double K-Fold Cross-Validation.
        
        Args:
            outer_splits: Number of outer CV splits
            inner_splits: Number of inner CV splits
            random_state: Random seed for reproducibility
            stratify: Whether to use stratified sampling
            shuffle: Whether to shuffle data before splitting
        """
        self.outer_splits = outer_splits
        self.inner_splits = inner_splits
        self.random_state = random_state
        self.stratify = stratify
        self.shuffle = shuffle
        
        # Initialize CV splitters
        if stratify:
            self.outer_cv = StratifiedKFold(
                n_splits=outer_splits, 
                shuffle=shuffle, 
                random_state=random_state
            )
            self.inner_cv = StratifiedKFold(
                n_splits=inner_splits, 
                shuffle=shuffle, 
                random_state=random_state
            )
        else:
            self.outer_cv = KFold(
                n_splits=outer_splits, 
                shuffle=shuffle, 
                random_state=random_state
            )
            self.inner_cv = KFold(
                n_splits=inner_splits, 
                shuffle=shuffle, 
                random_state=random_state
            )
    
    def fit(self, 
            X: np.ndarray, 
            y: np.ndarray,
            model,
            param_grid: Dict[str, List],
            scoring: Optional[str] = None,
            n_iter: int = 100,
            log_target: bool = False) -> Dict[str, Any]:
        """
        Perform double k-fold cross-validation with hyperparameter tuning.
        
        Args:
            X: Feature matrix
            y: Target values
            model: Model to evaluate
            param_grid: Hyperparameter grid for tuning
            scoring: Scoring metric for optimization
            n_iter: Number of parameter combinations to try
            log_target: Whether targets are log-transformed
            
        Returns:
            Dictionary containing CV results
        """
        results = {
            'test_scores': [],
            'train_scores': [],
            'val_scores': [],
            'best_params': [],
            'predictions': [],
            'true_values': [],
            'fold_indices': []
        }
        
        # Create stratification labels if needed
        if self.stratify:
            stratify_labels = self._create_stratify_labels(y)
        else:
            stratify_labels = None
        
        # Outer CV loop
        for fold, (train_val_idx, test_idx) in enumerate(
            self.outer_cv.split(X, stratify_labels)
        ):
            logging.info(f"Outer fold {fold + 1}/{self.outer_splits}")
            
            X_train_val, X_test = X[train_val_idx], X[test_idx]
            y_train_val, y_test = y[train_val_idx], y[test_idx]
            
            # Inner CV for hyperparameter tuning
            best_params, inner_scores = self._inner_cv_tuning(
                X_train_val, y_train_val, model, param_grid, 
                scoring, n_iter, log_target, fold
            )
            
            # Train final model with best parameters
            final_model = self._train_final_model(
                X_train_val, y_train_val, model, best_params
            )
            
            # Evaluate on test set
            test_pred = final_model.predict(X_test)
            # Normalize to 2D so vstack and save_predictions work for single target
            if test_pred.ndim == 1:
                test_pred = test_pred.reshape(-1, 1)
            if y_test.ndim == 1:
                y_test = y_test.reshape(-1, 1)
            test_score = compute_evaluation(y_test, test_pred, log_target)

            # Store results
            results['test_scores'].append(test_score)
            results['train_scores'].append(inner_scores['train'])
            results['val_scores'].append(inner_scores['val'])
            results['best_params'].append(best_params)
            results['predictions'].append(test_pred)
            results['true_values'].append(y_test)
            results['fold_indices'].append({
                'train_val': train_val_idx,
                'test': test_idx
            })
            
            logging.info(f"Fold {fold + 1} - Test RMSE: {np.mean(test_score['rmse']):.4f}")
        
        return results
    
    def _inner_cv_tuning(self, 
                        X: np.ndarray, 
                        y: np.ndarray,
                        model,
                        param_grid: Dict[str, List],
                        scoring: str,
                        n_iter: int,
                        log_target: bool,
                        fold: int) -> Tuple[Dict, Dict]:
        """
        Perform inner CV for hyperparameter tuning.
        
        Args:
            X: Training/validation features
            y: Training/validation targets
            model: Model to tune
            param_grid: Hyperparameter grid
            scoring: Scoring metric
            n_iter: Number of parameter combinations
            log_target: Whether targets are log-transformed
            fold: Current fold number
            
        Returns:
            Tuple of (best_params, inner_scores)
        """
        # Create scoring function
        if log_target:
            scoring_dict = {
                'neg_mean_squared_error': 'neg_mean_squared_error',
                'RMSE': make_scorer(neg_rmse_exp),
                'R2': make_scorer(r2_score_single)
            }
            refit = 'neg_mean_squared_error'
        else:
            scoring_dict = 'neg_mean_squared_error'
            refit = True
        
        # Create stratification labels for inner CV
        if self.stratify:
            stratify_labels = self._create_stratify_labels(y)
        else:
            stratify_labels = None
        
        # Perform randomized search
        search = RandomizedSearchCV(
            model, param_grid, n_iter=n_iter, cv=self.inner_cv,
            scoring=scoring_dict, refit=refit, n_jobs=-1,
            random_state=self.random_state, return_train_score=True
        )
        
        search.fit(X, y)

        # cv_results_ key names differ: dict scoring uses metric names, string scoring uses generic names
        if log_target:
            rank_key = 'rank_test_neg_mean_squared_error'
            rmse_train_key = 'mean_train_neg_mean_squared_error'
            rmse_val_key = 'mean_test_neg_mean_squared_error'
        else:
            rank_key = 'rank_test_score'
            rmse_train_key = 'mean_train_score'
            rmse_val_key = 'mean_test_score'

        best_mask = search.cv_results_[rank_key] == 1
        inner_scores = {
            'train': {
                'RMSE': -search.cv_results_[rmse_train_key][best_mask][0],
                'R2': search.cv_results_['mean_train_R2'][best_mask][0] if log_target else None
            },
            'val': {
                'RMSE': -search.cv_results_[rmse_val_key][best_mask][0],
                'R2': search.cv_results_['mean_test_R2'][best_mask][0] if log_target else None
            }
        }
        
        return search.best_params_, inner_scores
    
    def _train_final_model(self, 
                          X: np.ndarray, 
                          y: np.ndarray,
                          model,
                          best_params: Dict) -> Any:
        """
        Train final model with best parameters.
        
        Args:
            X: Training features
            y: Training targets
            model: Model class
            best_params: Best hyperparameters
            
        Returns:
            Trained model
        """
        final_model = clone(model)
        final_model.set_params(**best_params)
        final_model.fit(X, y)
        return final_model
    
    def _create_stratify_labels(self, y: np.ndarray) -> np.ndarray:
        """
        Create stratification labels for regression.
        
        Args:
            y: Target values
            
        Returns:
            Stratification labels
        """
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        
        # Use the first target for stratification
        return pd.cut(
            y[:, 0], 
            bins=10, 
            labels=False, 
            include_lowest=True
        )
    
    def get_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate summary statistics from CV results.
        
        Args:
            results: Results from fit method
            
        Returns:
            Summary statistics
        """
        summary = {}
        
        # Aggregate test scores
        test_rmse = np.array([score['RMSE'] for score in results['test_scores']])
        test_r2 = np.array([score['R2'] for score in results['test_scores']])
        test_mape = np.array([score['MAPE'] for score in results['test_scores']])
        
        summary['test_rmse_mean'] = np.mean(test_rmse, axis=0)
        summary['test_rmse_std'] = np.std(test_rmse, axis=0)
        summary['test_r2_mean'] = np.mean(test_r2, axis=0)
        summary['test_r2_std'] = np.std(test_r2, axis=0)
        summary['test_mape_mean'] = np.mean(test_mape, axis=0)
        summary['test_mape_std'] = np.std(test_mape, axis=0)
        
        return summary


class NestedCrossValidation:
    """
    Nested Cross-Validation for unbiased model evaluation.
    
    This class provides a simpler interface for nested CV
    when you don't need the full double k-fold functionality.
    """
    
    def __init__(self, 
                 outer_cv: int = 5,
                 inner_cv: int = 3,
                 random_state: int = 42):
        """
        Initialize Nested Cross-Validation.
        
        Args:
            outer_cv: Number of outer CV folds
            inner_cv: Number of inner CV folds
            random_state: Random seed
        """
        self.outer_cv = outer_cv
        self.inner_cv = inner_cv
        self.random_state = random_state
    
    def cross_val_score(self, 
                       estimator,
                       X: np.ndarray, 
                       y: np.ndarray,
                       param_grid: Dict[str, List],
                       scoring: str = 'neg_mean_squared_error',
                       n_iter: int = 50) -> np.ndarray:
        """
        Perform nested cross-validation.
        
        Args:
            estimator: Model to evaluate
            X: Feature matrix
            y: Target values
            param_grid: Hyperparameter grid
            scoring: Scoring metric
            n_iter: Number of parameter combinations
            
        Returns:
            Array of scores from outer CV
        """
        outer_scores = []
        
        # Outer CV
        outer_cv = KFold(
            n_splits=self.outer_cv, 
            shuffle=True, 
            random_state=self.random_state
        )
        
        for train_idx, test_idx in outer_cv.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Inner CV for hyperparameter tuning
            inner_cv = KFold(
                n_splits=self.inner_cv, 
                shuffle=True, 
                random_state=self.random_state
            )
            
            search = RandomizedSearchCV(
                estimator, param_grid, n_iter=n_iter, cv=inner_cv,
                scoring=scoring, n_jobs=-1, random_state=self.random_state
            )
            
            search.fit(X_train, y_train)
            
            # Evaluate on test set
            score = search.score(X_test, y_test)
            outer_scores.append(score)
        
        return np.array(outer_scores)


def create_cv_scorers(log_target: bool = False) -> Dict[str, Any]:
    """
    Create scoring functions for cross-validation.
    
    Args:
        log_target: Whether targets are log-transformed
        
    Returns:
        Dictionary of scoring functions
    """
    if log_target:
        return {
            'neg_mean_squared_error': 'neg_mean_squared_error',
            'RMSE': make_scorer(neg_rmse_exp),
            'R2': make_scorer(r2_score_single)
        }
    else:
        return {
            'neg_mean_squared_error': 'neg_mean_squared_error',
            'R2': 'r2'
        }


def stratified_regression_cv(X: np.ndarray, 
                           y: np.ndarray,
                           n_splits: int = 5,
                           random_state: int = 42) -> StratifiedKFold:
    """
    Create stratified k-fold for regression.
    
    Args:
        X: Feature matrix
        y: Target values
        n_splits: Number of splits
        random_state: Random seed
        
    Returns:
        StratifiedKFold object
    """
    if len(y.shape) == 1:
        y = y.reshape(-1, 1)
    
    # Create stratification labels based on first target
    stratify_labels = pd.cut(
        y[:, 0], 
        bins=10, 
        labels=False, 
        include_lowest=True
    )
    
    return StratifiedKFold(
        n_splits=n_splits, 
        shuffle=True, 
        random_state=random_state
    ) 