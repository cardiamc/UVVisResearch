"""
Evaluation metrics for UV-Vis spectral analysis.

This module provides various evaluation metrics for regression tasks,
including R2 score, MAPE, RMSE, and other statistical measures.
"""

import numpy as np
import scipy.stats
from typing import Union, Tuple, Optional
from sklearn.metrics import mean_squared_error, r2_score as sklearn_r2_score


def r2_score(x: np.ndarray, y: np.ndarray) -> Union[float, np.ndarray]:
    """
    Calculate R-squared (coefficient of determination) score.
    
    This function calculates the R-squared score using polynomial fitting.
    For multivariate data, it returns an array of R-squared values for each target.
    
    Args:
        x: Predicted values
        y: True values
        
    Returns:
        R-squared score(s). For multivariate data, returns array of scores.
        
    Example:
        >>> r2_score([1, 2, 3], [1.1, 1.9, 3.1])
        0.9999999999999999
    """
    dim = 1 if len(x.shape) == 1 else x.shape[1]
    
    if dim > 1:
        r2 = []
        for i in range(dim):
            try:
                r2_i = polyfit(x[:, i], y[:, i], 1)['determination']
            except:
                r2_i = float('-inf')
            if r2_i > 1:
                r2_i = float('-inf')
            r2.append(r2_i)
        return np.array(r2)
    else:
        try:
            r2 = polyfit(x, y, 1)['determination']
        except:
            r2 = float('-inf')
        if r2 > 1:
            r2 = float('-inf')
        return r2


def r2_score_single(x: np.ndarray, y: np.ndarray) -> float:
    """
    Calculate R-squared score for single target regression.
    
    Args:
        x: Predicted values (flattened)
        y: True values (flattened)
        
    Returns:
        R-squared score
        
    Example:
        >>> r2_score_single([1, 2, 3], [1.1, 1.9, 3.1])
        0.9999999999999999
    """
    r2 = polyfit(x.flatten(), y.flatten(), 1)['determination']
    if r2 > 1:
        r2 = float('-inf')
    return r2


def polyfit(x: np.ndarray, y: np.ndarray, degree: int) -> dict:
    """
    Perform polynomial fitting and return results.
    
    Args:
        x: Independent variable
        y: Dependent variable
        degree: Degree of polynomial
        
    Returns:
        Dictionary containing polynomial coefficients and R-squared value
        
    Example:
        >>> result = polyfit([1, 2, 3], [2, 4, 6], 1)
        >>> result['determination']
        1.0
    """
    results = {}
    
    coeffs = np.polyfit(x, y, degree)
    results['polynomial'] = coeffs.tolist()
    
    # R-squared calculation
    p = np.poly1d(coeffs)
    yhat = p(x)
    ybar = np.sum(y) / len(y)
    ssreg = np.sum((yhat - ybar) ** 2)
    sstot = np.sum((y - ybar) ** 2)
    results['determination'] = ssreg / sstot
    
    return results


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> Union[float, np.ndarray]:
    """
    Calculate Mean Absolute Percentage Error (MAPE).
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        MAPE value(s)
        
    Example:
        >>> mape([100, 200, 300], [110, 190, 310])
        0.03333333333333333
    """
    return (abs((np.array(y_true) - np.array(y_pred) + 1) / 
                (np.array(y_true) + 1))).mean(axis=0)


def rmse(y_true: np.ndarray, y_pred: np.ndarray, 
         squared: bool = False, multioutput: str = 'uniform_average') -> Union[float, np.ndarray]:
    """
    Calculate Root Mean Square Error (RMSE).
    
    Args:
        y_true: True values
        y_pred: Predicted values
        squared: If True, returns MSE, if False, returns RMSE
        multioutput: Defines aggregating of multiple output values
        
    Returns:
        RMSE value(s)
        
    Example:
        >>> rmse([1, 2, 3], [1.1, 1.9, 3.1])
        0.1414213562373095
    """
    return mean_squared_error(y_true, y_pred, squared=squared, multioutput=multioutput)


def rmse_exp(y_true: np.ndarray, y_pred: np.ndarray, 
             multioutput: str = 'uniform_average') -> Union[float, np.ndarray]:
    """
    Calculate RMSE after exponential transformation.
    
    Useful when working with log-transformed targets.
    
    Args:
        y_true: True values (log-transformed)
        y_pred: Predicted values (log-transformed)
        multioutput: Defines aggregating of multiple output values
        
    Returns:
        RMSE after exponential transformation
        
    Example:
        >>> rmse_exp([0, 0.693, 1.099], [0.1, 0.642, 1.131])
        0.1414213562373095
    """
    return mean_squared_error(np.exp(y_true), np.exp(y_pred), 
                            squared=False, multioutput=multioutput)


def neg_rmse_exp(y_true: np.ndarray, y_pred: np.ndarray, 
                 multioutput: str = 'uniform_average') -> Union[float, np.ndarray]:
    """
    Calculate negative RMSE after exponential transformation.
    
    Useful for optimization where we want to maximize (minimize negative).
    
    Args:
        y_true: True values (log-transformed)
        y_pred: Predicted values (log-transformed)
        multioutput: Defines aggregating of multiple output values
        
    Returns:
        Negative RMSE after exponential transformation
        
    Example:
        >>> neg_rmse_exp([0, 0.693, 1.099], [0.1, 0.642, 1.131])
        -0.1414213562373095
    """
    return -rmse_exp(y_true, y_pred, multioutput=multioutput)


def compute_evaluation(y_true: np.ndarray, y_pred: np.ndarray, 
                      log_target: bool = False) -> dict:
    """
    Compute comprehensive evaluation metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        log_target: Whether targets are log-transformed
        
    Returns:
        Dictionary containing all evaluation metrics
        
    Example:
        >>> metrics = compute_evaluation([1, 2, 3], [1.1, 1.9, 3.1])
        >>> metrics['RMSE']
        0.1414213562373095
    """
    evaluation_results = {}
    
    try:
        if log_target:
            evaluation_results["RMSE"] = rmse_exp(y_true, y_pred, multioutput='raw_values')
        else:
            evaluation_results["RMSE"] = rmse(y_true, y_pred, multioutput='raw_values')
    except:
        evaluation_results["RMSE"] = float('inf')
    
    evaluation_results["logRMSE"] = rmse(y_true, y_pred, multioutput='raw_values')
    evaluation_results["R2"] = r2_score(y_true, y_pred)
    evaluation_results["MAPE"] = mape(y_true, y_pred)
    
    return evaluation_results


def correlation_analysis(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Perform correlation analysis between true and predicted values.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Dictionary containing correlation statistics
        
    Example:
        >>> stats = correlation_analysis([1, 2, 3], [1.1, 1.9, 3.1])
        >>> stats['pearson_r']
        0.9999999999999999
    """
    if len(y_true.shape) == 1:
        y_true = y_true.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)
    
    results = {}
    
    for i in range(y_true.shape[1]):
        pearson_r, pearson_p = scipy.stats.pearsonr(y_true[:, i], y_pred[:, i])
        spearman_r, spearman_p = scipy.stats.spearmanr(y_true[:, i], y_pred[:, i])
        
        results[f'target_{i}'] = {
            'pearson_r': pearson_r,
            'pearson_p': pearson_p,
            'spearman_r': spearman_r,
            'spearman_p': spearman_p
        }
    
    return results


def bias_analysis(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Analyze prediction bias.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Dictionary containing bias statistics
        
    Example:
        >>> bias = bias_analysis([1, 2, 3], [1.1, 1.9, 3.1])
        >>> bias['mean_bias']
        0.03333333333333333
    """
    residuals = y_true - y_pred
    
    return {
        'mean_bias': np.mean(residuals, axis=0),
        'std_bias': np.std(residuals, axis=0),
        'median_bias': np.median(residuals, axis=0),
        'max_bias': np.max(residuals, axis=0),
        'min_bias': np.min(residuals, axis=0)
    } 