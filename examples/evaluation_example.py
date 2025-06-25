"""
Example script demonstrating the evaluation module functionality.

This script shows how to use the evaluation metrics and cross-validation
tools for UV-Vis spectral analysis.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

# Import the library
import sys
sys.path.append('..')

from uvvislib.evaluation import (
    # Metrics
    r2_score, r2_score_single, polyfit, mape, rmse, rmse_exp,
    neg_rmse_exp, compute_evaluation, correlation_analysis, bias_analysis,
    
    # Cross-validation
    DoubleKFoldCV, NestedCrossValidation, create_cv_scorers, 
    stratified_regression_cv
)


def main():
    """Main example function."""
    print("UV-Vis Evaluation Module Example")
    print("=" * 50)
    
    # Generate sample data
    np.random.seed(42)
    n_samples, n_features, n_targets = 100, 20, 3
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.randn(n_samples, n_targets)
    
    # Add some correlation between features and targets
    for i in range(n_targets):
        y[:, i] = 0.7 * X[:, i] + 0.3 * np.random.randn(n_samples)
    
    print(f"Generated data: {X.shape[0]} samples, {X.shape[1]} features, {y.shape[1]} targets")
    
    # Example 1: Basic metrics
    print("\n1. Basic Evaluation Metrics")
    print("-" * 30)
    
    # Split data for demonstration
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Train a simple model
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Compute various metrics
    print("R2 Score (single target):", r2_score_single(y_test[:, 0], y_pred[:, 0]))
    print("R2 Score (all targets):", r2_score(y_test, y_pred))
    print("RMSE:", rmse(y_test, y_pred))
    print("MAPE:", mape(y_test, y_pred))
    
    # Comprehensive evaluation
    eval_results = compute_evaluation(y_test, y_pred)
    print("\nComprehensive Evaluation:")
    for metric, value in eval_results.items():
        print(f"  {metric}: {value}")
    
    # Example 2: Correlation analysis
    print("\n2. Correlation Analysis")
    print("-" * 30)
    
    corr_results = correlation_analysis(y_test, y_pred)
    for target, stats in corr_results.items():
        print(f"\n{target}:")
        print(f"  Pearson r: {stats['pearson_r']:.4f}")
        print(f"  Spearman r: {stats['spearman_r']:.4f}")
    
    # Example 3: Bias analysis
    print("\n3. Bias Analysis")
    print("-" * 30)
    
    bias_results = bias_analysis(y_test, y_pred)
    for metric, value in bias_results.items():
        print(f"  {metric}: {value}")
    
    # Example 4: Cross-validation with Double K-Fold
    print("\n4. Double K-Fold Cross-Validation")
    print("-" * 40)
    
    # Define parameter grid
    param_grid = {
        'n_estimators': [5, 10, 15],
        'max_depth': [3, 5, None],
        'random_state': [42]
    }
    
    # Initialize CV
    cv = DoubleKFoldCV(outer_splits=3, inner_splits=2, random_state=42)
    
    # Perform CV
    print("Performing double k-fold cross-validation...")
    cv_results = cv.fit(
        X, y[:, 0],  # Use first target for simplicity
        RandomForestRegressor, 
        param_grid, 
        n_iter=10, 
        log_target=False
    )
    
    # Get summary
    summary = cv.get_summary(cv_results)
    print("\nCV Summary:")
    for metric, value in summary.items():
        print(f"  {metric}: {value}")
    
    # Example 5: Nested Cross-Validation
    print("\n5. Nested Cross-Validation")
    print("-" * 30)
    
    ncv = NestedCrossValidation(outer_cv=3, inner_cv=2, random_state=42)
    ridge_params = {'alpha': [0.1, 1.0, 10.0]}
    
    scores = ncv.cross_val_score(
        Ridge(), X, y[:, 0], ridge_params,
        scoring='neg_mean_squared_error', n_iter=5
    )
    
    print(f"Nested CV scores: {scores}")
    print(f"Mean score: {np.mean(scores):.4f}")
    print(f"Std score: {np.std(scores):.4f}")
    
    # Example 6: Log-transformed targets
    print("\n6. Log-Transformed Targets")
    print("-" * 30)
    
    # Simulate log-transformed data
    y_log = np.log(np.abs(y) + 1)  # Add 1 to avoid log(0)
    y_pred_log = np.log(np.abs(y_pred) + 1)
    
    # Compute metrics for log-transformed data
    eval_results_log = compute_evaluation(y_log, y_pred_log, log_target=True)
    print("Evaluation with log-transformed targets:")
    for metric, value in eval_results_log.items():
        print(f"  {metric}: {value}")
    
    # Example 7: Creating custom scorers
    print("\n7. Custom Scoring Functions")
    print("-" * 30)
    
    scorers_log = create_cv_scorers(log_target=True)
    scorers_normal = create_cv_scorers(log_target=False)
    
    print("Scorers for log targets:", list(scorers_log.keys()))
    print("Scorers for normal targets:", list(scorers_normal.keys()))
    
    print("\nExample completed successfully!")


if __name__ == "__main__":
    main() 