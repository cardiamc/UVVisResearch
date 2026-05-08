"""
Regression example: UV-Vis spectral data → multi-target prediction.

Pipeline:
- Data loading and preprocessing
- Model training and hold-out evaluation
- Double K-Fold CV with hyperparameter tuning (RandomForest)
- Results saved as CSV (per-target R², RMSE, MAPE)
- CV predictions aggregated across all test folds → scatter plot
- Experiment management and model persistence
"""

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — must be set before any pyplot import
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import logging

from uvvislib import (
    Config, setup_logging, DataLoader, Preprocessor,
    MLPRegressor, CNNRegressor, RandomForestRegressor,
    r2_score, rmse, compute_evaluation,
    DoubleKFoldCV,
    Plotter,
    ModelManager, ExperimentManager, DataPersistence
)


def main():
    setup_logging(log_level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("Starting regression example")

    config = Config(
        data_path="./Data/",
        uv_vis_data_file="Abs_100mm_completo_ARCHA.csv",
        chemical_data_file="Analisi_chimiche_complete_ARCHA.csv",
        wavelength_start=210,
        wavelength_end=622,
        wavelength_step=2,
        target_variables=[
            "CONDUCIBILITA'"
        ],
        log_target=False,
        apply_pca=False,
        k_fold_splits=5,
        random_state=42,
    )

    experiment_manager = ExperimentManager(config)
    model_manager = ModelManager(config)
    data_persistence = DataPersistence(config)

    experiment_id = experiment_manager.start_experiment(
        experiment_name="regression_analysis",
        description="Multi-target regression on UV-Vis spectra",
        tags=["regression", "uv-vis"],
    )

    try:
        # ── 1. DATA LOADING AND PREPROCESSING ────────────────────────────────
        logger.info("=== Data Loading and Preprocessing ===")

        data_loader = DataLoader(config)
        features_df, targets_df = data_loader.load_and_prepare_data()
        logger.info(f"Loaded: {features_df.shape[0]} samples, {features_df.shape[1]} features, "
                    f"{targets_df.shape[1]} targets")

        preprocessor = Preprocessor(config)
        processed_features, processed_targets = preprocessor.preprocess_pipeline(
            features_df, targets_df,
            clean_data=True,
            normalize_features=True,
            apply_pca=False,
            engineer_features=True,
        )
        logger.info(f"Preprocessed features: {processed_features.shape}")

        data_persistence.save_dataset(
            processed_features, processed_targets,
            dataset_name="processed_uv_vis_data",
            metadata={"description": "Preprocessed UV-Vis dataset"},
        )

        X = processed_features.values
        y = processed_targets.values
        target_names = targets_df.columns.tolist()

        # ── 2. HOLD-OUT EVALUATION ────────────────────────────────────────────
        logger.info("=== Hold-out Model Evaluation ===")

        models = {
            "MLP": MLPRegressor(config, hidden_size=100, epochs=100),
            "CNN": CNNRegressor(config, hidden_size=200, epochs=100),
            "RandomForest": RandomForestRegressor(config, n_estimators=50),
        }

        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        holdout_results = {}
        for name, model in models.items():
            logger.info(f"Training {name}...")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            metrics = compute_evaluation(y_test, y_pred)
            holdout_results[name] = {"model": model, "metrics": metrics,
                                     "predictions": y_pred, "true_values": y_test}

            logger.info(f"{name} — mean R²: {np.mean(metrics['r2_score']):.4f}, "
                        f"mean RMSE: {np.mean(metrics['rmse']):.4f}")

            experiment_manager.save_evaluation_csv(
                metrics,
                target_names=target_names,
                filename=f"{name.lower()}_holdout_evaluation.csv",
            )
            model_manager.save_model(model, f"{name.lower()}_model",
                                     experiment_name=experiment_id)

        # ── 3. CROSS-VALIDATION (RandomForest) ───────────────────────────────
        logger.info("=== Cross-Validation ===")

        cv = DoubleKFoldCV(
            outer_splits=5,
            inner_splits=4,
            random_state=config.random_state,
            stratify=False,  # multi-output continuous y incompatible with StratifiedKFold
        )

        rf_param_grid = {
            "n_estimators":    [5, 10, 15, 20, 23, 25, 30],
            "max_depth":       [2, 5, 6, 7, 8, 12, 15, 19, None],
            "min_samples_leaf":  [1, 3, 5, 10],
            "min_samples_split": [2, 3, 5, 8, 10, 15, 20, 30],
            "criterion":       ["poisson", "friedman_mse", "squared_error", "absolute_error"],
            "max_features":    ["sqrt", "log2"],
        }

        # DoubleKFoldCV needs a raw sklearn estimator
        rf_estimator = holdout_results["RandomForest"]["model"].model
        cv_scores = cv.fit(X, y, rf_estimator, rf_param_grid, n_iter=64)

        test_r2   = np.array([np.mean(s["r2_score"]) for s in cv_scores["test_scores"]])
        test_rmse = np.array([np.mean(s["rmse"])      for s in cv_scores["test_scores"]])
        logger.info(f"CV R²:   {np.mean(test_r2):.4f} ± {np.std(test_r2):.4f}")
        logger.info(f"CV RMSE: {np.mean(test_rmse):.4f} ± {np.std(test_rmse):.4f}")

        # Per-target CV summary (mean ± std across folds)
        experiment_manager.save_cv_results_csv(
            cv_scores,
            target_names=target_names,
            filename="cv_results_randomforest.csv",
        )

        # All test-fold predictions stacked into one array
        cv_y_pred = np.vstack(cv_scores["predictions"])   # (n_samples, n_targets)
        cv_y_true = np.vstack(cv_scores["true_values"])

        # Save every prediction made during CV (one row per sample, pred + true side-by-side)
        experiment_manager.save_predictions(
            cv_y_pred, cv_y_true,
            target_names=target_names,
            filename="cv_all_folds_predictions.csv",
        )

        cv_metrics = compute_evaluation(cv_y_true, cv_y_pred)
        experiment_manager.save_evaluation_csv(
            cv_metrics,
            target_names=target_names,
            filename="cv_all_folds_evaluation.csv",
        )

        # ── 4. VISUALIZATION ─────────────────────────────────────────────────
        logger.info("=== Visualization ===")

        plotter = Plotter()

        # Scatter: CV predictions vs actual (all test folds aggregated)
        # Use dpi=72 to keep the large multi-target grid within memory limits
        fig = plotter.plot_prediction_vs_actual(
            cv_y_true, cv_y_pred,
            target_names=target_names,
            title="RandomForest CV — Predicted vs Actual (all folds)",
            show=False,
        )
        experiment_manager.save_plot(fig, "cv_predictions_vs_actual.png", dpi=72)
        plt.close(fig)

        # Feature importance (Random Forest)
        rf_importance = holdout_results["RandomForest"]["model"].get_feature_importance()
        if rf_importance is not None:
            fig = plotter.plot_feature_importance(
                processed_features.columns.tolist(),
                rf_importance.reshape(1, -1),
                target_names=["RandomForest (mean importance)"],
                title="Random Forest Feature Importance",
                top_n=20,
                show=False,
            )
            experiment_manager.save_plot(fig, "random_forest_feature_importance.png")
            plt.close(fig)

        # ── 5. SAVE RESULTS ───────────────────────────────────────────────────
        logger.info("=== Saving Results ===")

        all_results = {
            "holdout_results": {n: r["metrics"] for n, r in holdout_results.items()},
            "cross_validation": cv_scores,
            "data_info": {
                "n_samples":    len(X),
                "n_features":   X.shape[1],
                "n_targets":    y.shape[1],
                "feature_names": processed_features.columns.tolist(),
                "target_names":  target_names,
            },
        }
        experiment_manager.save_results(all_results, "regression_results.json")

        for name, result in holdout_results.items():
            experiment_manager.save_predictions(
                result["predictions"], result["true_values"],
                target_names=target_names,
                filename=f"{name.lower()}_holdout_predictions.csv",
            )

        logger.info("All results saved successfully")

        # ── 6. SUMMARY ────────────────────────────────────────────────────────
        logger.info("=== Summary ===")
        best_name = max(holdout_results, key=lambda k: np.mean(holdout_results[k]["metrics"]["r2_score"]))
        logger.info(f"Experiment ID: {experiment_id}")
        logger.info(f"Best hold-out model: {best_name}  "
                    f"R² = {np.mean(holdout_results[best_name]['metrics']['r2_score']):.4f}")
        logger.info(f"CV (RF) mean R²: {np.mean(test_r2):.4f} ± {np.std(test_r2):.4f}")
        logger.info(f"CV aggregate R² (all folds): "
                    f"{np.mean(cv_metrics['r2_score']):.4f}")

    except Exception as e:
        logger.error(f"Error: {e}")
        experiment_manager.end_experiment(status="failed")
        raise
    else:
        experiment_manager.end_experiment(status="completed")
        logger.info("Regression analysis completed successfully!")


if __name__ == "__main__":
    main()
