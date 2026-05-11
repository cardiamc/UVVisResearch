"""
NIR regression example: NIR spectra (1 mm path, 4000–9984 cm⁻¹, Δ=16 cm⁻¹, 375 features)
→ multi-target chemical analysis prediction.

NIR-specific preprocessing strategies compared on a quick 80/20 hold-out:
  (A) Raw → MinMax → PCA(20)
  (B) SNV → MinMax → PCA(20)
  (C) SNV + SG smooth  (d=0) → MinMax → PCA(20)
  (D) SNV + SG 1st-deriv (d=1) → MinMax → PCA(20)   ← recommended for NIR
  (E) SNV + SG 2nd-deriv (d=2) → MinMax → PCA(20)

Main evaluation pipeline (D):
  SNV → Savitzky-Golay 1st derivative → MinMax → PCA(20) → RandomForest
  with double K-Fold cross-validation and hyperparameter tuning.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import logging

from sklearn.decomposition import PCA as SKPCA
from sklearn.ensemble import RandomForestRegressor as SKRF
from sklearn.metrics import r2_score as sk_r2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from uvvislib import (
    Config, setup_logging, DataLoader, Preprocessor,
    MLPRegressor, CNNRegressor, RandomForestRegressor,
    compute_evaluation,
    DoubleKFoldCV,
    Plotter,
    ModelManager, ExperimentManager, DataPersistence,
)

# ── Spectral and preprocessing constants ─────────────────────────────────────
SG_WINDOW      = 11    # Savitzky-Golay window length (spectral points)
SG_POLY        = 3     # Savitzky-Golay polynomial order
PCA_COMPONENTS = 20    # PCA components for dimensionality reduction

# All chemical targets in Analisi_chimiche_complete_ARCHA.csv
ALL_TARGETS = [
    "CONDUCIBILITA'", "POTENZIALE REDOX", "NITRITI", "NITRATI", "FLUORURI",
    "SODIO", "CALCIO", "MAGNESIO", "SOLFATI", "CLORURI", "DUREZZA (da calcolo)",
    "pH", "AMMONIO", "CLORATI", "ARSENICO", "ANTIMONIO", "ALLUMINIO", "CADMIO",
    "CROMO TOTALE", "FERRO", "MANGANESE", "NICHEL", "PIOMBO", "RAME", "SELENIO",
    "VANADIO", "MERCURIO", "BORO", "BROMODICLOROMETANO", "BROMOFORMIO",
    "CLOROFORMIO", "cis-1,2-DICLOROETILENE", "DIBROMOCLOROMETANO",
    "TETRACLOROETILENE", "TRICLOROETILENE", "1,1-DICLOROETANO",
    "1,1-DICLOROETILENE", "1,1,1-TRICLOROETANO", "1,2-DIBROMOETANO",
    "1,2-DICLOROETANO", "1,2-DICLOROPROPANO", "ESACLOROBUTADIENE",
    "CLORURO DI VINILE", "METILTERZIARBUTILETERE (MTBE)",
    "CONTA DI ESCHERICHIA COLI", "CONTA DI BATTERI COLIFORMI",
    "CONTA DI ENTEROCOCCHI INTESTINALI", "CONTA DI MICRORGANISMI VITALI A 36°C",
    "CONTA DI MICRORGANISMI VITALI A 22°C", "CONTA DI CLOSTRIDIUM PERFRINGENS",
    "RESIDUO FISSO A 180°C", "1,2,4-TRICLOROBENZENE", "1,2,3-TRICLOROBENZENE",
    "1,3-DICLOROBENZENE", "1,3,5-TRIMETILBENZENE", "n-PROPILBENZENE",
    "iso-PROPILBENZENE", "STIRENE", "o-XILENE", "(m+p)-XILENE", "ETILBENZENE",
    "TOLUENE", "CLORITI", "DICLOROMETANO", "1,4-DICLOROBENZENE",
    "IDROCARBURI LEGGERI (C5-C10)", "1,2,4-TRIMETILBENZENE", "TOC",
]


# ── Helper functions ──────────────────────────────────────────────────────────

def _apply_strategy(
    features_df: pd.DataFrame,
    strategy: str,
    pre: Preprocessor,
) -> np.ndarray:
    """Apply a named preprocessing strategy; return MinMax-scaled numpy array.

    Strategy keys: 'raw', 'snv', 'snv_sg0', 'snv_sg1', 'snv_sg2'.
    SNV and SG steps are stateless and can be called on any Preprocessor instance.
    MinMax scaling uses a fresh sklearn scaler so the Preprocessor state is unaffected.
    """
    X = features_df.copy()
    if "snv" in strategy:
        X = pre.apply_snv(X)
    if "sg0" in strategy:
        X = pre.apply_savitzky_golay(X, SG_WINDOW, SG_POLY, deriv=0)
    elif "sg1" in strategy:
        X = pre.apply_savitzky_golay(X, SG_WINDOW, SG_POLY, deriv=1)
    elif "sg2" in strategy:
        X = pre.apply_savitzky_golay(X, SG_WINDOW, SG_POLY, deriv=2)
    return MinMaxScaler().fit_transform(X.values.astype(float))


def _quick_r2(X: np.ndarray, y: np.ndarray, n_pca: int = 20, seed: int = 42) -> float:
    """80/20 hold-out mean R² with RF(50 trees) + PCA.

    Targets with near-zero relative variance in the test set are skipped —
    they represent analytes at a constant detection limit where R² is undefined.
    Per-target R² is clipped to [-1, 1] before averaging to prevent extreme
    values from targets with tiny SS_tot from dominating.
    """
    n_comp = min(n_pca, X.shape[1], X.shape[0] - 1)
    X_pca = SKPCA(n_components=n_comp, random_state=seed).fit_transform(X)
    X_tr, X_te, y_tr, y_te = train_test_split(X_pca, y, test_size=0.2, random_state=seed)
    rf = SKRF(n_estimators=50, random_state=seed, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    y_pred = rf.predict(X_te)
    r2s = []
    for j in range(y_te.shape[1]):
        col = y_te[:, j]
        # Skip targets where std is less than 1% of the absolute mean — too close
        # to constant to give a meaningful R² (SS_tot ≈ 0 causes numerical blow-up)
        if col.std() < max(1e-6, 0.01 * (abs(col.mean()) + 1e-10)):
            continue
        r2s.append(np.clip(sk_r2(col, y_pred[:, j]), -1.0, 1.0))
    return float(np.mean(r2s)) if r2s else 0.0


def _plot_preprocessing_effects(
    features_clean: pd.DataFrame,
    pre: Preprocessor,
    wavenums: np.ndarray,
    n_samples: int = 3,
) -> plt.Figure:
    """5-panel figure showing the effect of each preprocessing stage on raw NIR spectra."""
    sample_idx = np.linspace(0, len(features_clean) - 1, n_samples, dtype=int)
    colors = ["steelblue", "darkorange", "forestgreen"]

    snv = pre.apply_snv(features_clean)
    stages = [
        ("Raw NIR",                        features_clean.values),
        ("SNV",                            snv.values),
        ("SNV + SG smooth (d=0)",          pre.apply_savitzky_golay(snv, SG_WINDOW, SG_POLY, 0).values),
        ("SNV + SG 1st derivative (d=1)",  pre.apply_savitzky_golay(snv, SG_WINDOW, SG_POLY, 1).values),
        ("SNV + SG 2nd derivative (d=2)",  pre.apply_savitzky_golay(snv, SG_WINDOW, SG_POLY, 2).values),
    ]

    fig, axes = plt.subplots(5, 1, figsize=(12, 15), sharex=True)
    fig.suptitle("NIR Preprocessing Effect on Spectra", fontsize=13)
    for ax, (title, arr) in zip(axes, stages):
        for j, s in enumerate(sample_idx):
            ax.plot(wavenums, arr[s], color=colors[j], alpha=0.85, linewidth=0.8,
                    label=f"Sample {s}" if title == "Raw NIR" else None)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("a.u.")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Wavenumber (cm⁻¹)")
    axes[0].legend(fontsize=8)
    plt.tight_layout()
    return fig


def _plot_strategy_comparison(strategy_r2: dict, best_label: str) -> plt.Figure:
    """Bar chart comparing mean hold-out R² across preprocessing strategies."""
    labels = list(strategy_r2.keys())
    r2_vals = [strategy_r2[l] for l in labels]
    bar_colors = ["steelblue"] * len(labels)
    bar_colors[labels.index(best_label)] = "darkorange"

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(len(labels)), r2_vals, color=bar_colors, edgecolor="black")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Mean R² (80/20 hold-out, RF+PCA)")
    ax.set_title("NIR Preprocessing Strategy Comparison")
    y_min = min(r2_vals)
    y_max = max(r2_vals)
    pad = max(abs(y_max - y_min) * 0.15, 0.05)
    ax.set_ylim(y_min - pad, y_max + pad)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    for bar, v in zip(bars, r2_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005,
                f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    setup_logging(log_level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("Starting NIR regression example")

    # ── CONFIG ────────────────────────────────────────────────────────────────
    # np.arange(4000, 10000, 16) → 375 wavenumber features (4000–9984 cm⁻¹)
    config = Config(
        data_path="./Data/",
        uv_vis_data_file="NIR_1mm_completo_ARCHA.csv",
        chemical_data_file="Analisi_chimiche_complete_ARCHA.csv",
        wavelength_start=4000,
        wavelength_end=10000,
        wavelength_step=16,
        apply_smoothing=False,   # Gaussian replaced by Savitzky-Golay
        apply_pca=False,
        target_variables=ALL_TARGETS,
        log_target=False,
        k_fold_splits=5,
        random_state=42,
    )

    experiment_manager = ExperimentManager(config)
    model_manager = ModelManager(config)
    data_persistence = DataPersistence(config)

    experiment_id = experiment_manager.start_experiment(
        experiment_name="nir_regression",
        description="NIR 1mm spectra → chemical analysis (SNV + SG 1st-deriv + PCA(20))",
        tags=["nir", "regression", "savitzky-golay", "snv", "pca"],
    )

    try:
        # ── 1. DATA LOADING ───────────────────────────────────────────────────
        logger.info("=== Data Loading ===")

        data_loader = DataLoader(config)
        data_loader.load_uv_vis_data()       # reuses the 'uv_vis_data_file' field for NIR
        data_loader.load_chemical_data()
        data_loader.combine_data(["Week", "Gathering_Point"])

        # config.feature_wavelengths → ["4000", "4016", ..., "9984"] (375 cols, ascending)
        nir_wavenumbers = config.feature_wavelengths
        available_wn = [w for w in nir_wavenumbers if w in data_loader.combined_data.columns]
        features_df = data_loader.combined_data[available_wn].copy()
        logger.info(f"NIR features: {features_df.shape[1]} wavenumber channels "
                    f"({available_wn[0]}–{available_wn[-1]} cm⁻¹, Δ=16 cm⁻¹)")

        # Extract chemical targets present in the merged dataset
        available_targets = [t for t in ALL_TARGETS if t in data_loader.combined_data.columns]
        targets_df = data_loader.combined_data[available_targets].copy()

        # Coerce to numeric (handles comma decimal separators)
        for col in targets_df.columns:
            if not pd.api.types.is_numeric_dtype(targets_df[col]):
                targets_df[col] = pd.to_numeric(
                    targets_df[col].astype(str).str.strip().str.replace(",", ".", regex=False),
                    errors="coerce",
                )

        logger.info(f"Loaded: {len(features_df)} samples, "
                    f"{features_df.shape[1]} NIR features, {len(available_targets)} targets")

        # ── 2. TARGET COVERAGE FILTERING ──────────────────────────────────────
        logger.info("=== Target Coverage Filtering ===")

        min_coverage = 0.80
        coverage = targets_df.notna().mean()
        valid_targets = coverage[coverage >= min_coverage].index.tolist()
        dropped = [t for t in available_targets if t not in valid_targets]
        targets_df = targets_df[valid_targets]

        logger.info(f"Retained {len(valid_targets)}/{len(available_targets)} targets "
                    f"(>= {min_coverage*100:.0f}% non-null coverage)")
        if dropped:
            logger.info(f"Dropped targets (low coverage): {dropped}")

        # ── 3. NIR PREPROCESSING ──────────────────────────────────────────────
        logger.info("=== NIR Preprocessing ===")

        preprocessor = Preprocessor(config)
        features_clean, targets_clean = preprocessor.clean_data(
            features_df, targets_df, remove_nan_targets=True
        )
        logger.info(f"Samples after cleaning: {len(features_clean)}")

        # Wavenumber axis as integers (ascending), used for plotting
        wavenums = np.array([int(c) for c in features_clean.columns])

        # 3a. Visualise the effect of each preprocessing stage
        logger.info("Generating spectral preprocessing visualisation...")
        fig_preproc = _plot_preprocessing_effects(features_clean, preprocessor, wavenums)
        experiment_manager.save_plot(fig_preproc, "nir_preprocessing_effects.png", dpi=90)
        plt.close(fig_preproc)

        # 3b. Compare preprocessing strategies via quick 80/20 hold-out
        logger.info("Comparing preprocessing strategies (RF + PCA, 80/20 hold-out)...")
        y_clean = targets_clean.values
        strategies = {
            "A: Raw → MinMax → PCA":                "raw",
            "B: SNV → MinMax → PCA":                "snv",
            "C: SNV + SG smooth → MinMax → PCA":    "snv_sg0",
            "D: SNV + SG 1st-deriv → MinMax → PCA": "snv_sg1",
            "E: SNV + SG 2nd-deriv → MinMax → PCA": "snv_sg2",
        }
        strategy_r2 = {}
        for label, key in strategies.items():
            X_s = _apply_strategy(features_clean, key, preprocessor)
            r2 = _quick_r2(X_s, y_clean, n_pca=PCA_COMPONENTS, seed=config.random_state)
            strategy_r2[label] = r2
            logger.info(f"  {label}: mean R² = {r2:.4f}")

        best_label = max(strategy_r2, key=strategy_r2.get)
        logger.info(f"Best strategy (hold-out): {best_label}  R²={strategy_r2[best_label]:.4f}")

        fig_comp = _plot_strategy_comparison(strategy_r2, best_label)
        experiment_manager.save_plot(fig_comp, "nir_strategy_comparison.png")
        plt.close(fig_comp)

        # 3c. Main pipeline: SNV → SG 1st derivative → MinMax → PCA(20)
        logger.info("Applying main pipeline: SNV → SG-1d → MinMax → PCA(20)...")
        features_proc = preprocessor.apply_snv(features_clean)
        features_proc = preprocessor.apply_savitzky_golay(
            features_proc, window_length=SG_WINDOW, polyorder=SG_POLY, deriv=1
        )
        features_proc = preprocessor.normalize_features(
            features_proc, scaler_type="minmax", fit=True
        )
        features_proc = preprocessor.apply_pca(
            features_proc, n_components=PCA_COMPONENTS, fit=True
        )
        targets_proc = preprocessor.apply_target_transformation(
            targets_clean, log_transform=False
        )

        logger.info(f"Final feature matrix: {features_proc.shape} | "
                    f"Targets: {targets_proc.shape}")

        data_persistence.save_dataset(
            features_proc, targets_proc,
            dataset_name="nir_preprocessed_data",
            metadata={
                "description": "NIR spectra after SNV + SG-1d + MinMax + PCA(20)",
                "sg_window": SG_WINDOW,
                "sg_poly": SG_POLY,
            },
        )

        X = features_proc.values
        y = targets_proc.values
        target_names = targets_proc.columns.tolist()

        # ── 4. HOLD-OUT EVALUATION ────────────────────────────────────────────
        logger.info("=== Hold-out Model Evaluation ===")

        # CNN and MLP operate on the 20 PCA components (not raw spectra);
        # the convolution in CNN acts on the PCA feature axis.
        models = {
            "MLP":          MLPRegressor(config, hidden_size=100, epochs=100),
            "CNN":          CNNRegressor(config, hidden_size=200, epochs=100),
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
            holdout_results[name] = {
                "model":       model,
                "metrics":     metrics,
                "predictions": y_pred,
                "true_values": y_test,
            }
            logger.info(f"{name} — mean R²: {np.mean(metrics['r2_score']):.4f}, "
                        f"mean RMSE: {np.mean(metrics['rmse']):.4f}")
            experiment_manager.save_evaluation_csv(
                metrics,
                target_names=target_names,
                filename=f"{name.lower()}_holdout_evaluation.csv",
            )
            model_manager.save_model(model, f"{name.lower()}_model",
                                     experiment_name=experiment_id)

        # ── 5. CROSS-VALIDATION (RandomForest) ───────────────────────────────
        logger.info("=== Cross-Validation (RandomForest, double K-Fold) ===")

        cv = DoubleKFoldCV(
            outer_splits=5,
            inner_splits=4,
            random_state=config.random_state,
            stratify=False,   # continuous multi-output targets
        )

        rf_param_grid = {
            "n_estimators":      [5, 10, 15, 20, 23, 25, 30],
            "max_depth":         [2, 5, 6, 7, 8, 12, 15, 19, None],
            "min_samples_leaf":  [1, 3, 5, 10],
            "min_samples_split": [2, 3, 5, 8, 10, 15, 20, 30],
            "criterion":         ["friedman_mse", "squared_error", "absolute_error"],
            "max_features":      ["sqrt", "log2"],
        }

        rf_estimator = holdout_results["RandomForest"]["model"].model
        cv_scores = cv.fit(X, y, rf_estimator, rf_param_grid, n_iter=48)

        test_r2   = np.array([np.mean(s["r2_score"]) for s in cv_scores["test_scores"]])
        test_rmse = np.array([np.mean(s["rmse"])      for s in cv_scores["test_scores"]])
        logger.info(f"CV R²:   {np.mean(test_r2):.4f} ± {np.std(test_r2):.4f}")
        logger.info(f"CV RMSE: {np.mean(test_rmse):.4f} ± {np.std(test_rmse):.4f}")

        experiment_manager.save_cv_results_csv(
            cv_scores,
            target_names=target_names,
            filename="cv_results_randomforest.csv",
        )

        cv_y_pred = np.vstack(cv_scores["predictions"])
        cv_y_true = np.vstack(cv_scores["true_values"])

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

        # ── 6. VISUALIZATION ──────────────────────────────────────────────────
        logger.info("=== Visualization ===")

        plotter = Plotter()

        # Scatter: CV predictions vs actual (all test folds aggregated)
        fig = plotter.plot_prediction_vs_actual(
            cv_y_true, cv_y_pred,
            target_names=target_names,
            title="RandomForest CV — NIR → Chemical (all folds)",
            show=False,
        )
        experiment_manager.save_plot(fig, "nir_cv_predictions_vs_actual.png", dpi=72)
        plt.close(fig)

        # Feature importance for RF on PCA components
        rf_importance = holdout_results["RandomForest"]["model"].get_feature_importance()
        if rf_importance is not None:
            fig = plotter.plot_feature_importance(
                features_proc.columns.tolist(),
                rf_importance.reshape(1, -1),
                target_names=["RF mean importance (PCA components)"],
                title="Random Forest Feature Importance (PCA components)",
                top_n=PCA_COMPONENTS,
                show=False,
            )
            experiment_manager.save_plot(fig, "nir_rf_feature_importance.png")
            plt.close(fig)

        # ── 7. SAVE RESULTS ───────────────────────────────────────────────────
        logger.info("=== Saving Results ===")

        all_results = {
            "holdout_results": {n: r["metrics"] for n, r in holdout_results.items()},
            "cross_validation": cv_scores,
            "preprocessing_comparison": strategy_r2,
            "data_info": {
                "n_samples":         len(X),
                "n_nir_features":    len(available_wn),
                "wavenumber_range":  f"{available_wn[0]}–{available_wn[-1]} cm⁻¹",
                "wavenumber_step":   "16 cm⁻¹",
                "n_pca_components":  PCA_COMPONENTS,
                "n_targets":         y.shape[1],
                "sg_window":         SG_WINDOW,
                "sg_poly":           SG_POLY,
                "sg_deriv":          1,
                "target_names":      target_names,
                "dropped_targets":   dropped,
            },
        }
        experiment_manager.save_results(all_results, "nir_regression_results.json")

        for name, result in holdout_results.items():
            experiment_manager.save_predictions(
                result["predictions"], result["true_values"],
                target_names=target_names,
                filename=f"{name.lower()}_holdout_predictions.csv",
            )

        logger.info("All results saved successfully")

        # ── 8. SUMMARY ────────────────────────────────────────────────────────
        logger.info("=== Summary ===")
        best_ho = max(
            holdout_results,
            key=lambda k: np.mean(holdout_results[k]["metrics"]["r2_score"])
        )
        logger.info(f"Experiment ID: {experiment_id}")
        logger.info(f"NIR features:  {len(available_wn)} channels "
                    f"({available_wn[0]}–{available_wn[-1]} cm⁻¹, Δ=16 cm⁻¹)")
        logger.info(f"Samples:       {len(X)}")
        logger.info(f"Targets:       {y.shape[1]} (from {len(available_targets)} available; "
                    f"dropped {len(dropped)} with < {min_coverage*100:.0f}% coverage)")
        logger.info(f"Best preprocessing strategy (hold-out): {best_label}")
        logger.info(f"Best hold-out model: {best_ho}  "
                    f"R² = {np.mean(holdout_results[best_ho]['metrics']['r2_score']):.4f}")
        logger.info(f"CV (RF) mean R²: {np.mean(test_r2):.4f} ± {np.std(test_r2):.4f}")
        logger.info(f"CV aggregate R² (all folds): {np.mean(cv_metrics['r2_score']):.4f}")

    except Exception as e:
        logger.error(f"Error: {e}")
        experiment_manager.end_experiment(status="failed")
        raise
    else:
        experiment_manager.end_experiment(status="completed")
        logger.info("NIR regression analysis completed successfully!")


if __name__ == "__main__":
    main()
