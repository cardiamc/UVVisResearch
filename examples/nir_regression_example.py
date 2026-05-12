"""
NIR regression example: NIR spectra (1 mm path, 4000–9984 cm⁻¹, Δ=16 cm⁻¹, 375 features)
→ multi-target chemical analysis prediction.

Preprocessing strategies compared (8 × {no PCA, PCA-20}):
  (A) Raw                         → MinMax
  (B) SNV                         → MinMax
  (C) SNV + SG 1st-deriv          → MinMax
  (D) SNV + SG 2nd-deriv          → MinMax
  (E) H₂O sub                     → MinMax
  (F) H₂O sub + SNV               → MinMax
  (G) H₂O sub + SNV + SG 1st-deriv → MinMax   ← expected best for NIR
  (H) H₂O sub + SNV + SG 2nd-deriv → MinMax

Week-matched distilled-water spectra (NIR_1mm_completo_ARCHA_h20.csv) are used
to correct for instrument drift and the dominant water absorption background.

Main pipeline (best strategy from hold-out comparison, no PCA):
  H₂O sub → SNV → Savitzky-Golay 1st derivative → MinMax → RandomForest
  with double K-Fold cross-validation and hyperparameter tuning.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import logging
from pathlib import Path

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
PCA_COMPONENTS = 20    # PCA components used in the PCA-comparison branch

H2O_FILE = "NIR_1mm_completo_ARCHA_h20.csv"

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

def _build_water_ref_matrix(
    features_clean: pd.DataFrame,
    combined_data: pd.DataFrame,
    h2o_df: pd.DataFrame,
    available_wn: list,
) -> np.ndarray:
    """Build a (n_samples, n_wavenumbers) week-matched water reference matrix.

    For each sample in *features_clean* the corresponding week's mean water
    spectrum is looked up. Samples whose week has no water measurement fall
    back to the global mean across all water measurements.

    Returns
    -------
    np.ndarray, shape (n_samples, n_wavenumbers)
    """
    global_mean = h2o_df[available_wn].mean()
    h2o_by_week = h2o_df.groupby("WEEK")[available_wn].mean().fillna(global_mean)
    fallback = global_mean.values

    weeks = combined_data.loc[features_clean.index, "Week"]
    rows = []
    for week in weeks:
        rows.append(h2o_by_week.loc[week].values if week in h2o_by_week.index else fallback)
    return np.stack(rows)   # (n_samples, n_wavenumbers)


def _apply_strategy(
    features_df: pd.DataFrame,
    strategy: str,
    pre: Preprocessor,
    water_ref: np.ndarray | None = None,
) -> np.ndarray:
    """Apply a named preprocessing strategy; return MinMax-scaled numpy array.

    Strategy keys (combinable):
      'h2o'  → water background subtraction (requires *water_ref*)
      'snv'  → Standard Normal Variate normalisation
      'sg1'  → Savitzky-Golay 1st derivative
      'sg2'  → Savitzky-Golay 2nd derivative
    MinMax scaling always applied last; fresh sklearn scaler leaves Preprocessor
    state unaffected.
    """
    X = features_df.copy()
    if "h2o" in strategy and water_ref is not None:
        X = pre.apply_water_subtraction(X, water_ref)
    if "snv" in strategy:
        X = pre.apply_snv(X)
    if "sg1" in strategy:
        X = pre.apply_savitzky_golay(X, SG_WINDOW, SG_POLY, deriv=1)
    elif "sg2" in strategy:
        X = pre.apply_savitzky_golay(X, SG_WINDOW, SG_POLY, deriv=2)
    return MinMaxScaler().fit_transform(X.values.astype(float))


def _quick_r2(
    X: np.ndarray,
    y: np.ndarray,
    n_pca: int = 0,
    seed: int = 42,
) -> float:
    """80/20 hold-out mean R² with RF(50 trees).

    *n_pca=0* skips PCA and uses the raw *X* matrix (recommended when
    the spectral features are already informative after preprocessing).
    Near-constant targets (std < 1% of |mean|) are excluded from the average;
    remaining per-target R² values are clipped to [-1, 1].
    """
    if n_pca > 0:
        n_comp = min(n_pca, X.shape[1], X.shape[0] - 1)
        X = SKPCA(n_components=n_comp, random_state=seed).fit_transform(X)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=seed)
    rf = SKRF(n_estimators=50, random_state=seed, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    y_pred = rf.predict(X_te)
    r2s = []
    for j in range(y_te.shape[1]):
        col = y_te[:, j]
        # Skip analytes near constant detection limit (SS_tot ≈ 0 → R² undefined)
        if col.std() < max(1e-6, 0.01 * (abs(col.mean()) + 1e-10)):
            continue
        r2s.append(np.clip(sk_r2(col, y_pred[:, j]), -1.0, 1.0))
    return float(np.mean(r2s)) if r2s else 0.0


def _plot_preprocessing_effects(
    features_clean: pd.DataFrame,
    features_water: pd.DataFrame,
    pre: Preprocessor,
    wavenums: np.ndarray,
    n_samples: int = 3,
) -> plt.Figure:
    """7-panel figure: raw NIR → water sub → SNV → SG derivatives (both branches)."""
    sample_idx = np.linspace(0, len(features_clean) - 1, n_samples, dtype=int)
    colors = ["steelblue", "darkorange", "forestgreen"]

    snv_raw   = pre.apply_snv(features_clean)
    snv_water = pre.apply_snv(features_water)

    stages = [
        ("Raw NIR",                            features_clean.values),
        ("H₂O subtracted",                     features_water.values),
        ("SNV (no water sub)",                 snv_raw.values),
        ("H₂O sub + SNV",                     snv_water.values),
        ("H₂O sub + SNV + SG smooth (d=0)",   pre.apply_savitzky_golay(snv_water, SG_WINDOW, SG_POLY, 0).values),
        ("H₂O sub + SNV + SG 1st-deriv (d=1)",pre.apply_savitzky_golay(snv_water, SG_WINDOW, SG_POLY, 1).values),
        ("H₂O sub + SNV + SG 2nd-deriv (d=2)",pre.apply_savitzky_golay(snv_water, SG_WINDOW, SG_POLY, 2).values),
    ]

    fig, axes = plt.subplots(len(stages), 1, figsize=(12, 3 * len(stages)), sharex=True)
    fig.suptitle("NIR Preprocessing Stages", fontsize=13)
    for ax, (title, arr) in zip(axes, stages):
        for j, s in enumerate(sample_idx):
            ax.plot(wavenums, arr[s], color=colors[j], alpha=0.85, linewidth=0.8,
                    label=f"Sample {s}" if title == "Raw NIR" else None)
        ax.set_title(title, fontsize=9)
        ax.set_ylabel("a.u.")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Wavenumber (cm⁻¹)")
    axes[0].legend(fontsize=8)
    plt.tight_layout()
    return fig


def _plot_strategy_comparison(
    strategy_r2_nopca: dict,
    strategy_r2_pca: dict,
    best_label: str,
) -> plt.Figure:
    """Side-by-side bar charts comparing preprocessing strategies (no-PCA vs PCA-20)."""
    labels = list(strategy_r2_nopca.keys())
    r2_nopca = [strategy_r2_nopca[l] for l in labels]
    r2_pca   = [strategy_r2_pca[l]   for l in labels]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=False)
    for ax, r2_vals, title_suffix in zip(
        axes,
        [r2_nopca, r2_pca],
        ["No PCA (375 features)", f"PCA-{PCA_COMPONENTS}"],
    ):
        bar_colors = ["steelblue"] * len(labels)
        if best_label in labels:
            bar_colors[labels.index(best_label)] = "darkorange"
        bars = ax.bar(range(len(labels)), r2_vals, color=bar_colors, edgecolor="black")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel("Mean R² (80/20 hold-out, RF)")
        ax.set_title(f"NIR Strategy Comparison — {title_suffix}")
        y_min = min(r2_vals)
        y_max = max(r2_vals)
        pad = max(abs(y_max - y_min) * 0.15, 0.05)
        ax.set_ylim(y_min - pad, y_max + pad)
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        for bar, v in zip(bars, r2_vals):
            offset = 0.003 if v >= 0 else -0.015
            ax.text(bar.get_x() + bar.get_width() / 2, v + offset,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    return fig


def _plot_top_targets(
    cv_y_true: np.ndarray,
    cv_y_pred: np.ndarray,
    target_names: list,
    n_top: int = 12,
    title: str = "",
) -> plt.Figure:
    """Scatter plot grid for the top-N targets ranked by CV R²."""
    from uvvislib.evaluation.metrics import r2_score as lib_r2
    r2_per_target = lib_r2(cv_y_pred, cv_y_true)
    order = np.argsort(r2_per_target)[::-1][:n_top]

    ncols = 4
    nrows = int(np.ceil(n_top / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3.5))
    axes = axes.flatten()
    fig.suptitle(title, fontsize=11)

    for ax, idx in zip(axes, order):
        yt = cv_y_true[:, idx]
        yp = cv_y_pred[:, idx]
        r2 = r2_per_target[idx]
        ax.scatter(yt, yp, alpha=0.4, s=12, color="steelblue")
        lims = [min(yt.min(), yp.min()), max(yt.max(), yp.max())]
        ax.plot(lims, lims, "r--", linewidth=0.8)
        ax.set_title(f"{target_names[idx]}\nR²={r2:.3f}", fontsize=8)
        ax.set_xlabel("True", fontsize=7)
        ax.set_ylabel("Pred", fontsize=7)
        ax.tick_params(labelsize=7)

    for ax in axes[len(order):]:
        ax.set_visible(False)
    plt.tight_layout()
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    setup_logging(log_level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("Starting NIR regression example (with water subtraction)")

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
        apply_pca=False,         # PCA tested as a comparison, not the main pipeline
        pca_components=PCA_COMPONENTS,
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
        description="NIR 1mm → chemical analysis: H₂O sub + SNV + SG-1d + RF (no PCA)",
        tags=["nir", "regression", "water-subtraction", "savitzky-golay", "snv"],
    )

    try:
        # ── 1. DATA LOADING ───────────────────────────────────────────────────
        logger.info("=== Data Loading ===")

        data_loader = DataLoader(config)
        data_loader.load_uv_vis_data()
        data_loader.load_chemical_data()
        data_loader.combine_data(["Week", "Gathering_Point"])

        nir_wavenumbers = config.feature_wavelengths   # ["4000", "4016", ..., "9984"]
        available_wn = [w for w in nir_wavenumbers if w in data_loader.combined_data.columns]
        features_df  = data_loader.combined_data[available_wn].copy()
        logger.info(f"NIR features: {features_df.shape[1]} wavenumber channels "
                    f"({available_wn[0]}–{available_wn[-1]} cm⁻¹, Δ=16 cm⁻¹)")

        available_targets = [t for t in ALL_TARGETS if t in data_loader.combined_data.columns]
        targets_df = data_loader.combined_data[available_targets].copy()

        # Coerce targets to numeric
        for col in targets_df.columns:
            if not pd.api.types.is_numeric_dtype(targets_df[col]):
                targets_df[col] = pd.to_numeric(
                    targets_df[col].astype(str).str.strip().str.replace(",", ".", regex=False),
                    errors="coerce",
                )

        logger.info(f"Loaded: {len(features_df)} samples, "
                    f"{features_df.shape[1]} NIR features, {len(available_targets)} targets")

        # ── 1b. WATER REFERENCE ───────────────────────────────────────────────
        logger.info("=== Loading Water Reference ===")

        h2o_df = pd.read_csv(Path(config.data_path) / H2O_FILE, sep=";", decimal=",")
        # Select only the wavenumbers used by the NIR example
        h2o_avail = [w for w in available_wn if w in h2o_df.columns]
        h2o_by_week = h2o_df.groupby("WEEK")[h2o_avail].mean()
        logger.info(f"Water reference: {len(h2o_df)} measurements, "
                    f"{h2o_by_week.shape[0]} weeks")

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

        # ── 3. CLEANING AND WATER SUBTRACTION ─────────────────────────────────
        logger.info("=== NIR Preprocessing ===")

        preprocessor = Preprocessor(config)
        features_clean, targets_clean = preprocessor.clean_data(
            features_df, targets_df, remove_nan_targets=True
        )
        logger.info(f"Samples after cleaning: {len(features_clean)}")

        # Build week-matched water reference matrix aligned to cleaned index
        water_ref_matrix = _build_water_ref_matrix(
            features_clean, data_loader.combined_data, h2o_df, h2o_avail
        )
        features_water = preprocessor.apply_water_subtraction(features_clean, water_ref_matrix)
        logger.info(f"Water subtraction: baseline reduced from "
                    f"{features_clean.values.mean():.4f} to {features_water.values.mean():.4f} abs")

        wavenums = np.array([int(c) for c in features_clean.columns])

        # 3a. Visualise preprocessing stages
        logger.info("Generating spectral preprocessing visualisation...")
        fig_preproc = _plot_preprocessing_effects(
            features_clean, features_water, preprocessor, wavenums
        )
        experiment_manager.save_plot(fig_preproc, "nir_preprocessing_stages.png", dpi=90)
        plt.close(fig_preproc)

        # 3b. Strategy comparison: 8 strategies × {no PCA, PCA-20}
        logger.info("Comparing preprocessing strategies (RF, 80/20 hold-out)...")
        y_clean = targets_clean.values

        strategies = {
            "A: Raw":                   "raw",
            "B: SNV":                   "snv",
            "C: SNV+SG-1d":             "snv_sg1",
            "D: SNV+SG-2d":             "snv_sg2",
            "E: H₂O":                   "h2o",
            "F: H₂O+SNV":              "h2o_snv",
            "G: H₂O+SNV+SG-1d":        "h2o_snv_sg1",
            "H: H₂O+SNV+SG-2d":        "h2o_snv_sg2",
        }

        strategy_r2_nopca: dict[str, float] = {}
        strategy_r2_pca:   dict[str, float] = {}

        for label, key in strategies.items():
            X_s = _apply_strategy(features_clean, key, preprocessor, water_ref_matrix)
            r2_nopca = _quick_r2(X_s, y_clean, n_pca=0,            seed=config.random_state)
            r2_pca   = _quick_r2(X_s, y_clean, n_pca=PCA_COMPONENTS, seed=config.random_state)
            strategy_r2_nopca[label] = r2_nopca
            strategy_r2_pca[label]   = r2_pca
            logger.info(f"  {label}: no-PCA R²={r2_nopca:.4f}  PCA-{PCA_COMPONENTS} R²={r2_pca:.4f}")

        best_label_nopca = max(strategy_r2_nopca, key=strategy_r2_nopca.get)
        best_label_pca   = max(strategy_r2_pca,   key=strategy_r2_pca.get)
        logger.info(f"Best strategy (no PCA): {best_label_nopca}  "
                    f"R²={strategy_r2_nopca[best_label_nopca]:.4f}")
        logger.info(f"Best strategy (PCA-{PCA_COMPONENTS}): {best_label_pca}  "
                    f"R²={strategy_r2_pca[best_label_pca]:.4f}")

        fig_comp = _plot_strategy_comparison(strategy_r2_nopca, strategy_r2_pca, best_label_nopca)
        experiment_manager.save_plot(fig_comp, "nir_strategy_comparison.png")
        plt.close(fig_comp)

        # 3c. Main pipeline: H₂O sub → SNV → SG-1d → MinMax  (no PCA)
        logger.info("Applying main pipeline: H₂O sub → SNV → SG-1d → MinMax (no PCA)...")
        feat_proc = preprocessor.apply_water_subtraction(features_clean, water_ref_matrix)
        feat_proc = preprocessor.apply_snv(feat_proc)
        feat_proc = preprocessor.apply_savitzky_golay(
            feat_proc, window_length=SG_WINDOW, polyorder=SG_POLY, deriv=1
        )
        feat_proc = preprocessor.normalize_features(feat_proc, scaler_type="minmax", fit=True)
        targets_proc = preprocessor.apply_target_transformation(targets_clean, log_transform=False)

        logger.info(f"Final feature matrix: {feat_proc.shape} | Targets: {targets_proc.shape}")

        data_persistence.save_dataset(
            feat_proc, targets_proc,
            dataset_name="nir_preprocessed_data",
            metadata={
                "description": "NIR: H₂O sub + SNV + SG-1d + MinMax (no PCA)",
                "sg_window": SG_WINDOW,
                "sg_poly": SG_POLY,
                "water_subtraction": "week-matched",
            },
        )

        X = feat_proc.values
        y = targets_proc.values
        target_names = targets_proc.columns.tolist()

        # ── 4. HOLD-OUT EVALUATION ────────────────────────────────────────────
        logger.info("=== Hold-out Model Evaluation ===")

        # CNN uses Conv1d over 375 spectral channels — keep hidden_size small to
        # avoid OOM when operating on the full feature vector (not PCA-reduced).
        models = {
            "MLP":          MLPRegressor(config, hidden_size=128, epochs=300),
            "CNN":          CNNRegressor(config, hidden_size=64,  epochs=300),
            "RandomForest": RandomForestRegressor(config, n_estimators=100),
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
            stratify=False,
        )

        # Poisson criterion excluded: requires non-negative targets (some targets < 0).
        # Keep the param grid small — RandomizedSearchCV inside DoubleKFoldCV uses
        # n_jobs=-1, so peak memory = n_inner_folds × n_trees × n_features × n_targets.
        # With 375 features that grows large fast; stay within safe limits.
        rf_param_grid = {
            "n_estimators":      [10, 20, 30, 50],
            "max_depth":         [5, 8, 12, None],
            "min_samples_leaf":  [1, 3, 5],
            "min_samples_split": [2, 5, 10],
            "criterion":         ["friedman_mse", "squared_error"],
            "max_features":      ["sqrt", "log2"],
        }

        rf_estimator = holdout_results["RandomForest"]["model"].model
        cv_scores = cv.fit(X, y, rf_estimator, rf_param_grid, n_iter=24)

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

        # Top-12 targets scatter plot (CV)
        fig_top = _plot_top_targets(
            cv_y_true, cv_y_pred,
            target_names=target_names,
            n_top=12,
            title="RandomForest CV — Top-12 Targets by R² (NIR, H₂O sub + SNV + SG-1d)",
        )
        experiment_manager.save_plot(fig_top, "nir_cv_top12_targets.png", dpi=90)
        plt.close(fig_top)

        # Full scatter grid (all targets)
        fig_all = plotter.plot_prediction_vs_actual(
            cv_y_true, cv_y_pred,
            target_names=target_names,
            title="RandomForest CV — NIR → Chemical (all folds, H₂O sub + SNV + SG-1d)",
            show=False,
        )
        experiment_manager.save_plot(fig_all, "nir_cv_predictions_vs_actual.png", dpi=72)
        plt.close(fig_all)

        # RF feature importance (wavenumber channels)
        rf_importance = holdout_results["RandomForest"]["model"].get_feature_importance()
        if rf_importance is not None:
            fig_imp = plotter.plot_feature_importance(
                feat_proc.columns.tolist(),
                rf_importance.reshape(1, -1),
                target_names=["RF mean importance"],
                title="Random Forest Feature Importance (NIR wavenumber channels)",
                top_n=30,
                show=False,
            )
            experiment_manager.save_plot(fig_imp, "nir_rf_feature_importance.png")
            plt.close(fig_imp)

        # ── 7. SAVE RESULTS ───────────────────────────────────────────────────
        logger.info("=== Saving Results ===")

        all_results = {
            "holdout_results": {n: r["metrics"] for n, r in holdout_results.items()},
            "cross_validation": cv_scores,
            "preprocessing_comparison": {
                "no_pca": strategy_r2_nopca,
                "pca_20": strategy_r2_pca,
                "best_no_pca": best_label_nopca,
                "best_pca":    best_label_pca,
            },
            "data_info": {
                "n_samples":        len(X),
                "n_nir_features":   len(available_wn),
                "wavenumber_range": f"{available_wn[0]}–{available_wn[-1]} cm⁻¹",
                "wavenumber_step":  "16 cm⁻¹",
                "n_targets":        y.shape[1],
                "sg_window":        SG_WINDOW,
                "sg_poly":          SG_POLY,
                "sg_deriv":         1,
                "water_subtraction": "week-matched",
                "use_pca":          False,
                "target_names":     target_names,
                "dropped_targets":  dropped,
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
        cv_r2_per_target = cv_metrics["r2_score"]
        good_targets = [(target_names[i], float(cv_r2_per_target[i]))
                        for i in np.where(cv_r2_per_target > 0.1)[0]]
        good_targets.sort(key=lambda x: -x[1])

        logger.info(f"Experiment ID:     {experiment_id}")
        logger.info(f"NIR features:      {len(available_wn)} channels "
                    f"({available_wn[0]}–{available_wn[-1]} cm⁻¹, Δ=16 cm⁻¹)")
        logger.info(f"Samples:           {len(X)}")
        logger.info(f"Targets:           {y.shape[1]} (dropped {len(dropped)} low-coverage)")
        logger.info(f"Best preprocessing (no PCA):   {best_label_nopca}  "
                    f"R²={strategy_r2_nopca[best_label_nopca]:.4f}")
        logger.info(f"Best preprocessing (PCA-{PCA_COMPONENTS}): {best_label_pca}  "
                    f"R²={strategy_r2_pca[best_label_pca]:.4f}")
        logger.info(f"Best hold-out model: {best_ho}  "
                    f"R²={np.mean(holdout_results[best_ho]['metrics']['r2_score']):.4f}")
        logger.info(f"CV (RF) mean R²:   {np.mean(test_r2):.4f} ± {np.std(test_r2):.4f}")
        logger.info(f"CV aggregate R²:   {np.mean(cv_r2_per_target):.4f}")
        logger.info(f"Targets with CV R² > 0.10 ({len(good_targets)}):")
        for tname, tr2 in good_targets:
            logger.info(f"  {tname}: {tr2:.4f}")

    except Exception as e:
        logger.error(f"Error: {e}")
        experiment_manager.end_experiment(status="failed")
        raise
    else:
        experiment_manager.end_experiment(status="completed")
        logger.info("NIR regression analysis completed successfully!")


if __name__ == "__main__":
    main()
