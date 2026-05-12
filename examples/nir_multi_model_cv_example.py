"""
NIR multi-model cross-validation example
=========================================
NIR spectra (1 mm path, 4000–9984 cm⁻¹, Δ=16 cm⁻¹, 375 features)
→ multi-target chemical analysis prediction (up to 67 targets).

Main preprocessing pipeline:
    H₂O sub → SNV → SG 1st-deriv → MinMax  (no PCA)

Cross-validation is driven by the **CV_MODELS** list defined near the top
of main(). Each entry controls:

    name        label used in logs, filenames, and plots
    estimator   sklearn estimator or SklearnMLPWrapper instance
    param_grid  None  → simple K-Fold with fixed hyperparameters (fast)
                dict  → double K-Fold + RandomizedSearchCV HPO (thorough)
    n_iter      number of HPO candidates (ignored when param_grid is None)

To add a model, append an entry to CV_MODELS.
To remove a model, delete its entry.
To enable MLP HPO, change its param_grid from None to the commented example.

Notes on memory:
    The custom _run_cv() uses n_jobs=1 to avoid OOM when X is wide
    (375 features × 67 targets × parallel inner folds). The library's
    DoubleKFoldCV hardcodes n_jobs=-1, which is why we use our own runner.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import logging
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.decomposition import PCA as SKPCA
from sklearn.ensemble import RandomForestRegressor as SKRF
from sklearn.metrics import r2_score as sk_r2
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split
from sklearn.preprocessing import MinMaxScaler

from uvvislib import (
    Config, setup_logging, DataLoader, Preprocessor,
    MLPRegressor, RandomForestRegressor,
    compute_evaluation,
    Plotter,
    ModelManager, ExperimentManager, DataPersistence,
)
from uvvislib.models.mlp import MLP as _TorchMLP

# ── Spectral and preprocessing constants ─────────────────────────────────────
SG_WINDOW      = 11    # Savitzky-Golay window length (spectral points)
SG_POLY        = 3     # Savitzky-Golay polynomial order
PCA_COMPONENTS = 20    # components used in the strategy-comparison branch only
H2O_FILE       = "NIR_1mm_completo_ARCHA_h20.csv"

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


# ── sklearn-compatible MLP wrapper ────────────────────────────────────────────

class SklearnMLPWrapper(BaseEstimator, RegressorMixin):
    """
    sklearn-compatible multi-layer PyTorch MLP for use in CV_MODELS.

    hidden_layers is a tuple of hidden-unit counts, one entry per layer.
    Examples:
        (128,)           → single hidden layer  (shallow)
        (256, 128)       → two hidden layers
        (256, 128, 64)   → three hidden layers  (deep)

    get_params / set_params are inherited from BaseEstimator (introspects
    __init__ automatically), so clone() and RandomizedSearchCV work without
    any extra code. hidden_layers must be a tuple (not a list) because
    sklearn requires hashable parameter values.
    """

    def __init__(
        self,
        hidden_layers: tuple = (128,),
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        dropout_rate: float = 0.0,
        epochs: int = 150,
        activation: str = "relu",
    ):
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.weight_decay  = weight_decay
        self.dropout_rate  = dropout_rate
        self.epochs        = epochs
        self.activation    = activation

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        n_in, n_out = X.shape[1], y.shape[1]
        self._n_out  = n_out
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        _act = {"sigmoid": nn.Sigmoid, "relu": nn.ReLU, "tanh": nn.Tanh}
        act_cls = _act[self.activation]

        # Build a fully connected stack: n_in → h0 → h1 → ... → n_out
        layers: list[nn.Module] = []
        in_size = n_in
        for h in self.hidden_layers:
            layers.append(nn.Linear(in_size, h))
            layers.append(act_cls())
            if self.dropout_rate > 0:
                layers.append(nn.Dropout(self.dropout_rate))
            in_size = h
        layers.append(nn.Linear(in_size, n_out))
        self._net = nn.Sequential(*layers).to(self._device)

        opt  = optim.Adam(self._net.parameters(),
                          lr=self.learning_rate, weight_decay=self.weight_decay)
        crit = nn.MSELoss()
        X_t  = torch.from_numpy(X).to(self._device)
        y_t  = torch.from_numpy(y).to(self._device)

        self._net.train()
        for _ in range(self.epochs):
            opt.zero_grad()
            crit(self._net(X_t), y_t).backward()
            opt.step()
        return self

    def predict(self, X):
        if not hasattr(self, "_net"):
            raise ValueError("Call fit() before predict().")
        X = np.asarray(X, dtype=np.float32)
        self._net.eval()
        with torch.no_grad():
            out = self._net(torch.from_numpy(X).to(self._device))
        pred = np.array(out.cpu().tolist())
        return pred if self._n_out > 1 else pred.flatten()


# ── Generalized CV runner ─────────────────────────────────────────────────────

def _run_cv(
    name: str,
    estimator,
    X: np.ndarray,
    y: np.ndarray,
    outer_splits: int = 5,
    inner_splits: int = 4,
    param_grid: dict | None = None,
    n_iter: int = 20,
    random_state: int = 42,
    logger=None,
) -> dict:
    """
    Outer K-Fold cross-validation with optional inner HPO.

    param_grid is not None → double K-Fold: outer evaluation + inner
                              RandomizedSearchCV (n_jobs=1 to control memory)
    param_grid is None     → simple K-Fold with fixed estimator params

    Returns a dict compatible with ExperimentManager.save_cv_results_csv:
        test_scores   list[dict] — per-fold metrics from compute_evaluation()
        predictions   list[ndarray]
        true_values   list[ndarray]
        best_params   list[dict]  — empty dict for no-HPO runs
        train_scores  list[dict]  — placeholder for API compatibility
        val_scores    list[dict]  — placeholder for API compatibility
    """
    outer_cv = KFold(n_splits=outer_splits, shuffle=True, random_state=random_state)
    results: dict = {
        "test_scores": [], "best_params": [],
        "predictions": [], "true_values": [],
        "train_scores": [], "val_scores": [],
    }

    for fold, (tr_idx, te_idx) in enumerate(outer_cv.split(X)):
        if logger:
            logger.info(f"  [{name}] Outer fold {fold + 1}/{outer_splits}")
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        if param_grid:
            inner_cv = KFold(
                n_splits=inner_splits, shuffle=True, random_state=random_state
            )
            search = RandomizedSearchCV(
                estimator, param_grid,
                n_iter=n_iter, cv=inner_cv,
                scoring="neg_mean_squared_error", refit=True,
                n_jobs=1,           # predictable memory footprint with wide X
                random_state=random_state,
            )
            search.fit(X_tr, y_tr)
            best_params = search.best_params_
            # refit=True means best_estimator_ is already fitted on full X_tr
            final_model = search.best_estimator_
        else:
            final_model = clone(estimator)
            final_model.fit(X_tr, y_tr)
            best_params = {}

        y_pred = final_model.predict(X_te)
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)
        if y_te.ndim == 1:
            y_te = y_te.reshape(-1, 1)

        fold_metrics = compute_evaluation(y_te, y_pred)
        results["test_scores"].append(fold_metrics)
        results["predictions"].append(y_pred)
        results["true_values"].append(y_te)
        results["best_params"].append(best_params)
        results["train_scores"].append({})
        results["val_scores"].append({})

        r2_m, rmse_m = float(np.mean(fold_metrics["r2_score"])), float(np.mean(fold_metrics["rmse"]))
        if logger:
            suffix = f"  best={best_params}" if best_params else ""
            logger.info(f"  [{name}] Fold {fold + 1} — R²={r2_m:.4f}  RMSE={rmse_m:.4f}{suffix}")

    return results


# ── Preprocessing helpers ─────────────────────────────────────────────────────

def _build_water_ref_matrix(
    features_clean: pd.DataFrame,
    combined_data: pd.DataFrame,
    h2o_df: pd.DataFrame,
    available_wn: list,
) -> np.ndarray:
    """Build (n_samples, n_wavenumbers) week-matched water reference matrix.

    Each sample's water reference is the mean spectrum for its week.
    Weeks with no water measurement fall back to the global column mean.
    """
    global_mean = h2o_df[available_wn].mean()
    h2o_by_week = h2o_df.groupby("WEEK")[available_wn].mean().fillna(global_mean)
    fallback    = global_mean.values
    weeks       = combined_data.loc[features_clean.index, "Week"]
    rows = [
        h2o_by_week.loc[w].values if w in h2o_by_week.index else fallback
        for w in weeks
    ]
    return np.stack(rows)


def _apply_strategy(
    features_df: pd.DataFrame,
    strategy: str,
    pre: Preprocessor,
    water_ref: np.ndarray | None = None,
) -> np.ndarray:
    """Apply a named preprocessing strategy; return MinMax-scaled ndarray."""
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
    """80/20 hold-out mean R² with RF(50 trees). Near-constant targets excluded."""
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
        if col.std() < max(1e-6, 0.01 * (abs(col.mean()) + 1e-10)):
            continue
        r2s.append(np.clip(sk_r2(col, y_pred[:, j]), -1.0, 1.0))
    return float(np.mean(r2s)) if r2s else 0.0


# ── Plot helpers ──────────────────────────────────────────────────────────────

def _plot_preprocessing_effects(
    features_clean: pd.DataFrame,
    features_water: pd.DataFrame,
    pre: Preprocessor,
    wavenums: np.ndarray,
    n_samples: int = 3,
) -> plt.Figure:
    """7-panel figure showing NIR preprocessing stages for a few samples."""
    sample_idx = np.linspace(0, len(features_clean) - 1, n_samples, dtype=int)
    colors     = ["steelblue", "darkorange", "forestgreen"]
    snv_raw    = pre.apply_snv(features_clean)
    snv_water  = pre.apply_snv(features_water)
    stages = [
        ("Raw NIR",                 features_clean.values),
        ("H₂O subtracted",          features_water.values),
        ("SNV (no H₂O sub)",        snv_raw.values),
        ("H₂O sub + SNV",           snv_water.values),
        ("H₂O+SNV + SG smooth",     pre.apply_savitzky_golay(snv_water, SG_WINDOW, SG_POLY, 0).values),
        ("H₂O+SNV + SG 1st-deriv",  pre.apply_savitzky_golay(snv_water, SG_WINDOW, SG_POLY, 1).values),
        ("H₂O+SNV + SG 2nd-deriv",  pre.apply_savitzky_golay(snv_water, SG_WINDOW, SG_POLY, 2).values),
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
    """Side-by-side bar charts comparing preprocessing strategies."""
    labels   = list(strategy_r2_nopca.keys())
    r2_nopca = [strategy_r2_nopca[l] for l in labels]
    r2_pca   = [strategy_r2_pca[l]   for l in labels]
    fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=False)
    for ax, r2_vals, suffix in zip(
        axes, [r2_nopca, r2_pca],
        ["No PCA (375 features)", f"PCA-{PCA_COMPONENTS}"],
    ):
        bar_colors = ["steelblue"] * len(labels)
        if best_label in labels:
            bar_colors[labels.index(best_label)] = "darkorange"
        bars = ax.bar(range(len(labels)), r2_vals, color=bar_colors, edgecolor="black")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel("Mean R² (80/20 hold-out, RF)")
        ax.set_title(f"NIR Strategy Comparison — {suffix}")
        pad = max(abs(max(r2_vals) - min(r2_vals)) * 0.15, 0.05)
        ax.set_ylim(min(r2_vals) - pad, max(r2_vals) + pad)
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        for bar, v in zip(bars, r2_vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    v + (0.003 if v >= 0 else -0.015),
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    return fig


def _plot_top_targets(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: list,
    n_top: int = 12,
    title: str = "",
) -> plt.Figure:
    """Scatter grid for the top-N targets ranked by R²."""
    from uvvislib.evaluation.metrics import r2_score as lib_r2
    r2_per_target = lib_r2(y_pred, y_true)
    order  = np.argsort(r2_per_target)[::-1][:n_top]
    ncols  = 4
    nrows  = int(np.ceil(n_top / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3.5))
    axes = axes.flatten()
    fig.suptitle(title, fontsize=11)
    for ax, idx in zip(axes, order):
        yt, yp = y_true[:, idx], y_pred[:, idx]
        r2     = r2_per_target[idx]
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


def _plot_cv_model_comparison(cv_results_per_model: dict) -> plt.Figure:
    """Bar chart comparing mean CV R² (± std across outer folds) for all models."""
    names  = list(cv_results_per_model.keys())
    r2s    = [cv_results_per_model[n]["mean_r2"] for n in names]
    stds   = [cv_results_per_model[n]["std_r2"]  for n in names]
    colors = plt.cm.Set2(np.linspace(0, 0.8, len(names)))
    fig, ax = plt.subplots(figsize=(max(6, 2.5 * len(names)), 4))
    bars = ax.bar(names, r2s, yerr=stds, color=colors, edgecolor="black",
                  capsize=6, alpha=0.85)
    ax.set_ylabel("Mean CV R² (across outer folds and targets)")
    ax.set_title("Cross-Validation Model Comparison — NIR (H₂O+SNV+SG-1d)")
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    for bar, v, e in zip(bars, r2s, stds):
        ax.text(bar.get_x() + bar.get_width() / 2,
                max(v + e + 0.005, 0.005),
                f"{v:.3f}±{e:.3f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    setup_logging(log_level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("Starting NIR multi-model CV example")

    # ── CONFIG ────────────────────────────────────────────────────────────────
    config = Config(
        data_path="./Data/",
        uv_vis_data_file="NIR_1mm_completo_ARCHA.csv",
        chemical_data_file="Analisi_chimiche_complete_ARCHA.csv",
        wavelength_start=4000,
        wavelength_end=10000,
        wavelength_step=16,
        apply_smoothing=False,   # Savitzky-Golay is applied manually below
        apply_pca=False,         # PCA is used only in the strategy-comparison branch
        pca_components=PCA_COMPONENTS,
        target_variables=ALL_TARGETS,
        log_target=False,
        k_fold_splits=5,
        random_state=42,
    )

    experiment_manager = ExperimentManager(config)
    model_manager      = ModelManager(config)
    data_persistence   = DataPersistence(config)

    experiment_id = experiment_manager.start_experiment(
        experiment_name="nir_multi_model_cv",
        description="NIR: H₂O+SNV+SG-1d pipeline; CV for MLP and RandomForest",
        tags=["nir", "regression", "water-subtraction", "savitzky-golay", "multi-model-cv"],
    )

    try:
        # ── 1. DATA LOADING ───────────────────────────────────────────────────
        logger.info("=== Data Loading ===")

        data_loader = DataLoader(config)
        data_loader.load_uv_vis_data()
        data_loader.load_chemical_data()
        data_loader.combine_data(["Week", "Gathering_Point"])

        nir_wavenumbers = config.feature_wavelengths
        available_wn    = [w for w in nir_wavenumbers if w in data_loader.combined_data.columns]
        features_df     = data_loader.combined_data[available_wn].copy()
        logger.info(f"NIR features: {features_df.shape[1]} channels "
                    f"({available_wn[0]}–{available_wn[-1]} cm⁻¹, Δ=16 cm⁻¹)")

        available_targets = [t for t in ALL_TARGETS if t in data_loader.combined_data.columns]
        targets_df        = data_loader.combined_data[available_targets].copy()
        for col in targets_df.columns:
            if not pd.api.types.is_numeric_dtype(targets_df[col]):
                targets_df[col] = pd.to_numeric(
                    targets_df[col].astype(str).str.strip().str.replace(",", ".", regex=False),
                    errors="coerce",
                )
        logger.info(f"Loaded: {len(features_df)} samples, {len(available_targets)} targets")

        # ── 1b. WATER REFERENCE ───────────────────────────────────────────────
        logger.info("=== Loading Water Reference ===")

        h2o_df    = pd.read_csv(Path(config.data_path) / H2O_FILE, sep=";", decimal=",")
        h2o_avail = [w for w in available_wn if w in h2o_df.columns]
        logger.info(f"Water reference: {len(h2o_df)} measurements, "
                    f"{h2o_df.groupby('WEEK').ngroups} unique weeks")

        # ── 2. TARGET COVERAGE FILTERING ──────────────────────────────────────
        logger.info("=== Target Coverage Filtering ===")

        min_coverage  = 0.80
        coverage      = targets_df.notna().mean()
        valid_targets = coverage[coverage >= min_coverage].index.tolist()
        dropped       = [t for t in available_targets if t not in valid_targets]
        targets_df    = targets_df[valid_targets]
        logger.info(f"Retained {len(valid_targets)}/{len(available_targets)} targets "
                    f"(≥ {min_coverage * 100:.0f}% non-null coverage)")
        if dropped:
            logger.info(f"Dropped (low coverage): {dropped}")

        # ── 3. CLEANING AND MAIN PREPROCESSING ───────────────────────────────
        logger.info("=== NIR Preprocessing ===")

        preprocessor = Preprocessor(config)
        features_clean, targets_clean = preprocessor.clean_data(
            features_df, targets_df, remove_nan_targets=True
        )
        logger.info(f"Samples after cleaning: {len(features_clean)}")

        water_ref_matrix = _build_water_ref_matrix(
            features_clean, data_loader.combined_data, h2o_df, h2o_avail
        )
        features_water = preprocessor.apply_water_subtraction(features_clean, water_ref_matrix)
        logger.info(f"Water subtraction: mean abs {features_clean.values.mean():.4f} → "
                    f"{features_water.values.mean():.4f}")

        wavenums = np.array([int(c) for c in features_clean.columns])

        # 3a. Visualise preprocessing stages (7 panels)
        logger.info("Saving preprocessing stage visualisation...")
        fig_preproc = _plot_preprocessing_effects(
            features_clean, features_water, preprocessor, wavenums
        )
        experiment_manager.save_plot(fig_preproc, "nir_preprocessing_stages.png", dpi=90)
        plt.close(fig_preproc)

        # 3b. Quick preprocessing strategy comparison (RF, 80/20 hold-out)
        logger.info("Comparing preprocessing strategies (RF, 80/20 hold-out)...")
        y_clean = targets_clean.values
        strategies = {
            "A: Raw":            "raw",
            "B: SNV":            "snv",
            "C: SNV+SG-1d":      "snv_sg1",
            "D: SNV+SG-2d":      "snv_sg2",
            "E: H₂O":            "h2o",
            "F: H₂O+SNV":       "h2o_snv",
            "G: H₂O+SNV+SG-1d": "h2o_snv_sg1",
            "H: H₂O+SNV+SG-2d": "h2o_snv_sg2",
        }
        strategy_r2_nopca: dict[str, float] = {}
        strategy_r2_pca:   dict[str, float] = {}
        for label, key in strategies.items():
            X_s      = _apply_strategy(features_clean, key, preprocessor, water_ref_matrix)
            r2_nopca = _quick_r2(X_s, y_clean, n_pca=0,             seed=config.random_state)
            r2_pca   = _quick_r2(X_s, y_clean, n_pca=PCA_COMPONENTS, seed=config.random_state)
            strategy_r2_nopca[label] = r2_nopca
            strategy_r2_pca[label]   = r2_pca
            logger.info(f"  {label}: no-PCA R²={r2_nopca:.4f}  "
                        f"PCA-{PCA_COMPONENTS} R²={r2_pca:.4f}")

        best_label_nopca = max(strategy_r2_nopca, key=strategy_r2_nopca.get)
        best_label_pca   = max(strategy_r2_pca,   key=strategy_r2_pca.get)
        logger.info(f"Best strategy (no PCA): {best_label_nopca}  "
                    f"R²={strategy_r2_nopca[best_label_nopca]:.4f}")
        logger.info(f"Best strategy (PCA-{PCA_COMPONENTS}): {best_label_pca}  "
                    f"R²={strategy_r2_pca[best_label_pca]:.4f}")

        fig_comp = _plot_strategy_comparison(strategy_r2_nopca, strategy_r2_pca, best_label_nopca)
        experiment_manager.save_plot(fig_comp, "nir_strategy_comparison.png")
        plt.close(fig_comp)

        # 3c. Apply main pipeline: H₂O sub → SNV → SG 1st-deriv → MinMax
        logger.info("Applying main pipeline: H₂O sub → SNV → SG-1d → MinMax...")
        feat_proc = preprocessor.apply_water_subtraction(features_clean, water_ref_matrix)
        feat_proc = preprocessor.apply_snv(feat_proc)
        feat_proc = preprocessor.apply_savitzky_golay(feat_proc, SG_WINDOW, SG_POLY, deriv=1)
        feat_proc = preprocessor.normalize_features(feat_proc, scaler_type="minmax", fit=True)
        targets_proc = preprocessor.apply_target_transformation(targets_clean, log_transform=False)
        logger.info(f"Final feature matrix: {feat_proc.shape}  Targets: {targets_proc.shape}")

        data_persistence.save_dataset(
            feat_proc, targets_proc,
            dataset_name="nir_preprocessed_data",
            metadata={
                "pipeline": "H2O_sub+SNV+SG1+MinMax",
                "sg_window": SG_WINDOW, "sg_poly": SG_POLY,
                "water_subtraction": "week-matched",
            },
        )

        X            = feat_proc.values
        y            = targets_proc.values
        target_names = targets_proc.columns.tolist()

        # ── 4. HOLD-OUT EVALUATION (quick sanity check) ───────────────────────
        logger.info("=== Hold-out Evaluation ===")

        split_idx          = int(0.8 * len(X))
        X_tr_ho, X_te_ho   = X[:split_idx], X[split_idx:]
        y_tr_ho, y_te_ho   = y[:split_idx], y[split_idx:]

        ho_models = {
            "MLP":          MLPRegressor(config, hidden_size=128, epochs=300),
            "RandomForest": RandomForestRegressor(config, n_estimators=100),
        }
        holdout_results: dict = {}
        for m_name, model in ho_models.items():
            logger.info(f"Training {m_name} (hold-out)...")
            model.fit(X_tr_ho, y_tr_ho)
            y_pred  = model.predict(X_te_ho)
            metrics = compute_evaluation(y_te_ho, y_pred)
            holdout_results[m_name] = {
                "model": model, "metrics": metrics,
                "predictions": y_pred, "true_values": y_te_ho,
            }
            logger.info(f"  {m_name} — R²={np.mean(metrics['r2_score']):.4f}  "
                        f"RMSE={np.mean(metrics['rmse']):.4f}")
            experiment_manager.save_evaluation_csv(
                metrics, target_names=target_names,
                filename=f"{m_name.lower()}_holdout_evaluation.csv",
            )
            model_manager.save_model(model, f"{m_name.lower()}_model",
                                     experiment_name=experiment_id)

        # ── 5. CROSS-VALIDATION ───────────────────────────────────────────────
        #
        # ══════════════════════════════════════════════════════════════════════
        # CV_MODELS — edit this list to control which models run CV.
        #
        # Each dict entry:
        #   name        label used in logs, filenames, and plots
        #   estimator   sklearn estimator or SklearnMLPWrapper instance
        #   param_grid  None  → simple K-Fold with fixed hyperparameters
        #               dict  → double K-Fold + RandomizedSearchCV HPO
        #   n_iter      HPO candidates per fold (ignored when param_grid=None)
        #
        # To enable MLP HPO, replace "param_grid": None with:
        #   "param_grid": {
        #       "hidden_size":   [64, 128, 256],
        #       "learning_rate": [1e-3, 5e-4, 1e-4],
        #       "weight_decay":  [0.0, 1e-4, 1e-3],
        #       "dropout_rate":  [0.0, 0.1, 0.2],
        #       "epochs":        [100, 150, 200],
        #       "activation":    ["relu", "tanh"],
        #   }
        # ══════════════════════════════════════════════════════════════════════

        CV_MODELS = [
            {
                "name": "RandomForest",
                "estimator": SKRF(random_state=config.random_state),
                "param_grid": {
                    "n_estimators":      [10, 20, 30, 50],
                    "max_depth":         [5, 8, 12, None],
                    "min_samples_leaf":  [1, 3, 5],
                    "min_samples_split": [2, 5, 10],
                    "criterion":         ["friedman_mse", "squared_error"],
                    "max_features":      ["sqrt", "log2"],
                },
                "n_iter": 20,
            },
            {
                "name": "MLP",
                "estimator": SklearnMLPWrapper(),
                "param_grid": {
                    # Architecture: tuples encode layer sizes (shallow → deep)
                    "hidden_layers": [
                        (128,),
                        (256,),
                        (256, 128),
                        (256, 128, 64),
                        (512, 256, 128),
                        (512, 256, 128, 64),
                    ],
                    "learning_rate": [1e-3, 5e-4, 1e-4],
                    "weight_decay":  [0.0, 1e-4, 1e-3],
                    "dropout_rate":  [0.0, 0.1, 0.2],
                    "epochs":        [200, 400, 600],
                    "activation":    ["relu", "tanh"],
                },
                "n_iter": 20,
            },
        ]

        # ── Run CV for each model in CV_MODELS ────────────────────────────────
        logger.info("=== Cross-Validation ===")
        cv_results_per_model: dict = {}

        for model_cfg in CV_MODELS:
            m_name      = model_cfg["name"]
            m_estimator = model_cfg["estimator"]
            m_grid      = model_cfg.get("param_grid")
            m_n_iter    = model_cfg.get("n_iter", 20)
            hpo_label   = f"double K-Fold, n_iter={m_n_iter}" if m_grid else "simple K-Fold"

            logger.info(f"--- CV: {m_name} [{hpo_label}] ---")

            cv_scores = _run_cv(
                name=m_name,
                estimator=m_estimator,
                X=X, y=y,
                outer_splits=config.k_fold_splits,
                inner_splits=4,
                param_grid=m_grid,
                n_iter=m_n_iter,
                random_state=config.random_state,
                logger=logger,
            )

            # Aggregate fold-level means
            test_r2s   = np.array([np.mean(s["r2_score"]) for s in cv_scores["test_scores"]])
            test_rmses = np.array([np.mean(s["rmse"])      for s in cv_scores["test_scores"]])
            mean_r2    = float(np.mean(test_r2s))
            std_r2     = float(np.std(test_r2s))
            mean_rmse  = float(np.mean(test_rmses))
            std_rmse   = float(np.std(test_rmses))

            logger.info(f"  {m_name} CV R²:   {mean_r2:.4f} ± {std_r2:.4f}")
            logger.info(f"  {m_name} CV RMSE: {mean_rmse:.4f} ± {std_rmse:.4f}")

            # Per-target CSV (mean ± std across folds)
            experiment_manager.save_cv_results_csv(
                cv_scores, target_names=target_names,
                filename=f"cv_results_{m_name.lower()}.csv",
            )

            # All-folds predictions concatenated
            cv_y_pred = np.vstack(cv_scores["predictions"])
            cv_y_true = np.vstack(cv_scores["true_values"])
            experiment_manager.save_predictions(
                cv_y_pred, cv_y_true, target_names=target_names,
                filename=f"cv_predictions_{m_name.lower()}.csv",
            )

            # Aggregate evaluation across all folds
            agg_metrics = compute_evaluation(cv_y_true, cv_y_pred)
            experiment_manager.save_evaluation_csv(
                agg_metrics, target_names=target_names,
                filename=f"cv_evaluation_{m_name.lower()}.csv",
            )

            cv_results_per_model[m_name] = {
                "cv_scores":   cv_scores,
                "mean_r2":     mean_r2,
                "std_r2":      std_r2,
                "mean_rmse":   mean_rmse,
                "std_rmse":    std_rmse,
                "agg_metrics": agg_metrics,
                "y_pred":      cv_y_pred,
                "y_true":      cv_y_true,
            }

        # ── 6. VISUALIZATION ──────────────────────────────────────────────────
        logger.info("=== Visualization ===")

        plotter = Plotter()

        for m_name, m_data in cv_results_per_model.items():
            # Scatter grid — top-12 targets by R²
            fig_top = _plot_top_targets(
                m_data["y_true"], m_data["y_pred"],
                target_names=target_names, n_top=12,
                title=f"{m_name} CV — Top-12 Targets by R² (NIR, H₂O+SNV+SG-1d)",
            )
            experiment_manager.save_plot(
                fig_top, f"nir_cv_top12_{m_name.lower()}.png", dpi=90
            )
            plt.close(fig_top)

            # Full prediction grid (all targets)
            fig_all = plotter.plot_prediction_vs_actual(
                m_data["y_true"], m_data["y_pred"],
                target_names=target_names,
                title=f"{m_name} CV — NIR → Chemical (all folds, H₂O+SNV+SG-1d)",
                show=False,
            )
            experiment_manager.save_plot(
                fig_all, f"nir_cv_predictions_{m_name.lower()}.png", dpi=72
            )
            plt.close(fig_all)

        # Cross-model comparison bar chart
        fig_cmp = _plot_cv_model_comparison(cv_results_per_model)
        experiment_manager.save_plot(fig_cmp, "nir_cv_model_comparison.png")
        plt.close(fig_cmp)

        # RF feature importance (from hold-out model)
        if "RandomForest" in holdout_results:
            rf_imp = holdout_results["RandomForest"]["model"].get_feature_importance()
            if rf_imp is not None:
                fig_imp = plotter.plot_feature_importance(
                    feat_proc.columns.tolist(), rf_imp.reshape(1, -1),
                    target_names=["RF mean importance"],
                    title="Random Forest Feature Importance (NIR wavenumber channels)",
                    top_n=30, show=False,
                )
                experiment_manager.save_plot(fig_imp, "nir_rf_feature_importance.png")
                plt.close(fig_imp)

        # ── 7. SAVE RESULTS ───────────────────────────────────────────────────
        logger.info("=== Saving Results ===")

        all_results = {
            "holdout_results": {n: r["metrics"] for n, r in holdout_results.items()},
            "cross_validation": {
                n: {
                    "mean_r2":   d["mean_r2"],  "std_r2":   d["std_r2"],
                    "mean_rmse": d["mean_rmse"], "std_rmse": d["std_rmse"],
                }
                for n, d in cv_results_per_model.items()
            },
            "preprocessing_comparison": {
                "no_pca":     strategy_r2_nopca,
                "pca_20":     strategy_r2_pca,
                "best_no_pca": best_label_nopca,
                "best_pca":    best_label_pca,
            },
            "data_info": {
                "n_samples":         len(X),
                "n_nir_features":    len(available_wn),
                "wavenumber_range":  f"{available_wn[0]}–{available_wn[-1]} cm⁻¹",
                "n_targets":         y.shape[1],
                "sg_window":         SG_WINDOW,
                "sg_poly":           SG_POLY,
                "sg_deriv":          1,
                "water_subtraction": "week-matched",
                "use_pca":           False,
                "target_names":      target_names,
                "dropped_targets":   dropped,
                "cv_models":         [m["name"] for m in CV_MODELS],
            },
        }
        experiment_manager.save_results(all_results, "nir_multi_model_cv_results.json")
        logger.info("All results saved successfully")

        # ── 8. SUMMARY ────────────────────────────────────────────────────────
        logger.info("=== Summary ===")
        logger.info(f"Experiment ID:      {experiment_id}")
        logger.info(f"Samples:            {len(X)}")
        logger.info(f"NIR features:       {len(available_wn)} channels "
                    f"({available_wn[0]}–{available_wn[-1]} cm⁻¹)")
        logger.info(f"Targets:            {y.shape[1]} "
                    f"(dropped {len(dropped)} low-coverage)")
        logger.info(f"Best preprocessing: {best_label_nopca}  "
                    f"(R²={strategy_r2_nopca[best_label_nopca]:.4f})")
        logger.info("")
        logger.info("Cross-validation results:")
        for m_name, m_data in cv_results_per_model.items():
            r2_per_target = m_data["agg_metrics"]["r2_score"]
            good_targets  = [
                (target_names[i], float(r2_per_target[i]))
                for i in np.where(r2_per_target > 0.10)[0]
            ]
            good_targets.sort(key=lambda x: -x[1])
            logger.info(
                f"  {m_name:15s} R²={m_data['mean_r2']:.4f}±{m_data['std_r2']:.4f}  "
                f"RMSE={m_data['mean_rmse']:.4f}  "
                f"Targets with R²>0.10: {len(good_targets)}"
            )
            for tname, tr2 in good_targets:
                logger.info(f"      {tname}: {tr2:.4f}")

    except Exception as e:
        logger.error(f"Error: {e}")
        experiment_manager.end_experiment(status="failed")
        raise
    else:
        experiment_manager.end_experiment(status="completed")
        logger.info("NIR multi-model CV example completed successfully!")


if __name__ == "__main__":
    main()
