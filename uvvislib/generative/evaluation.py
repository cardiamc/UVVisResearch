"""
Evaluation metrics for comparing real and synthetic UV-Vis spectra.

Three metric groups, each callable independently:

* ``spectral_fidelity_metrics`` — MAE, RMSE, Pearson r, Wasserstein,
  peak-preservation ratio.
* ``pca_metrics`` — explained-variance difference, loading-vector cosine
  similarity, Wasserstein on PC1/PC2 score distributions.
* ``downstream_utility`` — train on real-only / synth-only / combined;
  evaluate on a held-out real test split.
* ``reliability_score`` — DTW-based real-vs-synthetic fidelity measure.
* ``compare_generators`` — run all groups for a dict of fitted generators;
  return a tidy ``pd.DataFrame``.
"""

from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr, wasserstein_distance
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from .base import BaseGenerativeModel


# ---------------------------------------------------------------------------
# Spectral fidelity
# ---------------------------------------------------------------------------

def spectral_fidelity_metrics(
    real: np.ndarray,
    synth: np.ndarray,
) -> Dict[str, float]:
    """
    Compute pointwise and distributional fidelity between mean spectra.

    Parameters
    ----------
    real : ndarray of shape (n_real, n_wavelengths)
    synth : ndarray of shape (n_synth, n_wavelengths)

    Returns
    -------
    dict with keys:
        ``mae``, ``rmse``, ``pearson_r``, ``wasserstein``,
        ``peak_preservation``, ``cod_correlation_profile_similarity``
    """
    real_mean = np.mean(real, axis=0)
    synth_mean = np.mean(synth, axis=0)

    mae = float(mean_absolute_error(real_mean, synth_mean))
    rmse = float(np.sqrt(mean_squared_error(real_mean, synth_mean)))
    pearson_r = float(pearsonr(real_mean, synth_mean)[0])
    wass = float(wasserstein_distance(real.ravel(), synth.ravel()))

    real_peaks, _ = find_peaks(real_mean, height=float(np.mean(real_mean)))
    synth_peaks, _ = find_peaks(synth_mean, height=float(np.mean(synth_mean)))
    if len(real_peaks) == 0:
        peak_preservation = float("nan")
    else:
        peak_preservation = len(set(real_peaks) & set(synth_peaks)) / len(real_peaks)

    return {
        "mae": mae,
        "rmse": rmse,
        "pearson_r": pearson_r,
        "wasserstein": wass,
        "peak_preservation": float(peak_preservation),
    }


# ---------------------------------------------------------------------------
# PCA-based distributional metrics
# ---------------------------------------------------------------------------

def pca_metrics(
    real: np.ndarray,
    synth: np.ndarray,
    n_components: int = 2,
) -> Dict[str, float]:
    """
    Compare PCA structure of real and synthetic spectra.

    Parameters
    ----------
    real : ndarray of shape (n_real, n_wavelengths)
    synth : ndarray of shape (n_synth, n_wavelengths)
    n_components : int
        Number of PCs to use (default 2).

    Returns
    -------
    dict with keys:
        ``explained_var_diff_mean``,
        ``loading_similarity_mean`` (cosine sim, higher = better),
        ``pc1_wasserstein``, ``pc2_wasserstein``,
        and per-PC ``loading_similarity_pc{i+1}``
    """
    pca_real = PCA(n_components=n_components).fit(real)
    pca_synth = PCA(n_components=n_components).fit(synth)

    real_proj = pca_real.transform(real)
    synth_proj = pca_synth.transform(synth)

    evr_diff = np.abs(
        pca_real.explained_variance_ratio_ - pca_synth.explained_variance_ratio_
    )

    loading_sims = [
        1.0 - float(cosine(pca_real.components_[i], pca_synth.components_[i]))
        for i in range(n_components)
    ]

    metrics: Dict[str, float] = {
        "explained_var_diff_mean": float(evr_diff.mean()),
        "loading_similarity_mean": float(np.mean(loading_sims)),
    }
    for i, sim in enumerate(loading_sims):
        metrics[f"loading_similarity_pc{i + 1}"] = sim

    metrics["pc1_wasserstein"] = float(
        wasserstein_distance(real_proj[:, 0], synth_proj[:, 0])
    )
    if n_components > 1:
        metrics["pc2_wasserstein"] = float(
            wasserstein_distance(real_proj[:, 1], synth_proj[:, 1])
        )

    return metrics


# ---------------------------------------------------------------------------
# Downstream utility
# ---------------------------------------------------------------------------

def downstream_utility(
    real_X: np.ndarray,
    real_y: np.ndarray,
    synth_X: np.ndarray,
    synth_y: np.ndarray,
    estimator: Any = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate downstream predictive utility of synthetic data.

    Trains a linear regressor on three datasets and evaluates each on the
    same held-out real test split.

    Parameters
    ----------
    real_X, real_y : training + test pool from the real dataset.
    synth_X, synth_y : synthetic dataset.
    estimator : sklearn estimator, optional
        Defaults to ``LinearRegression()``.
    test_size : float
        Fraction of real data held out for testing.
    random_state : int

    Returns
    -------
    dict with keys ``"real"``, ``"synthetic"``, ``"combined"``, each containing
    ``{"r2", "rmse", "mae"}``.

    Notes
    -----
    The ``"real"`` R² is the baseline.  ``"combined"`` > ``"real"`` indicates
    that synthetic data provides useful signal.  Do **not** interpret
    ``"synthetic"`` R² alone — a well-fit GAN can be internally consistent
    without transferring to real test data.
    """
    if estimator is None:
        estimator = LinearRegression()

    real_y = np.asarray(real_y).ravel()
    synth_y = np.asarray(synth_y).ravel()

    X_train, X_test, y_train, y_test = train_test_split(
        real_X, real_y, test_size=test_size, random_state=random_state
    )

    configs = {
        "real": (X_train, y_train),
        "synthetic": (synth_X, synth_y),
        "combined": (
            np.vstack([X_train, synth_X]),
            np.concatenate([y_train, synth_y]),
        ),
    }

    results: Dict[str, Dict[str, float]] = {}
    for name, (X_tr, y_tr) in configs.items():
        from sklearn.base import clone
        m = clone(estimator)
        m.fit(X_tr, y_tr)
        y_pred = m.predict(X_test)
        results[name] = {
            "r2": float(r2_score(y_test, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
            "mae": float(mean_absolute_error(y_test, y_pred)),
        }

    return results


# ---------------------------------------------------------------------------
# DTW reliability score
# ---------------------------------------------------------------------------

def _dtw_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Compute DTW distance between two 1-D series using dtaidistance."""
    try:
        from dtaidistance import dtw
    except ImportError as exc:
        raise ImportError(
            "dtaidistance is required for reliability_score. "
            "Install it with: pip install dtaidistance"
        ) from exc
    return float(dtw.distance(x.astype(np.float64), y.astype(np.float64)))


def reliability_score(
    real: np.ndarray,
    synth: np.ndarray,
    n_pairs: int = 50,
    seed: int = 42,
) -> Dict[str, float]:
    """
    DTW-based reliability scoring for synthetic spectra.

    Computes mean DTW distance for three pair types:

    * real-synthetic (the target)
    * real-real (intra-dataset baseline)
    * synthetic-synthetic (diversity check)

    Parameters
    ----------
    real : ndarray of shape (n_real, n_wavelengths)
    synth : ndarray of shape (n_synth, n_wavelengths)
    n_pairs : int
        Number of random pairs per group.  DTW is O(n²) per pair — keep this
        low for large datasets (subsampled by default at 50 pairs).
    seed : int

    Returns
    -------
    dict with keys:
        ``mean_rs`` (real-synthetic),
        ``mean_rr`` (real-real),
        ``mean_ss`` (synthetic-synthetic),
        ``ratio_rs_rr`` (lower = synthetic is closer to real than real is to itself),
        ``dtw_mean_spectra`` (DTW between the two mean spectra)
    """
    rng = np.random.default_rng(seed)
    n_real, n_synth = len(real), len(synth)

    real_mean = np.mean(real, axis=0)
    synth_mean = np.mean(synth, axis=0)
    dtw_mean = _dtw_distance(real_mean, synth_mean)

    def _sample_pairs(a: np.ndarray, b: np.ndarray, n: int, same: bool) -> list:
        dists = []
        na, nb = len(a), len(b)
        for _ in range(n):
            i = rng.integers(0, na)
            j = rng.integers(0, nb)
            if same:
                while j == i:
                    j = rng.integers(0, nb)
            dists.append(_dtw_distance(a[i], b[j]))
        return dists

    rs = _sample_pairs(real, synth, n_pairs, same=False)
    rr = _sample_pairs(real, real, n_pairs, same=True)
    ss = _sample_pairs(synth, synth, n_pairs, same=True)

    mean_rr = float(np.mean(rr))
    return {
        "mean_rs": float(np.mean(rs)),
        "mean_rr": mean_rr,
        "mean_ss": float(np.mean(ss)),
        "ratio_rs_rr": float(np.mean(rs)) / mean_rr if mean_rr > 0 else float("nan"),
        "dtw_mean_spectra": dtw_mean,
    }


# ---------------------------------------------------------------------------
# Aggregate comparison
# ---------------------------------------------------------------------------

def compare_generators(
    generators: Dict[str, BaseGenerativeModel],
    X_real: np.ndarray,
    y_real: np.ndarray,
    n_samples: int = 100,
    downstream_estimator: Any = None,
    dtw_n_pairs: int = 50,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Run all metric groups for every generator and return a tidy DataFrame.

    Parameters
    ----------
    generators : dict mapping name → fitted ``BaseGenerativeModel``
    X_real : ndarray of shape (n_real, n_features)
    y_real : ndarray of shape (n_real,)
    n_samples : int
        Number of synthetic samples to draw per generator.
    downstream_estimator : sklearn estimator, optional
        Defaults to ``LinearRegression()``.
    dtw_n_pairs : int
    random_state : int

    Returns
    -------
    pd.DataFrame
        One row per generator; columns for every metric.  Save with
        ``df.to_csv("results.csv")``.
    """
    rows = []
    for name, gen in generators.items():
        X_synth, y_synth = gen.sample(
            y_target=None, n_samples=n_samples
        )

        row: Dict[str, Any] = {"generator": name}

        # Spectral fidelity
        sf = spectral_fidelity_metrics(X_real, X_synth)
        row.update({f"spectral_{k}": v for k, v in sf.items()})

        # PCA
        pc = pca_metrics(X_real, X_synth)
        row.update({f"pca_{k}": v for k, v in pc.items()})

        # Downstream
        ds = downstream_utility(
            X_real, y_real, X_synth, y_synth,
            estimator=downstream_estimator,
            random_state=random_state,
        )
        for split, metrics in ds.items():
            row.update({f"downstream_{split}_{k}": v for k, v in metrics.items()})

        rows.append(row)

    return pd.DataFrame(rows).set_index("generator")
