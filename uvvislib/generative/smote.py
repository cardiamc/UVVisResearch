"""
Regression-friendly SMOTE wrapper for UV-Vis spectral augmentation.

Bins the continuous COD target into quantile bins, applies SMOTE on the
(spectrum, bin_label) pairs, then recovers continuous y values by random
sampling within each bin.  This avoids the pathological case of naive
SMOTE on raw continuous targets (which would interpolate y between unrelated
samples chosen only by spectral proximity).
"""

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd

from .base import BaseGenerativeModel
from ..utils.config import Config


class SmoteRegression(BaseGenerativeModel):
    """
    Regression SMOTE augmenter based on quantile binning.

    Parameters
    ----------
    config : Config
        Library configuration.  Relevant keys in ``config.cgan_config``:

        * ``smote_n_bins`` (int, default 10) — quantile bins for the target.
        * ``smote_k_neighbors`` (int, default 5) — SMOTE k_neighbors.
        * ``random_state`` (int, default 42).

    Notes
    -----
    Requires ``imbalanced-learn`` (``imblearn``).  Add to ``requirements.txt``
    if not already present.
    """

    def __init__(self, config: Config, model_name: str = "smote_regression"):
        super().__init__(config, model_name)
        self._X_fit: Optional[np.ndarray] = None
        self._y_fit: Optional[np.ndarray] = None
        self._y_binned: Optional[np.ndarray] = None

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.DataFrame],
        **kwargs,
    ) -> "SmoteRegression":
        """
        Fit on training data (stores X and y for use during sampling).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            MinMax-scaled spectra.
        y : array-like of shape (n_samples,)
            Log-COD targets.

        Returns
        -------
        self
        """
        X, y = self.validate_data(X, y)
        self._update_model_info(X, y)

        cfg = self.config.cgan_config
        n_bins: int = cfg.get("smote_n_bins", 10)

        y_1d = y.ravel()
        self._X_fit = X.astype(np.float32)
        self._y_fit = y_1d.astype(np.float32)

        # Quantile binning — handles skewed COD distributions correctly
        self._y_binned = pd.qcut(
            y_1d, q=n_bins, labels=False, duplicates="drop"
        ).astype(int)

        self.is_fitted = True
        self.logger.info(
            f"SmoteRegression fitted: {len(X)} samples, "
            f"{len(np.unique(self._y_binned))} unique bins"
        )
        return self

    def sample(
        self,
        y_target: Union[float, np.ndarray, None] = None,
        n_samples: int = 100,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate ``n_samples`` synthetic spectra via SMOTE.

        ``y_target`` is ignored — SMOTE determines the distribution of generated
        labels from the training data's bin structure.

        Parameters
        ----------
        y_target : ignored
            Present for API compatibility with ``BaseGenerativeModel``.
        n_samples : int
            Number of synthetic samples to return.

        Returns
        -------
        X_synth : ndarray of shape (n_samples, n_features)
        y_synth : ndarray of shape (n_samples,) — log-COD values
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit() before sample().")

        try:
            from imblearn.over_sampling import SMOTE
        except ImportError as exc:
            raise ImportError(
                "imbalanced-learn is required for SmoteRegression. "
                "Install it with: pip install imbalanced-learn"
            ) from exc

        cfg = self.config.cgan_config
        k_neighbors: int = cfg.get("smote_k_neighbors", 5)
        random_state: int = cfg.get("random_state", 42)
        n_original = len(self._X_fit)

        # Add at least ceil(n_samples / n_unique_bins) new samples per bin.
        # Sampling strategy = "target count per class after resampling"; it must
        # exceed the original bin count or SMOTE skips that class entirely.
        unique_bins = np.unique(self._y_binned)
        n_unique_bins = len(unique_bins)
        add_per_bin = max(int(np.ceil(n_samples / n_unique_bins)), k_neighbors + 1)
        sampling_strategy = {
            int(b): int(np.sum(self._y_binned == b)) + add_per_bin
            for b in unique_bins
        }

        smote = SMOTE(
            sampling_strategy=sampling_strategy,
            k_neighbors=k_neighbors,
            random_state=random_state,
        )
        X_res, y_binned_res = smote.fit_resample(self._X_fit, self._y_binned)

        X_synthetic = X_res[n_original:]
        y_binned_synthetic = y_binned_res[n_original:]

        # Recover continuous y by random draw from the matching real bin
        rng = np.random.default_rng(random_state)
        y_synthetic = np.array([
            rng.choice(self._y_fit[self._y_binned == b])
            if np.sum(self._y_binned == b) > 0
            else float(np.mean(self._y_fit))
            for b in y_binned_synthetic
        ], dtype=np.float32)

        # Subsample to exactly n_samples
        if len(X_synthetic) > n_samples:
            idx = rng.choice(len(X_synthetic), n_samples, replace=False)
            X_synthetic = X_synthetic[idx]
            y_synthetic = y_synthetic[idx]

        return X_synthetic, y_synthetic

    # ------------------------------------------------------------------
    # Persistence — no weights to save; store fit data as numpy archive
    # ------------------------------------------------------------------

    def _save_model_impl(self, filepath: Path) -> None:
        np.savez(
            filepath,
            X_fit=self._X_fit,
            y_fit=self._y_fit,
            y_binned=self._y_binned,
        )

    def _load_model_impl(self, filepath: Path) -> None:
        data = np.load(filepath)
        self._X_fit = data["X_fit"]
        self._y_fit = data["y_fit"]
        self._y_binned = data["y_binned"].astype(int)
        self.is_fitted = True
