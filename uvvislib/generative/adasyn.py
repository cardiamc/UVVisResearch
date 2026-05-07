"""
kNN-interpolation augmenter for UV-Vis spectral data.

For each synthetic sample: pick a random real anchor, pick one of its k nearest
neighbours, interpolate both spectrum and log-COD with a random alpha in [0, 1].
This is what the research file ``adasyn_sampling.py`` implements — it is kNN
interpolation, not ADASYN (which also uses density weighting).  The class is
named ``KnnInterpolation`` to be honest about the algorithm.
"""

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from .base import BaseGenerativeModel
from ..utils.config import Config


class KnnInterpolation(BaseGenerativeModel):
    """
    Random kNN-interpolation augmenter.

    Parameters
    ----------
    config : Config
        Library configuration.  Relevant keys in ``config.cgan_config``:

        * ``knn_k_neighbors`` (int, default 5) — neighbourhood size.
        * ``random_state`` (int, default 42).
    """

    def __init__(self, config: Config, model_name: str = "knn_interpolation"):
        super().__init__(config, model_name)
        self._X_fit: Optional[np.ndarray] = None
        self._y_fit: Optional[np.ndarray] = None
        self._knn: Optional[NearestNeighbors] = None
        self._neighbor_indices: Optional[np.ndarray] = None

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.DataFrame],
        **kwargs,
    ) -> "KnnInterpolation":
        """
        Fit NearestNeighbors on training spectra.

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
        k: int = cfg.get("knn_k_neighbors", 5)

        self._X_fit = X.astype(np.float32)
        self._y_fit = y.ravel().astype(np.float32)

        # k+1 because the first neighbour is the point itself
        self._knn = NearestNeighbors(n_neighbors=k + 1)
        self._knn.fit(self._X_fit)
        _, indices = self._knn.kneighbors(self._X_fit)
        self._neighbor_indices = indices   # shape (n_samples, k+1)

        self.is_fitted = True
        self.logger.info(
            f"KnnInterpolation fitted: {len(X)} samples, k={k}"
        )
        return self

    def sample(
        self,
        y_target: Union[float, np.ndarray, None] = None,
        n_samples: int = 100,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate ``n_samples`` synthetic spectra via kNN interpolation.

        ``y_target`` is ignored — the distribution of generated labels comes
        from interpolating within the training data.

        Parameters
        ----------
        y_target : ignored
        n_samples : int

        Returns
        -------
        X_synth : ndarray of shape (n_samples, n_features)
        y_synth : ndarray of shape (n_samples,) — log-COD values
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit() before sample().")

        cfg = self.config.cgan_config
        rng = np.random.default_rng(cfg.get("random_state", 42))

        n = len(self._X_fit)
        X_synthetic = np.empty((n_samples, self._X_fit.shape[1]), dtype=np.float32)
        y_synthetic = np.empty(n_samples, dtype=np.float32)

        for i in range(n_samples):
            anchor = rng.integers(0, n)
            # Skip index 0 in neighbors — that's the point itself
            neighbor = rng.choice(self._neighbor_indices[anchor][1:])
            alpha = rng.random()
            X_synthetic[i] = (
                self._X_fit[anchor] + alpha * (self._X_fit[neighbor] - self._X_fit[anchor])
            )
            y_synthetic[i] = (
                self._y_fit[anchor] + alpha * (self._y_fit[neighbor] - self._y_fit[anchor])
            )

        return X_synthetic, y_synthetic

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_model_impl(self, filepath: Path) -> None:
        np.savez(
            filepath,
            X_fit=self._X_fit,
            y_fit=self._y_fit,
            neighbor_indices=self._neighbor_indices,
        )

    def _load_model_impl(self, filepath: Path) -> None:
        data = np.load(filepath)
        self._X_fit = data["X_fit"]
        self._y_fit = data["y_fit"]
        self._neighbor_indices = data["neighbor_indices"]

        cfg = self.config.cgan_config
        k: int = cfg.get("knn_k_neighbors", 5)
        self._knn = NearestNeighbors(n_neighbors=k + 1)
        self._knn.fit(self._X_fit)
        self.is_fitted = True
