"""
Gaussian noise augmentation baseline.

Adds i.i.d. Gaussian noise to real spectra.  Simple but useful as a lower-bound
baseline when comparing generative methods.
"""

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd

from .base import BaseGenerativeModel
from ..utils.config import Config


class NoiseAugmenter(BaseGenerativeModel):
    """
    Gaussian noise augmentation baseline.

    Parameters
    ----------
    config : Config
        Library configuration.  Relevant keys in ``config.cgan_config``:

        * ``noise_sigma`` (float, default 0.1) — std of additive Gaussian noise.
        * ``random_state`` (int, default 42).
    """

    def __init__(self, config: Config, model_name: str = "noise_augmenter"):
        super().__init__(config, model_name)
        self._X_fit: Optional[np.ndarray] = None
        self._y_fit: Optional[np.ndarray] = None

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.DataFrame],
        **kwargs,
    ) -> "NoiseAugmenter":
        """
        Store training data (noise is added at sample time).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)

        Returns
        -------
        self
        """
        X, y = self.validate_data(X, y)
        self._update_model_info(X, y)
        self._X_fit = X.astype(np.float32)
        self._y_fit = y.ravel().astype(np.float32)
        self.is_fitted = True
        self.logger.info(f"NoiseAugmenter fitted: {len(X)} samples")
        return self

    def sample(
        self,
        y_target: Union[float, np.ndarray, None] = None,
        n_samples: int = 100,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Draw ``n_samples`` real spectra at random and add Gaussian noise.

        ``y_target`` is ignored — labels are inherited from the sampled real
        spectra.

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
        sigma: float = cfg.get("noise_sigma", 0.1)
        random_state: int = cfg.get("random_state", 42)

        rng = np.random.default_rng(random_state)
        n = len(self._X_fit)
        idx = rng.integers(0, n, size=n_samples)

        X_base = self._X_fit[idx]
        noise = rng.normal(0.0, sigma, X_base.shape).astype(np.float32)
        X_synth = X_base + noise
        y_synth = self._y_fit[idx]

        return X_synth, y_synth

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_model_impl(self, filepath: Path) -> None:
        np.savez(filepath, X_fit=self._X_fit, y_fit=self._y_fit)

    def _load_model_impl(self, filepath: Path) -> None:
        data = np.load(filepath)
        self._X_fit = data["X_fit"]
        self._y_fit = data["y_fit"]
        self.is_fitted = True
