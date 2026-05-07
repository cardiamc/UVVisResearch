"""
Abstract base class for generative models that produce synthetic UV-Vis spectra.
"""

from abc import abstractmethod
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd

from ..models.base import BaseModel
from ..utils.config import Config


class BaseGenerativeModel(BaseModel):
    """
    Abstract base for all generative / augmentation models.

    Subclasses must implement ``fit``, ``sample``, ``_save_model_impl``,
    and ``_load_model_impl``.  ``predict`` always raises ``NotImplementedError``
    — generative models are not discriminative regressors.
    """

    def __init__(self, config: Config, model_name: str = "base_generative"):
        super().__init__(config, model_name)

    @abstractmethod
    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.DataFrame],
        **kwargs,
    ) -> "BaseGenerativeModel":
        """
        Fit the generative model to training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Preprocessed spectra (Gaussian-smoothed, MinMax-scaled).
        y : array-like of shape (n_samples,)
            Target values in **log space** (``np.log(COD)``).

        Returns
        -------
        self
        """

    @abstractmethod
    def sample(
        self,
        y_target: Union[float, np.ndarray],
        n_samples: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic spectra conditioned on target values.

        Parameters
        ----------
        y_target : float or ndarray
            Target values in **log space** (log-COD).  A scalar is broadcast to
            ``n_samples`` identical conditions; an array of length n is used
            directly (``n_samples`` must be None or equal to n).
        n_samples : int, optional
            Number of samples to generate.  Required when ``y_target`` is a
            scalar.

        Returns
        -------
        X_synth : ndarray of shape (n_samples, n_features)
            Generated spectra in the **MinMax-scaled domain** (same scale as
            the training ``X``).  Callers that need physical absorbance units
            should apply the fitted ``MinMaxScaler.inverse_transform``.
        y_synth : ndarray of shape (n_samples,)
            Corresponding log-COD values.  Call ``np.exp(y_synth)`` to recover
            COD in mg/L.
        """

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        raise NotImplementedError(
            f"{self.__class__.__name__} is a generative model and does not implement "
            "predict().  Use sample() to generate synthetic spectra."
        )
