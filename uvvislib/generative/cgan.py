"""
Conditional GAN for UV-Vis spectrum generation conditioned on continuous COD labels.

Architecture mirrors the research code exactly so that pre-trained weight files
(state-dict format ``{"G": ..., "D": ...}``) remain loadable.
"""

import math
import time
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from .base import BaseGenerativeModel
from ..utils.config import Config


# ---------------------------------------------------------------------------
# Architecture helpers
# ---------------------------------------------------------------------------

def _conv1d_output_length(
    l_input: int,
    kernel_size: int,
    padding: int = 0,
    dilation: int = 1,
    stride: int = 1,
) -> int:
    """Compute the output length of a Conv1d or ConvTranspose1d layer."""
    return math.floor(
        (l_input + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1
    )


def _initialize_weights(net: nn.Module) -> None:
    """
    Apply normal(0, 0.02) initialisation to Conv2d, ConvTranspose2d, and Linear layers.

    Conv1d and ConvTranspose1d are intentionally skipped — the original research code
    did not initialise them, so they retain PyTorch's default Kaiming-uniform init.
    Changing this would break compatibility with pre-trained checkpoints.
    """
    for m in net.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()


def _generator_loss(
    discriminator_output: torch.Tensor,
    generated_signal: torch.Tensor,
    lambda_reg: float = 0.01,
) -> torch.Tensor:
    """
    Non-saturating GAN loss for the generator with total-variation regularisation.

    The TV term penalises high-frequency noise between consecutive wavelength
    bins, acting as a spectral smoothness prior.  Tune ``lambda_reg`` via
    ``Config.cgan_config["lambda_reg"]``.
    """
    gan_loss = -torch.mean(torch.log(discriminator_output + 1e-8))
    diff = torch.abs(generated_signal[:, :, 1:] - generated_signal[:, :, :-1])
    reg_term = lambda_reg * torch.mean(diff)
    return gan_loss + reg_term


# ---------------------------------------------------------------------------
# Network modules (private — identical to research code for reproducibility)
# ---------------------------------------------------------------------------

class _Generator(nn.Module):
    def __init__(
        self,
        input_dim: int = 200,
        output_dim: int = 1,
        input_size: int = 212,
        class_num: int = 1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size
        self.class_num = class_num

        spatial = _conv1d_output_length(
            _conv1d_output_length(input_size, 4, 1, 1, 2), 4, 1, 1, 2
        )

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim + self.class_num, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.8),
            nn.Dropout(0.25),
            nn.Linear(1024, 128 * spatial),
            nn.BatchNorm1d(128 * spatial),
            nn.LeakyReLU(0.8),
            nn.Dropout(0.25),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose1d(128, 64, 4, 2, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose1d(64, self.output_dim, 4, 2, 1),
            nn.BatchNorm1d(self.output_dim),
            nn.ReLU(),
        )
        self._spatial = spatial
        _initialize_weights(self)

    def forward(self, z: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z, label], dim=1)
        x = self.fc(x)
        x = x.view(-1, 128, self._spatial)
        x = self.deconv(x)
        x = self.deconv2(x)
        return x


class _Discriminator(nn.Module):
    def __init__(
        self,
        input_dim: int = 1,
        output_dim: int = 1,
        input_size: int = 212,
        class_num: int = 1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size
        self.class_num = class_num

        s1 = _conv1d_output_length(input_size, 4, 3, 1, 2)
        s2 = _conv1d_output_length(s1, 4, 1, 1, 2)

        self.conv = nn.Sequential(
            nn.Conv1d(self.input_dim + self.class_num, 64, 4, 2, 3),
            nn.LeakyReLU(0.2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, 4, 2, 1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * s2, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, self.output_dim),
            nn.Sigmoid(),
        )
        self._flat_size = 128 * s2
        _initialize_weights(self)

    def forward(self, spectrum: torch.Tensor, label_fill: torch.Tensor) -> torch.Tensor:
        x = torch.cat([spectrum, label_fill], dim=1)
        x = self.conv(x)
        x = self.conv2(x)
        x = x.view(-1, self._flat_size)
        x = self.fc(x)
        return x


# ---------------------------------------------------------------------------
# Public CGAN class
# ---------------------------------------------------------------------------

class CGAN(BaseGenerativeModel):
    """
    Conditional GAN for UV-Vis spectrum generation.

    Conditions on a continuous log-COD label (regression setting).  The
    discriminator concatenates ``[spectrum, label_fill]`` channel-wise; the
    generator concatenates ``[noise_z, label]`` feature-wise.

    Parameters
    ----------
    config : Config
        Library configuration.  CGAN-specific hypers live in
        ``config.cgan_config`` (see ``Config`` docstring for all fields).
    model_name : str
        Identifier used by ``ModelManager`` / ``ExperimentManager``.

    Notes
    -----
    *  ``fit(X, y)`` expects ``X`` already preprocessed (Gaussian + MinMax).
    *  ``sample()`` returns spectra in the **MinMax-scaled domain** — callers
       that need physical absorbance units must apply the inverse transform.
    *  Targets throughout are in **log space**: ``y = np.log(COD)``.
       Use ``np.exp(y_synth)`` to recover COD in mg/L.
    """

    def __init__(self, config: Config, model_name: str = "cgan"):
        super().__init__(config, model_name)
        cfg = config.cgan_config

        self.input_size: int = cfg.get("input_size", 212)
        self.z_dim: int = cfg.get("n_z", 200)
        self.class_num: int = cfg.get("class_num", 1)
        self.n_samples_gen: int = cfg.get("n_samples_gen", 500)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Seed before weight init for reproducibility (affects Linear/Conv2d init only;
        # Conv1d layers use PyTorch's default — see _initialize_weights docstring).
        torch.manual_seed(cfg.get("random_state", 42))

        self.G = _Generator(
            input_dim=self.z_dim,
            output_dim=1,
            input_size=self.input_size,
            class_num=self.class_num,
        ).to(self.device)

        self.D = _Discriminator(
            input_dim=1,
            output_dim=1,
            input_size=self.input_size,
            class_num=self.class_num,
        ).to(self.device)

        lrG = cfg.get("lrG", 1e-4)
        lrD = cfg.get("lrD", 1e-4)
        beta1 = cfg.get("beta1", 0.5)
        beta2 = cfg.get("beta2", 0.999)
        wd = cfg.get("weight_decay", 1e-4)

        self.G_optimizer = optim.Adam(
            self.G.parameters(), lr=lrG, betas=(beta1, beta2), weight_decay=wd
        )
        self.D_optimizer = optim.Adam(
            self.D.parameters(), lr=lrD, betas=(beta1, beta2), weight_decay=wd
        )
        self.BCE_loss = nn.BCELoss().to(self.device)

        # Fixed evaluation noise buffer (seeded separately from network init)
        rng = torch.Generator()
        rng.manual_seed(cfg.get("random_state", 42))
        self.sample_z_ = torch.rand((10, self.z_dim), generator=rng).to(self.device)

        self.logger.info(str(self.G))
        g_params = sum(p.numel() for p in self.G.parameters())
        d_params = sum(p.numel() for p in self.D.parameters())
        self.logger.info(f"Generator parameters: {g_params:,}")
        self.logger.info(str(self.D))
        self.logger.info(f"Discriminator parameters: {d_params:,}")

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        X: Union[np.ndarray, "pd.DataFrame"],
        y: Union[np.ndarray, "pd.DataFrame"],
        **kwargs,
    ) -> "CGAN":
        """
        Train the CGAN on preprocessed spectra.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            MinMax-scaled spectra (after Gaussian smoothing).
        y : array-like of shape (n_samples,)
            Log-COD targets (``np.log(COD)``).

        Returns
        -------
        self
        """
        X, y = self.validate_data(X, y)   # y → (n, 1)
        self._update_model_info(X, y)

        cfg = self.config.cgan_config
        epochs: int = cfg.get("epochs", 200)
        batch_size: int = cfg.get("batch_size", 64)
        lambda_reg: float = cfg.get("lambda_reg", 0.01)

        y_1d = y.ravel().astype(np.float32)
        X_f = X.astype(np.float32)

        dataset = TensorDataset(
            torch.from_numpy(X_f), torch.from_numpy(y_1d)
        )
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, drop_last=True
        )

        if len(loader) == 0:
            raise ValueError(
                f"Dataset ({len(X)} samples) is smaller than batch_size={batch_size}. "
                "Reduce Config.cgan_config['batch_size']."
            )

        y_real_ = torch.ones(batch_size, 1).to(self.device)
        y_fake_ = torch.zeros(batch_size, 1).to(self.device)

        self.training_history = {"D_loss": [], "G_loss": [], "per_epoch_time": []}

        self.D.train()
        start_time = time.time()

        for epoch in range(epochs):
            self.G.train()
            epoch_start = time.time()
            d_loss_sum = g_loss_sum = 0.0
            n_batches = 0
            last_fake: Optional[torch.Tensor] = None

            for x_b, y_b in loader:
                # shape: (B, 1, input_size) and (B, 1) and (B, 1, input_size)
                x_ = x_b.unsqueeze(1).to(self.device)
                z_ = torch.rand(batch_size, self.z_dim).to(self.device)
                y_vec_ = y_b.unsqueeze(1).to(self.device)
                y_fill_ = y_vec_.unsqueeze(2).expand(
                    batch_size, 1, self.input_size
                ).to(self.device)

                # --- Discriminator update ---
                self.D_optimizer.zero_grad()
                D_real = self.D(x_, y_fill_)
                D_real_loss = self.BCE_loss(D_real, y_real_)
                # Detach G output so gradients don't flow back through G here
                G_detached = self.G(z_, y_vec_).detach()
                D_fake = self.D(G_detached, y_fill_)
                D_fake_loss = self.BCE_loss(D_fake, y_fake_)
                D_loss = D_real_loss + D_fake_loss
                D_loss.backward()
                self.D_optimizer.step()

                # --- Generator update ---
                self.G_optimizer.zero_grad()
                G_ = self.G(z_, y_vec_)
                D_fake_g = self.D(G_, y_fill_)
                G_loss = _generator_loss(D_fake_g, G_, lambda_reg)
                G_loss.backward()
                self.G_optimizer.step()

                d_loss_sum += D_loss.item()
                g_loss_sum += G_loss.item()
                n_batches += 1
                last_fake = G_.detach()

            epoch_d_loss = d_loss_sum / n_batches
            epoch_g_loss = g_loss_sum / n_batches
            epoch_time = time.time() - epoch_start

            self.training_history["D_loss"].append(epoch_d_loss)
            self.training_history["G_loss"].append(epoch_g_loss)
            self.training_history["per_epoch_time"].append(epoch_time)

            if epoch % 100 == 0:
                g_std = torch.std(last_fake, dim=0).mean().item() if last_fake is not None else float("nan")
                self.logger.info(
                    f"Epoch [{epoch + 1}/{epochs}]  "
                    f"D_loss: {epoch_d_loss:.6f}  "
                    f"G_loss: {epoch_g_loss:.6f}  "
                    f"G_std: {g_std:.4f}"
                )
                if g_std < 0.01:
                    self.logger.warning(
                        f"G_std={g_std:.4f} is very low — possible mode collapse. "
                        "Consider reducing lrD, increasing beta1, or adding noise to "
                        "discriminator inputs."
                    )

        total_time = time.time() - start_time
        self.training_history["total_time"] = [total_time]
        avg_epoch = np.mean(self.training_history["per_epoch_time"])
        self.logger.info(
            f"Training complete.  Avg epoch: {avg_epoch:.2f}s  "
            f"Total: {total_time:.2f}s"
        )

        self.is_fitted = True
        return self

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(
        self,
        y_target: Union[float, np.ndarray],
        n_samples: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic spectra conditioned on log-COD target values.

        Parameters
        ----------
        y_target : float or ndarray
            Target(s) in log space.  Scalar → broadcast to ``n_samples``;
            array of shape (n,) or (n, 1) → used directly.
        n_samples : int, optional
            Required when ``y_target`` is a scalar.

        Returns
        -------
        X_synth : ndarray of shape (n_samples, input_size)
            Generated spectra in MinMax-scaled domain.
        y_synth : ndarray of shape (n_samples,)
            Log-COD values.  ``np.exp(y_synth)`` gives COD in mg/L.
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit() before sample().")

        y_arr = np.asarray(y_target, dtype=np.float32).ravel()

        if y_arr.size == 1:
            if n_samples is None:
                raise ValueError("n_samples is required when y_target is a scalar.")
            y_arr = np.full(n_samples, y_arr[0], dtype=np.float32)
        else:
            if n_samples is not None and n_samples != len(y_arr):
                raise ValueError(
                    f"n_samples={n_samples} but y_target has {len(y_arr)} values."
                )
            n_samples = len(y_arr)

        y_t = torch.from_numpy(y_arr).float().unsqueeze(1).to(self.device)
        z_ = torch.rand(n_samples, self.z_dim).to(self.device)

        self.G.eval()
        with torch.no_grad():
            out = self.G(z_, y_t)   # (n, 1, input_size)

        X_synth = out.cpu().numpy().squeeze(1)   # (n, input_size)
        return X_synth, y_arr

    # ------------------------------------------------------------------
    # Persistence (BaseModel hooks)
    # ------------------------------------------------------------------

    def _save_model_impl(self, filepath: Path) -> None:
        torch.save(
            {"G": self.G.state_dict(), "D": self.D.state_dict()},
            filepath,
        )

    def _load_model_impl(self, filepath: Path) -> None:
        ckpt = torch.load(filepath, map_location=self.device)
        self.G.load_state_dict(ckpt["G"])
        self.D.load_state_dict(ckpt["D"])
        self.is_fitted = True
