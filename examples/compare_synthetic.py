"""
Comparison of synthetic spectrum generators: CGAN vs SMOTE vs kNN vs Noise.

Usage
-----
    python examples/compare_synthetic.py

Prerequisites
-------------
* Preprocessed dataset at ``Data/cleaned_data_rivisited.csv``
* A trained CGAN checkpoint at ``models/CGAN/cgan.pt`` (optional — skip if
  you want to train from scratch by setting TRAIN_CGAN = True below).
* ``pip install imbalanced-learn dtaidistance`` for SMOTE and reliability metrics.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from scipy.ndimage import gaussian_filter1d

from uvvislib.utils.config import Config
from uvvislib.generative import (
    CGAN,
    SmoteRegression,
    KnnInterpolation,
    NoiseAugmenter,
)
from uvvislib.generative.evaluation import compare_generators, reliability_score
from uvvislib.visualization.plots import (
    plot_gan_losses,
    plot_generative_samples,
    plot_pca_comparison,
    plot_wasserstein_profile,
    plot_cod_qq,
    plot_comparison_results,
)

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

DATA_FILE = "Data/cleaned_data_rivisited.csv"
RESULTS_DIR = "examples/results/"
TRAIN_CGAN = False          # set True to train from scratch
CGAN_CHECKPOINT = "models/CGAN/cgan.pt"
N_SYNTH = 100               # synthetic samples per generator
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# 1. Load and preprocess
# ---------------------------------------------------------------------------

import os
os.makedirs(RESULTS_DIR, exist_ok=True)

df = pd.read_csv(DATA_FILE).dropna()
wavelength_cols = [str(wav) for wav in np.arange(200, 730, 2.5)]
X_raw = df[wavelength_cols].values
y_raw = df["COD LAB"].values

# Apply same preprocessing pipeline used during CGAN training:
# Gaussian smoothing (sigma=1.5) then MinMax scaling.
ss = MinMaxScaler()
X = ss.fit_transform(gaussian_filter1d(X_raw, 1.5))
y = np.log(y_raw)   # log-COD

print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")

# ---------------------------------------------------------------------------
# 2. Build configuration
# ---------------------------------------------------------------------------

cfg = Config()
cfg.cgan_config["random_state"] = RANDOM_STATE
cfg.cgan_config["n_samples_gen"] = N_SYNTH

# ---------------------------------------------------------------------------
# 3. Fit baseline generators
# ---------------------------------------------------------------------------

smote = SmoteRegression(cfg).fit(X, y)
knn = KnnInterpolation(cfg).fit(X, y)
noise = NoiseAugmenter(cfg).fit(X, y)

# ---------------------------------------------------------------------------
# 4. CGAN — train or load
# ---------------------------------------------------------------------------

cgan = CGAN(cfg)
if TRAIN_CGAN:
    cgan.fit(X, y)
    os.makedirs(os.path.dirname(CGAN_CHECKPOINT), exist_ok=True)
    cgan.save_model(CGAN_CHECKPOINT)
    plot_gan_losses(
        cgan.training_history,
        save_path=f"{RESULTS_DIR}gan_losses.png",
    )
    print("CGAN training complete. Losses saved.")
else:
    if os.path.exists(CGAN_CHECKPOINT):
        cgan.load_model(CGAN_CHECKPOINT)
        print(f"Loaded CGAN from {CGAN_CHECKPOINT}")
    else:
        print(
            f"Checkpoint {CGAN_CHECKPOINT} not found. "
            "Set TRAIN_CGAN = True to train from scratch."
        )
        cgan = None

# ---------------------------------------------------------------------------
# 5. Compare generators
# ---------------------------------------------------------------------------

generators = {
    "smote": smote,
    "knn": knn,
    "noise": noise,
}
if cgan is not None:
    generators["cgan"] = cgan

report = compare_generators(
    generators, X, y,
    n_samples=N_SYNTH,
    random_state=RANDOM_STATE,
)
report.to_csv(f"{RESULTS_DIR}synthetic_comparison.csv")
print("\n--- Generator comparison ---")
print(report.to_string())

# ---------------------------------------------------------------------------
# 6. Visualise one generator in detail (CGAN if available, else kNN)
# ---------------------------------------------------------------------------

focus_name = "cgan" if "cgan" in generators else "knn"
focus = generators[focus_name]
X_synth, y_synth = focus.sample(y_target=None, n_samples=N_SYNTH)

wavelengths = np.arange(200, 730, 2.5)

plot_generative_samples(
    X, X_synth, wavelengths,
    title=f"Real vs {focus_name.upper()} Synthetic Spectra",
    save_path=f"{RESULTS_DIR}mean_spectra_{focus_name}.png",
)
plot_pca_comparison(
    X, X_synth,
    save_path=f"{RESULTS_DIR}pca_{focus_name}.png",
)
plot_wasserstein_profile(
    X, X_synth, wavelengths,
    save_path=f"{RESULTS_DIR}wasserstein_{focus_name}.png",
)
plot_cod_qq(
    y, y_synth,
    save_path=f"{RESULTS_DIR}cod_qq_{focus_name}.png",
)

# ---------------------------------------------------------------------------
# 7. Reliability score (requires dtaidistance)
# ---------------------------------------------------------------------------

try:
    rel = reliability_score(X, X_synth, n_pairs=50, seed=RANDOM_STATE)
    print(f"\n--- Reliability ({focus_name}) ---")
    for k, v in rel.items():
        print(f"  {k}: {v:.4f}")
except ImportError:
    print("\ndtaidistance not installed — skipping reliability_score.")

# ---------------------------------------------------------------------------
# 8. Summary bar chart
# ---------------------------------------------------------------------------

plot_comparison_results(
    report,
    metric_groups=["spectral", "downstream_combined"],
    save_path=f"{RESULTS_DIR}comparison_summary.png",
)
print(f"\nAll results saved to {RESULTS_DIR}")
