# Examples Gallery

This gallery provides comprehensive examples of how to use the UV-Vis Research Library for various spectral data analysis tasks.

## Table of Contents

- [Basic Examples](#basic-examples)
- [Advanced Workflows](#advanced-workflows)
- [Real-world Applications](#real-world-applications)
- [Performance Optimization](#performance-optimization)
- [Custom Extensions](#custom-extensions)

## Basic Examples

### Example 1: Simple Spectral Regression

**Objective**: Predict a single target variable from spectral data using MLP.

```python
import numpy as np
import pandas as pd
from uvvislib.data.loader import load_data
from uvvislib.data.preprocessing import preprocess_spectra
from uvvislib.models.mlp import MLPRegressor
from uvvislib.evaluation.metrics import calculate_metrics
from uvvislib.utils.config import Config
from uvvislib.visualization.plots import plot_predictions_vs_actual

# Load data
X, y = load_data('spectral_data.csv', 
                 feature_columns=range(200, 800, 2),
                 target_columns=['concentration'])

# Preprocess
X_processed = preprocess_spectra(X, normalize=True, baseline_correction=True)

# Split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42
)

# Configure model
config = Config(
    input_size=X_processed.shape[1],
    hidden_sizes=[128, 64],
    learning_rate=0.001,
    batch_size=32,
    epochs=100,
    validation_split=0.2
)

# Train model
model = MLPRegressor(config)
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
metrics = calculate_metrics(y_test, predictions)

print(f"R² Score: {metrics['r2_score']:.4f}")
print(f"RMSE: {metrics['rmse']:.4f}")

# Visualize results
plot_predictions_vs_actual(y_test, predictions, 
                          title="MLP Regression Results")
```

### Example 2: Multi-target Regression

**Objective**: Predict multiple target variables simultaneously.

```python
from uvvislib.models.mlp import MLPRegressor

# Load multi-target data
X, y = load_data('multi_target_data.csv',
                 feature_columns=range(200, 800, 2),
                 target_columns=['concentration', 'pH', 'turbidity'])

# Preprocess
X_processed = preprocess_spectra(X, normalize=True)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42
)

# Configure for multi-target
config = Config(
    input_size=X_processed.shape[1],
    hidden_sizes=[256, 128, 64],
    learning_rate=0.001,
    batch_size=32,
    epochs=150,
    validation_split=0.2
)

# Train model
model = MLPRegressor(config)
model.fit(X_train, y_train)

# Evaluate each target
predictions = model.predict(X_test)
for i, target in enumerate(y.columns):
    target_metrics = calculate_metrics(y_test.iloc[:, i], predictions[:, i])
    print(f"{target}: R² = {target_metrics['r2_score']:.4f}, RMSE = {target_metrics['rmse']:.4f}")
```

### Example 3: Random Forest Analysis

**Objective**: Use Random Forest for regression and feature importance analysis.

```python
from uvvislib.models.random_forest import RandomForestRegressor
from uvvislib.evaluation.cross_validation import CrossValidator
import matplotlib.pyplot as plt

# Load and preprocess data
X, y = load_data('spectral_data.csv')
X_processed = preprocess_spectra(X, normalize=True)

# Train Random Forest
rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42
)
rf_model.fit(X_processed, y)

# Cross-validation
cv = CrossValidator(n_splits=5, random_state=42)
cv_results = cv.cross_validate(rf_model, X_processed, y)

print(f"Cross-validation R²: {cv_results['r2_score'].mean():.4f} ± {cv_results['r2_score'].std():.4f}")

# Feature importance analysis
importance = rf_model.get_feature_importance()
wavelengths = X_processed.columns

# Plot feature importance
plt.figure(figsize=(12, 6))
plt.bar(range(len(importance)), importance)
plt.xlabel('Wavelength Index')
plt.ylabel('Feature Importance')
plt.title('Random Forest Feature Importance')
plt.xticks(range(0, len(wavelengths), 50), wavelengths[::50], rotation=45)
plt.tight_layout()
plt.show()

# Get top important wavelengths
top_indices = np.argsort(importance)[-10:]
print("Top 10 important wavelengths:", wavelengths[top_indices])
```

## Advanced Workflows

### Example 4: CNN for Spectral Data

**Objective**: Use Convolutional Neural Networks for spectral pattern recognition.

```python
from uvvislib.models.cnn import CNNRegressor
from uvvislib.visualization.plots import plot_spectra

# Load data
X, y = load_data('spectral_data.csv')
X_processed = preprocess_spectra(X, normalize=True, smoothing=True)

# Visualize original spectra
plot_spectra(X_processed.iloc[:10], 
             title="Sample Spectra",
             xlabel="Wavelength (nm)",
             ylabel="Absorbance")

# Configure CNN
config = Config(
    input_size=X_processed.shape[1],
    conv_layers=[32, 64, 128],  # Multiple conv layers
    fc_layers=[256, 128, 64],   # Fully connected layers
    kernel_size=5,              # Larger kernel for spectral patterns
    stride=1,
    learning_rate=0.001,
    batch_size=16,              # Smaller batch size for CNN
    epochs=200,
    validation_split=0.2
)

# Train CNN
cnn_model = CNNRegressor(config)
cnn_model.fit(X_processed, y)

# Evaluate
predictions = cnn_model.predict(X_processed)
metrics = calculate_metrics(y, predictions)
print(f"CNN R² Score: {metrics['r2_score']:.4f}")

# Plot training history
cnn_model.plot_training_history()
```

### Example 5: Hybrid CNN-MLP Model

**Objective**: Combine CNN and MLP for complex spectral analysis.

```python
from uvvislib.models.cnn_mlp import CNNMLPRegressor

# Configure hybrid model
config = Config(
    input_size=X_processed.shape[1],
    conv_layers=[32, 64],       # CNN layers for feature extraction
    mlp_layers=[256, 128, 64],  # MLP layers for regression
    kernel_size=3,
    stride=1,
    learning_rate=0.001,
    batch_size=32,
    epochs=150,
    validation_split=0.2
)

# Train hybrid model
hybrid_model = CNNMLPRegressor(config)
hybrid_model.fit(X_processed, y)

# Compare with individual models
mlp_model = MLPRegressor(config)
mlp_model.fit(X_processed, y)

mlp_pred = mlp_model.predict(X_processed)
hybrid_pred = hybrid_model.predict(X_processed)

mlp_metrics = calculate_metrics(y, mlp_pred)
hybrid_metrics = calculate_metrics(y, hybrid_pred)

print(f"MLP R²: {mlp_metrics['r2_score']:.4f}")
print(f"Hybrid R²: {hybrid_metrics['r2_score']:.4f}")
```

### Example 6: Clustering Analysis

**Objective**: Perform unsupervised clustering on spectral data.

```python
from uvvislib.models.clustering import SpectralClustering
from uvvislib.visualization.plots import plot_clustering_results
from sklearn.decomposition import PCA

# Load data
X, y = load_data('spectral_data.csv')
X_processed = preprocess_spectra(X, normalize=True)

# Perform clustering
clustering = SpectralClustering(
    n_clusters=3,
    method='kmeans',
    random_state=42
)
cluster_labels = clustering.fit_predict(X_processed)

# Visualize clustering results
plot_clustering_results(X_processed, cluster_labels, 
                       title="Spectral Clustering Results")

# PCA for dimensionality reduction and visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_processed)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis')
plt.colorbar(scatter)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Spectral Clustering - PCA Visualization')
plt.show()

# Analyze cluster characteristics
for cluster_id in np.unique(cluster_labels):
    cluster_data = X_processed[cluster_labels == cluster_id]
    print(f"Cluster {cluster_id}: {len(cluster_data)} samples")
    print(f"  Mean spectrum: {cluster_data.mean().mean():.4f}")
    print(f"  Std spectrum: {cluster_data.std().mean():.4f}")
```

## Real-world Applications

### Example 7: Water Quality Analysis

**Objective**: Analyze water quality parameters from UV-Vis spectra.

```python
import numpy as np
from uvvislib.evaluation.cross_validation import CrossValidator
from uvvislib.persistence.experiment_manager import ExperimentManager

# Load water quality data
X, y = load_data('water_quality_data.csv',
                 feature_columns=range(200, 800, 2),
                 target_columns=['TOC', 'pH', 'Turbidity', 'Conductivity'])

# Preprocess with water-specific techniques
X_processed = preprocess_spectra(
    X,
    normalize=True,
    baseline_correction=True,
    smoothing=True,
    smoothing_window=7
)

# Multiple model comparison
models = {
    'MLP': MLPRegressor(Config(
        input_size=X_processed.shape[1],
        hidden_sizes=[256, 128, 64],
        learning_rate=0.001,
        batch_size=32,
        epochs=100
    )),
    'CNN': CNNRegressor(Config(
        input_size=X_processed.shape[1],
        conv_layers=[32, 64],
        fc_layers=[128, 64],
        learning_rate=0.001,
        batch_size=16,
        epochs=150
    )),
    'RandomForest': RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        random_state=42
    )
}

# Cross-validation comparison
cv = CrossValidator(n_splits=5, random_state=42)
results = {}

for name, model in models.items():
    print(f"Training {name}...")
    cv_results = cv.cross_validate(model, X_processed, y)
    results[name] = cv_results
    
    print(f"{name} - R²: {cv_results['r2_score'].mean():.4f} ± {cv_results['r2_score'].std():.4f}")

# Save best experiment
best_model_name = max(results.keys(), key=lambda k: results[k]['r2_score'].mean())
best_model = models[best_model_name]
best_results = results[best_model_name]

exp_manager = ExperimentManager()
exp_manager.save_experiment(
    model=best_model,
    results=best_results,
    config=best_model.config if hasattr(best_model, 'config') else None,
    name="water_quality_analysis",
    description=f"Best model: {best_model_name} for water quality prediction"
)
```

### Example 8: Pharmaceutical Analysis

**Objective**: Analyze drug concentration from UV-Vis spectra.

```python
from uvvislib.models.mlp import MLPRegressor
from uvvislib.visualization.plots import plot_spectra

# Load pharmaceutical data
X, y = load_data('pharma_data.csv',
                 feature_columns=range(200, 400, 1),  # UV range
                 target_columns=['concentration'])

# Pharmaceutical-specific preprocessing
X_processed = preprocess_spectra(
    X,
    normalize=True,
    baseline_correction=True,
    smoothing=True,
    smoothing_window=3  # Smaller window for UV data
)

# Plot sample spectra
plot_spectra(X_processed.iloc[:5], 
             title="Pharmaceutical UV Spectra",
             xlabel="Wavelength (nm)",
             ylabel="Absorbance")

# Train model with pharmaceutical focus
config = Config(
    input_size=X_processed.shape[1],
    hidden_sizes=[128, 64, 32],
    learning_rate=0.0005,  # Lower learning rate for precision
    batch_size=16,
    epochs=200,
    validation_split=0.2,
    early_stopping_patience=20
)

model = MLPRegressor(config)
model.fit(X_processed, y)

# Evaluate with pharmaceutical metrics
predictions = model.predict(X_processed)
metrics = calculate_metrics(y, predictions)

print("Pharmaceutical Analysis Results:")
print(f"R² Score: {metrics['r2_score']:.4f}")
print(f"RMSE: {metrics['rmse']:.4f}")
print(f"MAE: {metrics['mae']:.4f}")

# Concentration range analysis
concentration_ranges = [(0, 10), (10, 50), (50, 100)]
for low, high in concentration_ranges:
    mask = (y.values.flatten() >= low) & (y.values.flatten() < high)
    if mask.sum() > 0:
        range_metrics = calculate_metrics(y.values[mask], predictions[mask])
        print(f"Concentration {low}-{high}: R² = {range_metrics['r2_score']:.4f}")
```

## Performance Optimization

### Example 9: Hyperparameter Tuning

**Objective**: Optimize model performance through systematic hyperparameter tuning.

```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from uvvislib.evaluation.metrics import calculate_metrics

# Define parameter grids
mlp_params = {
    'hidden_sizes': [[64, 32], [128, 64], [256, 128, 64], [512, 256, 128, 64]],
    'learning_rate': [0.0001, 0.001, 0.01],
    'batch_size': [16, 32, 64],
    'dropout_rate': [0.0, 0.1, 0.2, 0.3]
}

# Grid search for MLP
best_score = 0
best_params = None

for hidden_sizes in mlp_params['hidden_sizes']:
    for lr in mlp_params['learning_rate']:
        for batch_size in mlp_params['batch_size']:
            for dropout in mlp_params['dropout_rate']:
                config = Config(
                    input_size=X_processed.shape[1],
                    hidden_sizes=hidden_sizes,
                    learning_rate=lr,
                    batch_size=batch_size,
                    dropout_rate=dropout,
                    epochs=50  # Shorter for tuning
                )
                
                model = MLPRegressor(config)
                cv_results = cv.cross_validate(model, X_processed, y)
                score = cv_results['r2_score'].mean()
                
                if score > best_score:
                    best_score = score
                    best_params = {
                        'hidden_sizes': hidden_sizes,
                        'learning_rate': lr,
                        'batch_size': batch_size,
                        'dropout_rate': dropout
                    }

print(f"Best parameters: {best_params}")
print(f"Best CV score: {best_score:.4f}")

# Train final model with best parameters
final_config = Config(
    input_size=X_processed.shape[1],
    **best_params,
    epochs=200
)

final_model = MLPRegressor(final_config)
final_model.fit(X_processed, y)
```

### Example 10: Ensemble Methods

**Objective**: Combine multiple models for improved performance.

```python
from uvvislib.models.mlp import MLPRegressor
from uvvislib.models.random_forest import RandomForestRegressor
from uvvislib.models.cnn import CNNRegressor

# Train multiple models
models = {
    'MLP': MLPRegressor(Config(
        input_size=X_processed.shape[1],
        hidden_sizes=[256, 128, 64],
        learning_rate=0.001,
        batch_size=32,
        epochs=100
    )),
    'CNN': CNNRegressor(Config(
        input_size=X_processed.shape[1],
        conv_layers=[32, 64],
        fc_layers=[128, 64],
        learning_rate=0.001,
        batch_size=16,
        epochs=150
    )),
    'RF': RandomForestRegressor(n_estimators=100, random_state=42)
}

# Train all models
predictions = {}
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_processed, y)
    predictions[name] = model.predict(X_processed)

# Ensemble predictions (simple average)
ensemble_pred = np.mean([pred for pred in predictions.values()], axis=0)

# Compare individual vs ensemble
for name, pred in predictions.items():
    metrics = calculate_metrics(y, pred)
    print(f"{name}: R² = {metrics['r2_score']:.4f}")

ensemble_metrics = calculate_metrics(y, ensemble_pred)
print(f"Ensemble: R² = {ensemble_metrics['r2_score']:.4f}")

# Weighted ensemble (based on individual performance)
weights = {}
for name, pred in predictions.items():
    metrics = calculate_metrics(y, pred)
    weights[name] = metrics['r2_score']

# Normalize weights
total_weight = sum(weights.values())
weights = {k: v/total_weight for k, v in weights.items()}

weighted_ensemble_pred = np.zeros_like(ensemble_pred)
for name, pred in predictions.items():
    weighted_ensemble_pred += weights[name] * pred

weighted_metrics = calculate_metrics(y, weighted_ensemble_pred)
print(f"Weighted Ensemble: R² = {weighted_metrics['r2_score']:.4f}")
```

## Custom Extensions

### Example 11: Custom Loss Function

**Objective**: Implement a custom loss function for specific requirements.

```python
import torch
import torch.nn as nn
from uvvislib.models.mlp import MLPRegressor

class CustomLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()
    
    def forward(self, pred, target):
        mse_loss = self.mse(pred, target)
        mae_loss = self.mae(pred, target)
        return self.alpha * mse_loss + (1 - self.alpha) * mae_loss

# Custom MLP with custom loss
class CustomMLPRegressor(MLPRegressor):
    def __init__(self, config, custom_loss=None):
        super().__init__(config)
        self.custom_loss = custom_loss or CustomLoss()
    
    def _create_criterion(self):
        return self.custom_loss

# Use custom model
config = Config(
    input_size=X_processed.shape[1],
    hidden_sizes=[128, 64],
    learning_rate=0.001,
    batch_size=32,
    epochs=100
)

custom_model = CustomMLPRegressor(config, CustomLoss(alpha=0.7))
custom_model.fit(X_processed, y)
```

### Example 12: Custom Preprocessing Pipeline

**Objective**: Create a custom preprocessing pipeline for specific data requirements.

```python
from uvvislib.data.preprocessing import preprocess_spectra
import numpy as np
from scipy.signal import savgol_filter

def custom_preprocessing(X, window_length=11, polyorder=3, **kwargs):
    """Custom preprocessing with Savitzky-Golay smoothing."""
    
    # Apply standard preprocessing
    X_processed = preprocess_spectra(X, **kwargs)
    
    # Apply Savitzky-Golay smoothing
    X_smoothed = np.apply_along_axis(
        lambda x: savgol_filter(x, window_length, polyorder),
        1, X_processed.values
    )
    
    return pd.DataFrame(X_smoothed, columns=X_processed.columns, index=X_processed.index)

# Use custom preprocessing
X_custom = custom_preprocessing(
    X,
    normalize=True,
    baseline_correction=True,
    window_length=15,
    polyorder=3
)

# Train model with custom preprocessing
model = MLPRegressor(config)
model.fit(X_custom, y)
```

---

These examples demonstrate the versatility and power of the UV-Vis Research Library. Each example can be adapted and extended for your specific research needs. For more advanced usage patterns and best practices, see the [Best Practices](advanced/best_practices.md) section. 