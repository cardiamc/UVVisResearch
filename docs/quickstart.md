# Quick Start Guide

This guide will get you up and running with the UV-Vis Research Library in minutes. You'll learn how to load data, train models, and evaluate results.

## Prerequisites

Before starting, make sure you have:
- ✅ Installed the library (see [Installation Guide](installation.md))
- ✅ Basic knowledge of Python
- ✅ Some spectral data to work with

## 🚀 Your First Analysis

### Step 1: Import the Library

```python
import numpy as np
import pandas as pd
from uvvislib.data.loader import load_data
from uvvislib.data.preprocessing import preprocess_spectra
from uvvislib.models.mlp import MLPRegressor
from uvvislib.evaluation.metrics import calculate_metrics
from uvvislib.utils.config import Config
from uvvislib.utils.logging import setup_logging

# Set up logging
setup_logging(level="INFO")
```

### Step 2: Load Your Data

```python
# Load spectral data
# Assuming you have a CSV file with spectral data and target variables
X, y = load_data('your_spectral_data.csv', 
                 feature_columns=range(200, 800, 2),  # Wavelength range
                 target_columns=['target1', 'target2'])

print(f"Data shape: X={X.shape}, y={y.shape}")
print(f"Wavelength range: {X.columns[0]} - {X.columns[-1]} nm")
```

### Step 3: Preprocess the Data

```python
# Apply preprocessing
X_processed = preprocess_spectra(
    X,
    normalize=True,           # Standardize the data
    baseline_correction=True, # Remove baseline drift
    smoothing=True,           # Apply smoothing
    smoothing_window=5        # Window size for smoothing
)

print("Data preprocessing completed!")
```

### Step 4: Configure Your Model

```python
# Create configuration
config = Config(
    input_size=X_processed.shape[1],  # Number of spectral features
    hidden_sizes=[128, 64, 32],       # Hidden layer sizes
    learning_rate=0.001,              # Learning rate
    batch_size=32,                    # Batch size
    epochs=100,                       # Number of training epochs
    validation_split=0.2,             # Validation split
    early_stopping_patience=10        # Early stopping patience
)

print("Configuration created successfully!")
```

### Step 5: Train Your Model

```python
# Create and train the model
model = MLPRegressor(config)
model.fit(X_processed, y)

print("Model training completed!")
```

### Step 6: Evaluate Results

```python
# Make predictions
predictions = model.predict(X_processed)

# Calculate metrics
metrics = calculate_metrics(y, predictions)

print("Model Performance:")
print(f"R² Score: {metrics['r2_score']:.4f}")
print(f"RMSE: {metrics['rmse']:.4f}")
print(f"MAE: {metrics['mae']:.4f}")
```

## 📊 Common Workflows

### Workflow 1: Basic Regression Analysis

```python
from uvvislib.models.random_forest import RandomForestRegressor
from uvvislib.evaluation.cross_validation import CrossValidator

# Load and preprocess data
X, y = load_data('spectral_data.csv')
X_processed = preprocess_spectra(X, normalize=True)

# Train Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_processed, y)

# Cross-validation
cv = CrossValidator(n_splits=5, random_state=42)
cv_results = cv.cross_validate(rf_model, X_processed, y)

print(f"Cross-validation R²: {cv_results['r2_score'].mean():.4f} ± {cv_results['r2_score'].std():.4f}")
```

### Workflow 2: Deep Learning with CNN

```python
from uvvislib.models.cnn import CNNRegressor

# Configure CNN
config = Config(
    input_size=X_processed.shape[1],
    conv_layers=[32, 64],      # Convolutional layers
    fc_layers=[128, 64],       # Fully connected layers
    learning_rate=0.001,
    batch_size=16,
    epochs=200
)

# Train CNN
cnn_model = CNNRegressor(config)
cnn_model.fit(X_processed, y)

# Evaluate
predictions = cnn_model.predict(X_processed)
metrics = calculate_metrics(y, predictions)
print(f"CNN R² Score: {metrics['r2_score']:.4f}")
```

### Workflow 3: Clustering Analysis

```python
from uvvislib.models.clustering import SpectralClustering
from uvvislib.visualization.plots import plot_clustering_results

# Perform clustering
clustering = SpectralClustering(n_clusters=3, method='kmeans')
cluster_labels = clustering.fit_predict(X_processed)

# Visualize results
plot_clustering_results(X_processed, cluster_labels, 
                       title="Spectral Clustering Results")
```

### Workflow 4: Feature Importance Analysis

```python
# Get feature importance from Random Forest
importance = rf_model.get_feature_importance()

# Plot feature importance
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.bar(range(len(importance)), importance)
plt.xlabel('Wavelength Index')
plt.ylabel('Feature Importance')
plt.title('Random Forest Feature Importance')
plt.show()

# Get top important features
top_features = np.argsort(importance)[-10:]  # Top 10 features
print("Top 10 important wavelengths:", X_processed.columns[top_features])
```

## 🔧 Configuration Examples

### Basic Configuration

```python
config = Config(
    input_size=1000,
    hidden_sizes=[256, 128, 64],
    learning_rate=0.001,
    batch_size=32,
    epochs=100
)
```

### Advanced Configuration

```python
config = Config(
    # Model architecture
    input_size=1000,
    hidden_sizes=[512, 256, 128, 64],
    dropout_rate=0.2,
    
    # Training parameters
    learning_rate=0.001,
    batch_size=32,
    epochs=200,
    validation_split=0.2,
    early_stopping_patience=15,
    
    # Data preprocessing
    normalize=True,
    baseline_correction=True,
    smoothing=True,
    smoothing_window=5,
    
    # Logging
    log_level="INFO",
    save_logs=True,
    log_dir="./logs"
)
```

## 📈 Visualization Examples

### Plot Training History

```python
# Plot training curves
model.plot_training_history()
```

### Plot Predictions vs Actual

```python
from uvvislib.visualization.plots import plot_predictions_vs_actual

plot_predictions_vs_actual(y, predictions, 
                          title="Model Predictions vs Actual Values")
```

### Plot Spectral Data

```python
from uvvislib.visualization.plots import plot_spectra

# Plot first 10 spectra
plot_spectra(X_processed.iloc[:10], 
             title="Sample Spectra",
             xlabel="Wavelength (nm)",
             ylabel="Absorbance")
```

## 💾 Saving and Loading

### Save Your Model

```python
from uvvislib.persistence.model_manager import ModelManager

# Save model
model_manager = ModelManager()
model_manager.save_model(model, "my_mlp_model.pkl")

# Save experiment
from uvvislib.persistence.experiment_manager import ExperimentManager
exp_manager = ExperimentManager()
exp_manager.save_experiment(
    model=model,
    results=metrics,
    config=config,
    name="experiment_001",
    description="MLP with baseline correction"
)
```

### Load Your Model

```python
# Load model
loaded_model = model_manager.load_model("my_mlp_model.pkl")

# Load experiment
loaded_exp = exp_manager.load_experiment("experiment_001")
```

## 🎯 Next Steps

Now that you've completed the quick start:

1. **Explore the [Examples Gallery](examples.md)** for more complex workflows
2. **Read the [Configuration Guide](configuration.md)** for advanced customization
3. **Check [Best Practices](advanced/best_practices.md)** for optimal performance
4. **Review the [API Reference](api_reference.md)** for detailed documentation

## 🆘 Getting Help

If you encounter issues:

1. Check the [Troubleshooting](advanced/troubleshooting.md) section
2. Review the error messages carefully
3. Ensure your data format is correct
4. Try the examples in this guide
5. Open an issue on the project repository

---

**Congratulations! You've successfully completed the quick start guide!** 🎉

You now have the basic skills to work with the UV-Vis Research Library. Continue exploring the documentation to unlock the full potential of the library. 