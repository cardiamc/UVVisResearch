# API Reference

This document provides a comprehensive reference for all classes, functions, and methods in the UV-Vis Research Library.

## Table of Contents

- [Data Management](#data-management)
- [Models](#models)
- [Evaluation](#evaluation)
- [Utilities](#utilities)
- [Visualization](#visualization)
- [Persistence](#persistence)

## Data Management

### Data Loading

#### `load_data(file_path, feature_columns=None, target_columns=None, **kwargs)`

Load spectral data from various file formats.

**Parameters:**
- `file_path` (str): Path to the data file
- `feature_columns` (list/range, optional): Column indices or names for features
- `target_columns` (list, optional): Column names for target variables
- `**kwargs`: Additional arguments passed to pandas read functions

**Returns:**
- `tuple`: (X, y) where X is feature DataFrame and y is target DataFrame

**Example:**
```python
from uvvislib.data.loader import load_data

# Load data with specific wavelength range
X, y = load_data('spectra.csv', 
                 feature_columns=range(200, 800, 2),
                 target_columns=['concentration', 'pH'])
```

### Data Preprocessing

#### `preprocess_spectra(X, normalize=True, baseline_correction=False, smoothing=False, smoothing_window=5, **kwargs)`

Apply preprocessing techniques to spectral data.

**Parameters:**
- `X` (DataFrame): Input spectral data
- `normalize` (bool): Whether to standardize the data
- `baseline_correction` (bool): Whether to apply baseline correction
- `smoothing` (bool): Whether to apply smoothing
- `smoothing_window` (int): Window size for smoothing
- `**kwargs`: Additional preprocessing parameters

**Returns:**
- `DataFrame`: Preprocessed spectral data

**Example:**
```python
from uvvislib.data.preprocessing import preprocess_spectra

X_processed = preprocess_spectra(
    X,
    normalize=True,
    baseline_correction=True,
    smoothing=True,
    smoothing_window=7
)
```

## Models

### Base Model Interface

#### `BaseModel`

Abstract base class for all models in the library.

**Methods:**
- `fit(X, y)`: Train the model
- `predict(X)`: Make predictions
- `get_params()`: Get model parameters
- `set_params(**params)`: Set model parameters

### MLP Models

#### `MLPRegressor(config)`

Multi-layer perceptron regressor for spectral data.

**Parameters:**
- `config` (Config): Configuration object containing model parameters

**Configuration Parameters:**
- `input_size` (int): Number of input features
- `hidden_sizes` (list): List of hidden layer sizes
- `learning_rate` (float): Learning rate for optimization
- `batch_size` (int): Batch size for training
- `epochs` (int): Number of training epochs
- `dropout_rate` (float, optional): Dropout rate for regularization
- `activation` (str): Activation function ('relu', 'sigmoid', 'tanh')

**Methods:**
- `fit(X, y)`: Train the MLP
- `predict(X)`: Make predictions
- `plot_training_history()`: Plot training curves
- `get_feature_importance()`: Get feature importance scores

**Example:**
```python
from uvvislib.models.mlp import MLPRegressor
from uvvislib.utils.config import Config

config = Config(
    input_size=1000,
    hidden_sizes=[256, 128, 64],
    learning_rate=0.001,
    batch_size=32,
    epochs=100
)

model = MLPRegressor(config)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

#### `CNNRegressor(config)`

Convolutional neural network regressor for spectral data.

**Configuration Parameters:**
- `input_size` (int): Number of input features
- `conv_layers` (list): List of convolutional layer sizes
- `fc_layers` (list): List of fully connected layer sizes
- `kernel_size` (int): Size of convolutional kernels
- `stride` (int): Stride for convolutions
- `learning_rate` (float): Learning rate
- `batch_size` (int): Batch size
- `epochs` (int): Number of epochs

**Example:**
```python
config = Config(
    input_size=1000,
    conv_layers=[32, 64],
    fc_layers=[128, 64],
    kernel_size=3,
    stride=1,
    learning_rate=0.001,
    batch_size=16,
    epochs=200
)

cnn_model = CNNRegressor(config)
cnn_model.fit(X_train, y_train)
```

#### `CNNMLPRegressor(config)`

Hybrid CNN-MLP model combining convolutional and fully connected layers.

**Configuration Parameters:**
- `input_size` (int): Number of input features
- `conv_layers` (list): Convolutional layer configuration
- `mlp_layers` (list): MLP layer configuration
- `learning_rate` (float): Learning rate
- `batch_size` (int): Batch size
- `epochs` (int): Number of epochs

### Random Forest

#### `RandomForestRegressor(n_estimators=100, max_depth=None, random_state=None, **kwargs)`

Random Forest regressor for spectral data.

**Parameters:**
- `n_estimators` (int): Number of trees in the forest
- `max_depth` (int, optional): Maximum depth of trees
- `random_state` (int, optional): Random seed for reproducibility
- `**kwargs`: Additional scikit-learn RandomForestRegressor parameters

**Methods:**
- `fit(X, y)`: Train the Random Forest
- `predict(X)`: Make predictions
- `get_feature_importance()`: Get feature importance scores
- `get_params()`: Get model parameters

**Example:**
```python
from uvvislib.models.random_forest import RandomForestRegressor

rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42
)
rf_model.fit(X_train, y_train)
importance = rf_model.get_feature_importance()
```

### Clustering

#### `SpectralClustering(n_clusters=3, method='kmeans', random_state=None, **kwargs)`

Clustering algorithms for spectral data.

**Parameters:**
- `n_clusters` (int): Number of clusters
- `method` (str): Clustering method ('kmeans', 'hierarchical', 'dbscan')
- `random_state` (int, optional): Random seed
- `**kwargs`: Additional clustering parameters

**Methods:**
- `fit(X)`: Fit the clustering model
- `fit_predict(X)`: Fit and predict cluster labels
- `predict(X)`: Predict cluster labels for new data
- `get_cluster_centers()`: Get cluster centers

**Example:**
```python
from uvvislib.models.clustering import SpectralClustering

clustering = SpectralClustering(
    n_clusters=3,
    method='kmeans',
    random_state=42
)
cluster_labels = clustering.fit_predict(X)
```

## Evaluation

### Metrics

#### `calculate_metrics(y_true, y_pred, metrics=None)`

Calculate various performance metrics.

**Parameters:**
- `y_true` (array-like): True target values
- `y_pred` (array-like): Predicted target values
- `metrics` (list, optional): List of metrics to calculate

**Returns:**
- `dict`: Dictionary containing calculated metrics

**Available Metrics:**
- `'r2_score'`: R-squared score
- `'mse'`: Mean squared error
- `'rmse'`: Root mean squared error
- `'mae'`: Mean absolute error
- `'explained_variance'`: Explained variance score

**Example:**
```python
from uvvislib.evaluation.metrics import calculate_metrics

metrics = calculate_metrics(y_true, y_pred)
print(f"R² Score: {metrics['r2_score']:.4f}")
print(f"RMSE: {metrics['rmse']:.4f}")
```

### Cross-Validation

#### `CrossValidator(n_splits=5, random_state=None, shuffle=True)`

Cross-validation utility for model evaluation.

**Parameters:**
- `n_splits` (int): Number of CV folds
- `random_state` (int, optional): Random seed
- `shuffle` (bool): Whether to shuffle data before splitting

**Methods:**
- `cross_validate(model, X, y, metrics=None)`: Perform cross-validation
- `split(X, y)`: Generate train/test splits

**Example:**
```python
from uvvislib.evaluation.cross_validation import CrossValidator

cv = CrossValidator(n_splits=5, random_state=42)
cv_results = cv.cross_validate(model, X, y)
print(f"CV R²: {cv_results['r2_score'].mean():.4f} ± {cv_results['r2_score'].std():.4f}")
```

## Utilities

### Configuration

#### `Config(**kwargs)`

Configuration class for model and training parameters.

**Common Parameters:**
- `input_size` (int): Number of input features
- `hidden_sizes` (list): Hidden layer sizes for neural networks
- `learning_rate` (float): Learning rate
- `batch_size` (int): Batch size
- `epochs` (int): Number of training epochs
- `validation_split` (float): Validation split ratio
- `early_stopping_patience` (int): Early stopping patience
- `dropout_rate` (float): Dropout rate
- `random_state` (int): Random seed

**Example:**
```python
from uvvislib.utils.config import Config

config = Config(
    input_size=1000,
    hidden_sizes=[256, 128, 64],
    learning_rate=0.001,
    batch_size=32,
    epochs=100,
    validation_split=0.2,
    early_stopping_patience=10
)
```

### Logging

#### `setup_logging(level='INFO', log_file=None, format_string=None)`

Set up logging configuration.

**Parameters:**
- `level` (str): Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
- `log_file` (str, optional): Path to log file
- `format_string` (str, optional): Custom log format

**Example:**
```python
from uvvislib.utils.logging import setup_logging

setup_logging(level="INFO", log_file="experiment.log")
```

### Early Stopping

#### `EarlyStopping(patience=10, min_delta=0.0, restore_best_weights=True)`

Early stopping mechanism for training optimization.

**Parameters:**
- `patience` (int): Number of epochs to wait before stopping
- `min_delta` (float): Minimum change to qualify as improvement
- `restore_best_weights` (bool): Whether to restore best weights

**Methods:**
- `__call__(val_loss)`: Check if training should stop
- `reset()`: Reset the early stopper state

## Visualization

### Plotting Functions

#### `plot_spectra(X, title=None, xlabel=None, ylabel=None, **kwargs)`

Plot spectral data.

**Parameters:**
- `X` (DataFrame): Spectral data to plot
- `title` (str, optional): Plot title
- `xlabel` (str, optional): X-axis label
- `ylabel` (str, optional): Y-axis label
- `**kwargs`: Additional matplotlib parameters

#### `plot_predictions_vs_actual(y_true, y_pred, title=None, **kwargs)`

Plot predictions vs actual values.

**Parameters:**
- `y_true` (array-like): True values
- `y_pred` (array-like): Predicted values
- `title` (str, optional): Plot title
- `**kwargs`: Additional matplotlib parameters

#### `plot_clustering_results(X, cluster_labels, title=None, **kwargs)`

Plot clustering results.

**Parameters:**
- `X` (DataFrame): Input data
- `cluster_labels` (array-like): Cluster assignments
- `title` (str, optional): Plot title
- `**kwargs`: Additional matplotlib parameters

**Example:**
```python
from uvvislib.visualization.plots import plot_spectra, plot_predictions_vs_actual

# Plot spectra
plot_spectra(X, title="Sample Spectra", xlabel="Wavelength (nm)", ylabel="Absorbance")

# Plot predictions
plot_predictions_vs_actual(y_true, y_pred, title="Model Performance")
```

## Persistence

### Model Management

#### `ModelManager()`

Manager for saving and loading models.

**Methods:**
- `save_model(model, filepath)`: Save model to file
- `load_model(filepath)`: Load model from file
- `get_model_info(filepath)`: Get model information

**Example:**
```python
from uvvislib.persistence.model_manager import ModelManager

# Save model
model_manager = ModelManager()
model_manager.save_model(model, "my_model.pkl")

# Load model
loaded_model = model_manager.load_model("my_model.pkl")
```

### Experiment Management

#### `ExperimentManager()`

Manager for saving and loading experiments.

**Methods:**
- `save_experiment(model, results, config, name, description=None)`: Save experiment
- `load_experiment(name)`: Load experiment by name
- `list_experiments()`: List all saved experiments
- `delete_experiment(name)`: Delete experiment

**Example:**
```python
from uvvislib.persistence.experiment_manager import ExperimentManager

# Save experiment
exp_manager = ExperimentManager()
exp_manager.save_experiment(
    model=model,
    results=metrics,
    config=config,
    name="experiment_001",
    description="MLP with baseline correction"
)

# Load experiment
loaded_exp = exp_manager.load_experiment("experiment_001")
```

### Data Persistence

#### `save_data(data, filepath, format='csv')`

Save data to various formats.

**Parameters:**
- `data` (DataFrame): Data to save
- `filepath` (str): Output file path
- `format` (str): Output format ('csv', 'pkl', 'h5')

#### `load_data(filepath, format='csv')`

Load data from various formats.

**Parameters:**
- `filepath` (str): Input file path
- `format` (str): Input format ('csv', 'pkl', 'h5')

**Returns:**
- `DataFrame`: Loaded data

---

This API reference covers the main functionality of the UV-Vis Research Library. For more detailed examples and advanced usage patterns, see the [Examples Gallery](examples.md) and [Best Practices](advanced/best_practices.md) sections. 