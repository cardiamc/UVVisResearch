# UV-Vis Research Library (uvvislib)

A comprehensive, modular Python library for UV-Vis and NIR spectral data analysis, machine learning model development, and scientific research applications. Built with professional software engineering practices, this library provides a complete toolkit for spectral data processing, model training, evaluation, and interpretation.

## 🚀 Features

### Core Functionality
- **Data Management**: Robust data loading, preprocessing, and persistence
- **Machine Learning Models**: MLP, CNN, Random Forest, and clustering algorithms
- **Evaluation Framework**: Comprehensive metrics and cross-validation
- **Model Interpretation**: Feature importance analysis and model explainability
- **Visualization**: Advanced plotting and data visualization tools
- **Experiment Management**: Save, load, and track experiments

### Advanced Capabilities
- **Modular Architecture**: Clean separation of concerns with professional design patterns
- **Configuration Management**: Centralized settings and parameter management
- **Logging System**: Comprehensive logging for debugging and experiment tracking
- **Early Stopping**: Intelligent training optimization
- **Cross-Validation**: Robust model evaluation with multiple validation strategies

## 📁 Library Structure

```
uvvislib/
├── data/                 # Data handling modules
│   ├── loader.py        # Data loading utilities
│   └── preprocessing.py # Data preprocessing functions
├── models/              # Machine learning models
│   ├── base.py         # Base model interface
│   ├── mlp.py          # Multi-layer perceptron
│   ├── cnn.py          # Convolutional neural network
│   ├── cnn_mlp.py      # Hybrid CNN-MLP model
│   ├── random_forest.py # Random Forest implementation
│   └── clustering.py   # Clustering algorithms
├── evaluation/          # Model evaluation
│   ├── metrics.py      # Performance metrics
│   └── cross_validation.py # Cross-validation utilities
├── persistence/         # Data and model persistence
│   ├── data_persistence.py # Data saving/loading
│   ├── model_manager.py   # Model serialization
│   └── experiment_manager.py # Experiment tracking
├── utils/              # Utility modules
│   ├── config.py       # Configuration management
│   ├── logging.py      # Logging utilities
│   └── early_stopping.py # Early stopping mechanism
└── visualization/      # Visualization tools
    └── plots.py        # Plotting functions
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- pip

### Install from source
```bash
# Clone the repository
git clone <repository-url>
cd UVVisResearch

# Install in development mode
pip install -e .

# Install dependencies
pip install -r requirements.txt
```

### Dependencies
The library requires the following key dependencies:
- `numpy` - Numerical computing
- `pandas` - Data manipulation
- `scikit-learn` - Machine learning utilities
- `torch` - Deep learning framework
- `matplotlib` - Visualization
- `seaborn` - Statistical visualization

## 📖 Quick Start

### Basic Usage Example

```python
from uvvislib.data.loader import load_data
from uvvislib.data.preprocessing import preprocess_spectra
from uvvislib.models.mlp import MLPRegressor
from uvvislib.evaluation.metrics import calculate_metrics
from uvvislib.utils.config import Config

# Load and preprocess data
X, y = load_data('spectral_data.csv')
X_processed = preprocess_spectra(X, normalize=True, baseline_correction=True)

# Configure and train model
config = Config(
    input_size=X_processed.shape[1],
    hidden_sizes=[128, 64],
    learning_rate=0.001,
    batch_size=32,
    epochs=100
)

model = MLPRegressor(config)
model.fit(X_processed, y)

# Evaluate model
predictions = model.predict(X_processed)
metrics = calculate_metrics(y, predictions)
print(f"R² Score: {metrics['r2_score']:.4f}")
```

### Advanced Usage with Cross-Validation

```python
from uvvislib.evaluation.cross_validation import CrossValidator
from uvvislib.models.cnn import CNNRegressor
from uvvislib.persistence.experiment_manager import ExperimentManager

# Set up cross-validation
cv = CrossValidator(n_splits=5, random_state=42)

# Configure CNN model
config = Config(
    input_size=X.shape[1],
    conv_layers=[32, 64],
    fc_layers=[128, 64],
    learning_rate=0.001
)

# Train with cross-validation
model = CNNRegressor(config)
cv_results = cv.cross_validate(model, X, y)

# Save experiment
exp_manager = ExperimentManager()
exp_manager.save_experiment(
    model=model,
    results=cv_results,
    config=config,
    name="cnn_cv_experiment"
)
```

### Clustering Analysis

```python
from uvvislib.models.clustering import SpectralClustering
from uvvislib.visualization.plots import plot_clustering_results

# Perform clustering
clustering = SpectralClustering(n_clusters=3, method='kmeans')
cluster_labels = clustering.fit_predict(X)

# Visualize results
plot_clustering_results(X, cluster_labels, title="Spectral Clustering Results")
```

## 🔧 Configuration

The library uses a centralized configuration system:

```python
from uvvislib.utils.config import Config

config = Config(
    # Model parameters
    input_size=1000,
    hidden_sizes=[256, 128, 64],
    learning_rate=0.001,
    batch_size=32,
    epochs=200,
    
    # Training parameters
    validation_split=0.2,
    early_stopping_patience=10,
    
    # Data preprocessing
    normalize=True,
    baseline_correction=True,
    
    # Logging
    log_level="INFO",
    save_logs=True
)
```

## 📊 Available Models

### Neural Networks
- **MLPRegressor**: Multi-layer perceptron for regression
- **CNNRegressor**: Convolutional neural network for spectral data
- **CNNMLPRegressor**: Hybrid CNN-MLP architecture

### Traditional ML
- **RandomForestRegressor**: Random Forest implementation
- **SpectralClustering**: Clustering algorithms for spectral data

## 📈 Evaluation Metrics

The library provides comprehensive evaluation metrics:
- R² Score
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- Explained Variance Score

## 🎯 Model Interpretation

```python
from uvvislib.models.random_forest import RandomForestRegressor

# Train Random Forest
rf_model = RandomForestRegressor(n_estimators=100)
rf_model.fit(X, y)

# Get feature importance
importance = rf_model.get_feature_importance()
print("Top 10 most important features:", importance[:10])
```

## 💾 Experiment Management

```python
from uvvislib.persistence.experiment_manager import ExperimentManager

# Save experiment
exp_manager = ExperimentManager()
exp_manager.save_experiment(
    model=model,
    results=results,
    config=config,
    name="experiment_001",
    description="MLP with baseline correction"
)

# Load experiment
loaded_exp = exp_manager.load_experiment("experiment_001")
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=uvvislib --cov-report=html
```

## 📚 Documentation

- **API Documentation**: See `docs/` folder for detailed API reference
- **Examples**: Check `examples/` folder for comprehensive usage examples
- **Tests**: Review `tests/` folder for usage patterns and edge cases

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For questions, issues, or contributions, please:
1. Check the documentation in the `docs/` folder
2. Review existing issues
3. Create a new issue with detailed information

---

**Built with ❤️ for the scientific community** 