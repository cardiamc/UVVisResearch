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
git clone https://github.com/cardiamc/UVVisResearch.git
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
from uvvislib import (
    Config, DataLoader, Preprocessor, MLPRegressor, compute_evaluation,
)

# Configuration drives every component (paths, target list, CV seed, etc.)
config = Config(
    data_path="./Data/",
    uv_vis_data_file="abs_spectra_100mm.csv",
    chemical_data_file="Full_chemical_analysis.csv",
    target_variables=["COD"],
    log_target=True,  # COD is predicted in log space
)

# Load + combine the spectral and chemical data
loader = DataLoader(config)
loader.load_uv_vis_data()
loader.load_chemical_data()
loader.combine_data()
features_df, targets_df = loader.extract_features_and_targets()

# Canonical preprocessing: Gaussian smoothing (sigma=1.5) + MinMax + log target
pre = Preprocessor(config)
X_train, y_train = pre.preprocess_pipeline(features_df, targets_df)

# Train an MLP regressor
model = MLPRegressor(config, hidden_size=200, learning_rate=1e-3, epochs=2000)
model.fit(X_train.values, y_train.values)

predictions = model.predict(X_train.values)
metrics = compute_evaluation(y_train.values, predictions, log_target=config.log_target)
print(f"R² = {metrics['r2']:.4f}, RMSE = {metrics['rmse']:.4f}")
```

### Advanced Usage with Cross-Validation

```python
from uvvislib import (
    Config, CNNRegressor, DoubleKFoldCV, ExperimentManager,
)

config = Config(target_variables=["COD"], log_target=True)

cv = DoubleKFoldCV(outer_splits=5, inner_splits=4, random_state=42)
param_grid = {
    "hidden_size": [200, 400, 800],
    "kernel_size": [3, 5],
    "learning_rate": [1e-4, 5e-4, 1e-3],
}

results = cv.fit(
    X_train.values, y_train.values,
    model=CNNRegressor(config),
    param_grid=param_grid,
    n_iter=20,
    log_target=config.log_target,
)

exp = ExperimentManager(config, base_path="./experiments")
experiment_id = exp.start_experiment("cnn_cv", description="Nested CV on COD")
exp.save_results(results, filename="cv_results.json")
exp.end_experiment()
```

### Clustering Analysis

```python
from uvvislib import Config, SpectralClusterer, Plotter

config = Config()
clusterer = SpectralClusterer(config, algorithm="kmeans", n_clusters=3)
clusterer.fit(X_train.values)
labels = clusterer.predict(X_train.values)

Plotter(config).plot_clusters(X_train.values, labels, title="Spectral clusters")
```

## 🔧 Configuration

The library uses a centralized `Config` dataclass. The defaults below match the
canonical UV-Vis pipeline (200–727.5 nm at 2.5 nm step → 212 features, Gaussian
smoothing, MinMax scaling).

```python
from uvvislib import Config

config = Config(
    # Spectral grid (212 wavelengths)
    wavelength_start=200.0,
    wavelength_end=730.0,
    wavelength_step=2.5,

    # Preprocessing
    apply_smoothing=True,
    gaussian_sigma=1.5,
    log_target=True,

    # Cross-validation
    k_fold_splits=5,
    k_fold_inner_splits=4,
    random_state=42,

    # Output
    output_dir="./LOG/",
    save_models=True,
)
```

## 📊 Available Models

### Neural Networks
- **MLPRegressor**: Multi-layer perceptron for regression
- **CNNRegressor**: Convolutional neural network for spectral data
- **CNNMLPRegressor**: Hybrid CNN-MLP architecture (spectra + extracted features)

### Traditional ML
- **RandomForestRegressor**: Random Forest implementation
- **SpectralClusterer**: Clustering (kmeans, hierarchical, dbscan, gmm) for spectral data

## 📈 Evaluation Metrics

The library provides comprehensive evaluation metrics:
- R² Score
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- Explained Variance Score

## 🎯 Model Interpretation

```python
from uvvislib import Config, RandomForestRegressor

rf_model = RandomForestRegressor(Config(), n_estimators=100, random_state=42)
rf_model.fit(X_train.values, y_train.values)

importance = rf_model.get_feature_importance()
print("Top 10 most important features:", importance[:10])
```

## 💾 Experiment Management

```python
from uvvislib import Config, ExperimentManager

exp = ExperimentManager(Config(), base_path="./experiments")
experiment_id = exp.start_experiment(
    "mlp_baseline",
    description="MLP on COD with default Gaussian + MinMax preprocessing",
    tags=["mlp", "cod"],
)

exp.save_results({"r2": 0.91, "rmse": 0.18})
exp.end_experiment(status="completed")
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