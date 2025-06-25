# uvvislib

A modular, professional Python library for UV-Vis and NIR data analysis, machine learning, and model interpretation.

## Features
- Data loading and preprocessing
- Feature engineering
- Model training (MLP, CNN, Random Forest, etc.)
- Model evaluation and cross-validation
- Feature importance and interpretation
- Visualization utilities
- Model persistence (save/load)

## Installation
```bash
pip install -e .
```

## Usage Example
```python
from uvvislib.data.loader import load_data
from uvvislib.models.mlp import MLPRegressor

X, y = load_data('data.csv')
model = MLPRegressor(...)
model.fit(X, y)
preds = model.predict(X)
```

## Documentation
See the `docs/` folder for full API documentation and examples. 