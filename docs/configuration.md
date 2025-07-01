# Configuration Guide

This guide explains how to use the configuration system in the UV-Vis Research Library to customize model behavior, training parameters, and data preprocessing.

## Overview

The library uses a centralized `Config` class to manage all parameters for models, training, and preprocessing. This provides a clean, consistent interface for configuring all components.

## Basic Configuration

### Creating a Configuration

```python
from uvvislib.utils.config import Config

# Basic configuration
config = Config(
    input_size=1000,
    hidden_sizes=[256, 128, 64],
    learning_rate=0.001,
    batch_size=32,
    epochs=100
)
```

### Configuration Parameters

#### Model Architecture Parameters

- `input_size` (int): Number of input features (spectral wavelengths)
- `hidden_sizes` (list): List of hidden layer sizes for neural networks
- `conv_layers` (list): List of convolutional layer sizes for CNN models
- `fc_layers` (list): List of fully connected layer sizes for CNN models
- `mlp_layers` (list): List of MLP layer sizes for hybrid models
- `kernel_size` (int): Size of convolutional kernels
- `stride` (int): Stride for convolutions
- `dropout_rate` (float): Dropout rate for regularization (0.0 to 1.0)
- `activation` (str): Activation function ('relu', 'sigmoid', 'tanh')

#### Training Parameters

- `learning_rate` (float): Learning rate for optimization
- `batch_size` (int): Batch size for training
- `epochs` (int): Number of training epochs
- `validation_split` (float): Validation split ratio (0.0 to 1.0)
- `early_stopping_patience` (int): Number of epochs to wait before early stopping
- `optimizer` (str): Optimizer type ('adam', 'sgd', 'rmsprop')
- `weight_decay` (float): Weight decay for regularization
- `momentum` (float): Momentum for SGD optimizer

#### Data Preprocessing Parameters

- `normalize` (bool): Whether to standardize the data
- `baseline_correction` (bool): Whether to apply baseline correction
- `smoothing` (bool): Whether to apply smoothing
- `smoothing_window` (int): Window size for smoothing
- `wavelength_range` (tuple): Wavelength range to use (start, end, step)

#### Logging and Output Parameters

- `log_level` (str): Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
- `save_logs` (bool): Whether to save logs to file
- `log_dir` (str): Directory for log files
- `save_model` (bool): Whether to save the trained model
- `model_dir` (str): Directory for saved models

## Configuration Examples

### MLP Configuration

```python
# Basic MLP configuration
mlp_config = Config(
    input_size=1000,
    hidden_sizes=[256, 128, 64],
    learning_rate=0.001,
    batch_size=32,
    epochs=100,
    validation_split=0.2,
    early_stopping_patience=10,
    dropout_rate=0.2,
    activation='relu'
)

# Advanced MLP configuration
advanced_mlp_config = Config(
    input_size=1000,
    hidden_sizes=[512, 256, 128, 64, 32],
    learning_rate=0.0005,
    batch_size=16,
    epochs=200,
    validation_split=0.2,
    early_stopping_patience=15,
    dropout_rate=0.3,
    activation='relu',
    optimizer='adam',
    weight_decay=0.001,
    normalize=True,
    baseline_correction=True,
    smoothing=True,
    smoothing_window=5,
    log_level='INFO',
    save_logs=True,
    log_dir='./logs'
)
```

### CNN Configuration

```python
# CNN configuration
cnn_config = Config(
    input_size=1000,
    conv_layers=[32, 64, 128],
    fc_layers=[256, 128, 64],
    kernel_size=5,
    stride=1,
    learning_rate=0.001,
    batch_size=16,
    epochs=200,
    validation_split=0.2,
    early_stopping_patience=15,
    dropout_rate=0.2,
    activation='relu'
)

# Lightweight CNN configuration
lightweight_cnn_config = Config(
    input_size=1000,
    conv_layers=[16, 32],
    fc_layers=[64, 32],
    kernel_size=3,
    stride=2,
    learning_rate=0.01,
    batch_size=32,
    epochs=100,
    validation_split=0.2
)
```

### Hybrid CNN-MLP Configuration

```python
# Hybrid model configuration
hybrid_config = Config(
    input_size=1000,
    conv_layers=[32, 64],
    mlp_layers=[256, 128, 64],
    kernel_size=3,
    stride=1,
    learning_rate=0.001,
    batch_size=32,
    epochs=150,
    validation_split=0.2,
    early_stopping_patience=10,
    dropout_rate=0.2
)
```

### Random Forest Configuration

```python
# Random Forest configuration
rf_config = Config(
    n_estimators=100,
    max_depth=10,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    normalize=True,
    baseline_correction=True
)
```

## Configuration Best Practices

### 1. Start Simple

Begin with a simple configuration and gradually add complexity:

```python
# Start with basic configuration
config = Config(
    input_size=1000,
    hidden_sizes=[64, 32],
    learning_rate=0.001,
    batch_size=32,
    epochs=50
)

# Gradually add features
config = Config(
    input_size=1000,
    hidden_sizes=[64, 32],
    learning_rate=0.001,
    batch_size=32,
    epochs=50,
    validation_split=0.2,  # Add validation
    early_stopping_patience=10,  # Add early stopping
    dropout_rate=0.1  # Add regularization
)
```

### 2. Use Appropriate Learning Rates

```python
# For different model types
mlp_config = Config(learning_rate=0.001)  # Standard for MLP
cnn_config = Config(learning_rate=0.001)  # Standard for CNN
rf_config = Config()  # Not applicable for Random Forest
```

### 3. Choose Appropriate Batch Sizes

```python
# Based on data size and memory constraints
small_dataset_config = Config(batch_size=16)  # Small dataset
large_dataset_config = Config(batch_size=64)  # Large dataset
memory_constrained_config = Config(batch_size=8)  # Limited memory
```

### 4. Configure Early Stopping

```python
# Prevent overfitting
config = Config(
    epochs=200,
    validation_split=0.2,
    early_stopping_patience=15  # Stop if no improvement for 15 epochs
)
```

### 5. Use Appropriate Preprocessing

```python
# Based on data characteristics
clean_data_config = Config(
    normalize=True,
    baseline_correction=False,
    smoothing=False
)

noisy_data_config = Config(
    normalize=True,
    baseline_correction=True,
    smoothing=True,
    smoothing_window=7
)
```

## Configuration Validation

The configuration system includes validation to catch common errors:

```python
# This will raise an error
try:
    invalid_config = Config(
        input_size=1000,
        hidden_sizes=[256, 128, 64],
        learning_rate=-0.001  # Invalid negative learning rate
    )
except ValueError as e:
    print(f"Configuration error: {e}")

# This will raise an error
try:
    invalid_config = Config(
        input_size=1000,
        hidden_sizes=[256, 128, 64],
        dropout_rate=1.5  # Invalid dropout rate > 1.0
    )
except ValueError as e:
    print(f"Configuration error: {e}")
```

## Configuration Inheritance

You can create configuration templates and extend them:

```python
# Base configuration template
base_config = Config(
    input_size=1000,
    learning_rate=0.001,
    batch_size=32,
    epochs=100,
    validation_split=0.2,
    normalize=True
)

# Extend for specific model types
mlp_config = Config(
    **base_config.__dict__,
    hidden_sizes=[256, 128, 64],
    dropout_rate=0.2
)

cnn_config = Config(
    **base_config.__dict__,
    conv_layers=[32, 64],
    fc_layers=[128, 64],
    kernel_size=3,
    stride=1
)
```

## Configuration Persistence

Save and load configurations:

```python
import json

# Save configuration
config = Config(
    input_size=1000,
    hidden_sizes=[256, 128, 64],
    learning_rate=0.001,
    batch_size=32,
    epochs=100
)

# Save to file
with open('config.json', 'w') as f:
    json.dump(config.__dict__, f, indent=2)

# Load from file
with open('config.json', 'r') as f:
    config_dict = json.load(f)
    loaded_config = Config(**config_dict)
```

## Environment-Specific Configurations

Create different configurations for different environments:

```python
# Development configuration
dev_config = Config(
    input_size=1000,
    hidden_sizes=[64, 32],
    learning_rate=0.001,
    batch_size=16,
    epochs=50,
    log_level='DEBUG',
    save_logs=True
)

# Production configuration
prod_config = Config(
    input_size=1000,
    hidden_sizes=[512, 256, 128, 64],
    learning_rate=0.0005,
    batch_size=32,
    epochs=200,
    log_level='INFO',
    save_logs=True,
    save_model=True
)

# Testing configuration
test_config = Config(
    input_size=1000,
    hidden_sizes=[32, 16],
    learning_rate=0.01,
    batch_size=8,
    epochs=10,
    log_level='WARNING'
)
```

## Advanced Configuration Patterns

### 1. Configuration Factory

```python
def create_mlp_config(input_size, complexity='medium'):
    """Create MLP configuration based on complexity level."""
    
    configs = {
        'simple': {
            'hidden_sizes': [64, 32],
            'learning_rate': 0.01,
            'epochs': 50
        },
        'medium': {
            'hidden_sizes': [256, 128, 64],
            'learning_rate': 0.001,
            'epochs': 100
        },
        'complex': {
            'hidden_sizes': [512, 256, 128, 64, 32],
            'learning_rate': 0.0005,
            'epochs': 200
        }
    }
    
    base_config = {
        'input_size': input_size,
        'batch_size': 32,
        'validation_split': 0.2,
        'early_stopping_patience': 10,
        'dropout_rate': 0.2
    }
    
    return Config(**base_config, **configs[complexity])

# Usage
simple_config = create_mlp_config(1000, 'simple')
complex_config = create_mlp_config(1000, 'complex')
```

### 2. Configuration Validation

```python
def validate_config(config):
    """Validate configuration parameters."""
    errors = []
    
    if config.learning_rate <= 0:
        errors.append("Learning rate must be positive")
    
    if config.batch_size <= 0:
        errors.append("Batch size must be positive")
    
    if config.epochs <= 0:
        errors.append("Epochs must be positive")
    
    if config.dropout_rate < 0 or config.dropout_rate > 1:
        errors.append("Dropout rate must be between 0 and 1")
    
    if errors:
        raise ValueError(f"Configuration errors: {', '.join(errors)}")
    
    return True

# Usage
config = Config(learning_rate=0.001, batch_size=32, epochs=100, dropout_rate=0.2)
validate_config(config)
```

## Troubleshooting Configuration Issues

### Common Issues and Solutions

1. **Memory Issues**
   ```python
   # Reduce batch size and model size
   config = Config(
       batch_size=8,  # Smaller batch size
       hidden_sizes=[64, 32],  # Smaller model
       epochs=50  # Fewer epochs
   )
   ```

2. **Training Too Slow**
   ```python
   # Increase learning rate and reduce model complexity
   config = Config(
       learning_rate=0.01,  # Higher learning rate
       hidden_sizes=[64, 32],  # Simpler model
       batch_size=64  # Larger batch size
   )
   ```

3. **Overfitting**
   ```python
   # Add regularization and early stopping
   config = Config(
       dropout_rate=0.3,  # More dropout
       early_stopping_patience=5,  # Earlier stopping
       validation_split=0.3  # More validation data
   )
   ```

4. **Underfitting**
   ```python
   # Increase model capacity and training time
   config = Config(
       hidden_sizes=[512, 256, 128, 64],  # Larger model
       epochs=300,  # More epochs
       learning_rate=0.0005  # Lower learning rate
   )
   ```

---

This configuration guide covers the essential aspects of using the configuration system in the UV-Vis Research Library. For more advanced usage patterns, see the [Examples Gallery](examples.md) and [Best Practices](advanced/best_practices.md) sections. 