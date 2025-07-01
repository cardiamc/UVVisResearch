# Installation Guide

This guide will help you install the UV-Vis Research Library (uvvislib) on your system.

## System Requirements

### Operating Systems
- **Linux**: Ubuntu 18.04+, CentOS 7+, or similar
- **macOS**: 10.14+ (Mojave or later)
- **Windows**: Windows 10 or later

### Python Requirements
- **Python**: 3.8 or higher
- **pip**: Latest version recommended

### Hardware Requirements
- **RAM**: Minimum 4GB, 8GB+ recommended for large datasets
- **Storage**: At least 1GB free space
- **GPU**: Optional but recommended for deep learning models (CUDA-compatible)

## Installation Methods

### Method 1: Install from Source (Recommended)

This is the recommended method for development and customization:

```bash
# Clone the repository
git clone https://github.com/your-username/UVVisResearch.git
cd UVVisResearch

# Create a virtual environment (recommended)
python -m venv uvvis_env
source uvvis_env/bin/activate  # On Windows: uvvis_env\Scripts\activate

# Install in development mode
pip install -e .

# Install dependencies
pip install -r requirements.txt
```

### Method 2: Install via pip

For production use, you can install directly via pip:

```bash
# Create virtual environment
python -m venv uvvis_env
source uvvis_env/bin/activate

# Install the library
pip install uvvislib
```

### Method 3: Conda Installation

If you prefer using Conda:

```bash
# Create conda environment
conda create -n uvvis_env python=3.9
conda activate uvvis_env

# Install dependencies
conda install numpy pandas scikit-learn matplotlib seaborn
conda install pytorch torchvision -c pytorch

# Install the library
pip install uvvislib
```

## Dependencies

The library has the following key dependencies:

### Core Dependencies
- **numpy** (>=1.21.0) - Numerical computing
- **pandas** (>=1.3.0) - Data manipulation and analysis
- **scikit-learn** (>=1.0.0) - Machine learning utilities
- **matplotlib** (>=3.5.0) - Plotting and visualization
- **seaborn** (>=0.11.0) - Statistical visualization

### Deep Learning Dependencies
- **torch** (>=1.9.0) - PyTorch deep learning framework
- **torchvision** (>=0.10.0) - Computer vision utilities

### Optional Dependencies
- **jupyter** - For Jupyter notebook support
- **pytest** - For running tests
- **pytest-cov** - For test coverage

## Verification

After installation, verify that everything is working correctly:

### 1. Basic Import Test

```python
# Test basic imports
import uvvislib
from uvvislib.data.loader import load_data
from uvvislib.models.mlp import MLPRegressor
from uvvislib.utils.config import Config

print("✅ UV-Vis library imported successfully!")
```

### 2. Configuration Test

```python
# Test configuration system
config = Config(
    input_size=100,
    hidden_sizes=[64, 32],
    learning_rate=0.001
)
print("✅ Configuration system working!")
```

### 3. Model Creation Test

```python
# Test model creation
model = MLPRegressor(config)
print("✅ Model creation successful!")
```

### 4. Run Tests

```bash
# Run the test suite
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=uvvislib --cov-report=html
```

## Troubleshooting

### Common Issues

#### 1. Import Errors
**Problem**: `ModuleNotFoundError: No module named 'uvvislib'`

**Solution**:
```bash
# Make sure you're in the correct directory
cd UVVisResearch

# Reinstall in development mode
pip install -e .
```

#### 2. PyTorch Installation Issues
**Problem**: PyTorch installation fails

**Solution**:
```bash
# Install PyTorch separately first
pip install torch torchvision

# Then install the library
pip install -e .
```

#### 3. CUDA Issues
**Problem**: CUDA not available for PyTorch

**Solution**:
```bash
# Install CPU-only version
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Or install with CUDA support (if you have CUDA installed)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

#### 4. Permission Errors
**Problem**: Permission denied during installation

**Solution**:
```bash
# Use user installation
pip install --user -e .

# Or use virtual environment (recommended)
python -m venv uvvis_env
source uvvis_env/bin/activate
pip install -e .
```

### Environment Setup

#### Virtual Environment Best Practices

```bash
# Create virtual environment
python -m venv uvvis_env

# Activate (Linux/macOS)
source uvvis_env/bin/activate

# Activate (Windows)
uvvis_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import uvvislib; print('Installation successful!')"
```

#### Jupyter Integration

If you want to use the library in Jupyter notebooks:

```bash
# Install Jupyter
pip install jupyter

# Install ipykernel for the virtual environment
pip install ipykernel

# Add the virtual environment to Jupyter
python -m ipykernel install --user --name=uvvis_env --display-name="UV-Vis Library"
```

## Next Steps

After successful installation:

1. **Read the [Quick Start Guide](quickstart.md)** to get up and running
2. **Explore the [Examples Gallery](examples.md)** for usage patterns
3. **Check the [Configuration Guide](configuration.md)** for customization
4. **Review [Best Practices](advanced/best_practices.md)** for optimal usage

## Support

If you encounter issues during installation:

1. Check the [Troubleshooting](advanced/troubleshooting.md) section
2. Review the error messages carefully
3. Ensure all system requirements are met
4. Try the verification steps above
5. Open an issue on the project repository with detailed error information

---

**Installation completed successfully!** 🎉 