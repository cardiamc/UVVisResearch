"""
Convolutional Neural Network (CNN) model for UV-Vis analysis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Union, List, Tuple
import math
from pathlib import Path

from .base import BaseModel
from ..utils.config import Config
from ..utils.early_stopping import EarlyStopper


def calculate_output_length(input_length: int, kernel_size: int, padding: int = 0, 
                           dilation: int = 1, stride: int = 1) -> int:
    """
    Calculate output length for 1D convolution.
    
    Args:
        input_length: Input sequence length
        kernel_size: Kernel size
        padding: Padding size
        dilation: Dilation rate
        stride: Stride size
        
    Returns:
        Output length
    """
    return math.floor((input_length + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)


class CNN(nn.Module):
    """
    PyTorch CNN implementation for 1D spectral data.
    """
    
    def __init__(
        self, 
        n_inputs: int,
        hidden_size: int,
        kernel_size: int,
        stride_size: int,
        n_output: int,
        dropout_rate: float = 0.0
    ):
        """
        Initialize CNN.
        
        Args:
            n_inputs: Number of input features (spectral length)
            hidden_size: Number of hidden channels
            kernel_size: Kernel size for convolutions
            stride_size: Stride size for pooling
            n_output: Number of output units
            dropout_rate: Dropout rate
        """
        super(CNN, self).__init__()
        
        self.n_inputs = n_inputs
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.stride_size = stride_size
        
        # Calculate padding to maintain size
        padding = int(kernel_size / 2) + 1
        
        # First convolution layer
        self.conv1 = nn.Conv1d(1, hidden_size, kernel_size=kernel_size, padding=padding)
        self.batch_norm1 = nn.BatchNorm1d(hidden_size)
        self.pool1 = nn.MaxPool1d(kernel_size=1, stride=stride_size)
        
        # Calculate output size after first conv + pool
        conv1_out = calculate_output_length(n_inputs, kernel_size, padding)
        pool1_out = calculate_output_length(conv1_out, 1, stride=stride_size)
        
        # Second convolution layer
        self.conv2 = nn.Conv1d(hidden_size, 1, kernel_size=kernel_size)
        self.batch_norm2 = nn.BatchNorm1d(1)
        self.pool2 = nn.MaxPool1d(kernel_size=1, stride=stride_size)
        
        # Calculate output size after second conv + pool
        conv2_out = calculate_output_length(pool1_out, kernel_size)
        pool2_out = calculate_output_length(conv2_out, 1, stride=stride_size)
        
        # Fully connected layers
        self.fc1 = nn.Linear(pool2_out, 84)
        self.fc2 = nn.Linear(84, 10)
        self.fc3 = nn.Linear(10, n_output)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        
        self.logger = None  # Will be set by parent class
    
    def forward(self, x):
        """Forward pass."""
        # Reshape for 1D convolution: (batch, channels, length)
        x = x.view(x.size(0), 1, -1)
        
        # First conv block
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.batch_norm1(x)
        
        # Second conv block
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.batch_norm2(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x


class CNNRegressor(BaseModel):
    """
    Convolutional Neural Network regressor for UV-Vis analysis.
    """
    
    def __init__(
        self, 
        config: Config,
        hidden_size: int = 200,
        kernel_size: int = 3,
        stride_size: int = 2,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0,
        dropout_rate: float = 0.0,
        epochs: int = 1000,
        batch_size: Optional[int] = None,
        early_stopping_patience: int = 50,
        early_stopping_min_delta: float = 0.001,
        device: Optional[str] = None
    ):
        """
        Initialize CNN regressor.
        
        Args:
            config: Configuration object
            hidden_size: Number of hidden channels
            kernel_size: Kernel size for convolutions
            stride_size: Stride size for pooling
            learning_rate: Learning rate
            weight_decay: Weight decay (L2 regularization)
            dropout_rate: Dropout rate
            epochs: Number of training epochs
            batch_size: Batch size (None for full batch)
            early_stopping_patience: Early stopping patience
            early_stopping_min_delta: Early stopping minimum delta
            device: Device to use ('cpu', 'cuda', or None for auto)
        """
        super().__init__(config, model_name="CNN")
        
        # Model parameters
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.stride_size = stride_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.dropout_rate = dropout_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        
        # Device setup
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Model components
        self.model: Optional[CNN] = None
        self.optimizer: Optional[optim.Optimizer] = None
        self.criterion = nn.MSELoss()
        
        # Store parameters
        self.model_params = {
            'hidden_size': hidden_size,
            'kernel_size': kernel_size,
            'stride_size': stride_size,
            'learning_rate': learning_rate,
            'weight_decay': weight_decay,
            'dropout_rate': dropout_rate,
            'epochs': epochs,
            'batch_size': batch_size,
            'early_stopping_patience': early_stopping_patience,
            'early_stopping_min_delta': early_stopping_min_delta,
            'device': str(self.device)
        }
        
        self.logger.info(f"CNN initialized on device: {self.device}")
    
    def _create_model(self) -> None:
        """Create the CNN model."""
        if self.n_features is None or self.n_targets is None:
            raise ValueError("Model dimensions not set. Call fit() first.")
        
        self.model = CNN(
            n_inputs=self.n_features,
            hidden_size=self.hidden_size,
            kernel_size=self.kernel_size,
            stride_size=self.stride_size,
            n_output=self.n_targets,
            dropout_rate=self.dropout_rate
        ).to(self.device)
        
        # Set logger for the model
        self.model.logger = self.logger
        
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        self.logger.info(f"Created CNN: {self.n_features} -> {self.hidden_size} -> {self.n_targets}")
    
    def fit(
        self, 
        X: Union[np.ndarray, pd.DataFrame], 
        y: Union[np.ndarray, pd.DataFrame],
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        **kwargs
    ) -> 'CNNRegressor':
        """
        Fit the CNN model.
        
        Args:
            X: Training features
            y: Training targets
            validation_data: Optional validation data (X_val, y_val)
            **kwargs: Additional parameters
            
        Returns:
            Self for method chaining
        """
        # Validate and preprocess data
        X, y = self.validate_data(X, y)
        
        # Update model info
        self._update_model_info(X, y)
        
        # Create model
        self._create_model()
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        
        # Prepare validation data
        if validation_data is not None:
            X_val, y_val = validation_data
            X_val, y_val = self.validate_data(X_val, y_val)
            X_val_tensor = torch.FloatTensor(X_val).to(self.device)
            y_val_tensor = torch.FloatTensor(y_val).to(self.device)
        else:
            X_val_tensor = None
            y_val_tensor = None
        
        # Training setup
        batch_size = self.batch_size if self.batch_size else len(X)
        early_stopper = EarlyStopper(
            patience=self.early_stopping_patience,
            min_delta=self.early_stopping_min_delta
        )
        
        # Training history
        self.training_history = {
            'train_loss': [],
            'val_loss': []
        }
        
        self.logger.info(f"Starting training for {self.epochs} epochs")
        
        # Training loop
        for epoch in range(self.epochs):
            # Training
            self.model.train()
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(X_tensor)
            train_loss = self.criterion(outputs, y_tensor)
            
            # Backward pass
            train_loss.backward()
            self.optimizer.step()
            
            # Validation
            val_loss = None
            if X_val_tensor is not None:
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(X_val_tensor)
                    val_loss = self.criterion(val_outputs, y_val_tensor).item()
            
            # Record history
            self.training_history['train_loss'].append(train_loss.item())
            if val_loss is not None:
                self.training_history['val_loss'].append(val_loss)
            
            # Early stopping
            if val_loss is not None and early_stopper(val_loss, self.model):
                self.logger.info(f"Early stopping at epoch {epoch}")
                break
            
            # Logging
            if epoch % 100 == 0:
                val_str = f", val_loss: {val_loss:.4f}" if val_loss is not None else ""
                self.logger.info(f"Epoch {epoch}: train_loss: {train_loss.item():.4f}{val_str}")
        
        # Restore best weights if early stopping was used
        if X_val_tensor is not None:
            early_stopper.restore_best_weights(self.model)
        
        self.is_fitted = True
        self.logger.info("Training completed")
        
        return self
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Features to predict on
            
        Returns:
            Predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Validate data
        X, _ = self.validate_data(X)
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        # Make predictions
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_tensor)
        
        # Convert to numpy
        predictions = predictions.cpu().numpy()
        
        # Reshape if single target
        if self.n_targets == 1:
            predictions = predictions.reshape(-1)
        
        return predictions
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """
        Get feature importance based on first convolution layer weights.
        
        Returns:
            Feature importance array
        """
        if not self.is_fitted or self.model is None:
            return None
        
        # Use L1 norm of first conv layer weights as feature importance
        weights = self.model.conv1.weight.data.cpu().numpy()
        # Average across channels and kernel positions
        importance = np.mean(np.abs(weights), axis=(0, 2))
        
        return importance
    
    def _save_model_impl(self, filepath: Path) -> None:
        """Save the PyTorch model."""
        if self.model is not None:
            torch.save(self.model.state_dict(), filepath)
    
    def _load_model_impl(self, filepath: Path) -> None:
        """Load the PyTorch model."""
        if self.model is None:
            self._create_model()
        
        if self.model is not None:
            self.model.load_state_dict(torch.load(filepath, map_location=self.device))
            self.is_fitted = True
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get detailed model summary."""
        summary = super().get_model_summary()
        
        if self.model is not None:
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            summary.update({
                'total_parameters': total_params,
                'trainable_parameters': trainable_params,
                'model_architecture': str(self.model),
                'convolutional_layers': 2,
                'fully_connected_layers': 3
            })
        
        return summary 