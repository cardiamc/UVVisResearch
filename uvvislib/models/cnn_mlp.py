"""
CNN-MLP Hybrid Model for UV-Vis spectral analysis.

This module implements a hybrid model that combines:
- CNN layers for processing spectral data
- MLP layers for processing extracted features
- Fusion layer to combine both representations
"""

import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
from pathlib import Path

from .base import BaseModel
from ..utils.config import Config
from ..utils.early_stopping import EarlyStopper


def calculate_output_length(l_input: int, kernel_size: int, padding: int = 0, 
                           dilation: int = 1, stride: int = 1) -> int:
    """
    Calculate output length for 1D convolution.
    
    Args:
        l_input: Input length
        kernel_size: Kernel size
        padding: Padding size
        dilation: Dilation factor
        stride: Stride size
        
    Returns:
        Output length
    """
    return math.floor((l_input + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)


class CNN_MLP(nn.Module):
    """
    CNN-MLP Hybrid Model for spectral data analysis.
    
    This model combines:
    - CNN layers for processing spectral data (absorption spectra)
    - MLP layers for processing extracted features
    - Fusion layer to combine both representations
    """
    
    def __init__(
        self, 
        n_abs: int = 212,
        n_extr: int = 10,
        hidden_size: int = 200,
        kernel_size: int = 3,
        stride_size: int = 2,
        l1_mlp: int = 100,
        act_fun_mlp: nn.Module = nn.Sigmoid,
        n_output: int = 1,
        dropout_rate: float = 0.0
    ):
        """
        Initialize CNN-MLP hybrid model.
        
        Args:
            n_abs: Number of absorption spectral features
            n_extr: Number of extracted features
            hidden_size: Number of hidden channels for CNN
            kernel_size: Kernel size for convolutions
            stride_size: Stride size for pooling
            l1_mlp: Number of hidden units in MLP
            act_fun_mlp: Activation function for MLP
            n_output: Number of output units
            dropout_rate: Dropout rate
        """
        super().__init__()
        
        self.n_abs = n_abs
        self.n_extr = n_extr
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.stride_size = stride_size
        self.l1_mlp = l1_mlp
        
        # CNN branch for absorption spectra
        padding = int(kernel_size / 2) + 1
        l_shape = n_abs
        
        # First convolution layer
        self.conv1 = nn.Conv1d(1, hidden_size, kernel_size=kernel_size, padding=padding)
        l_shape = calculate_output_length(l_shape, kernel_size, padding)
        
        self.batch_norm1 = nn.BatchNorm1d(hidden_size)
        self.pool1 = nn.MaxPool1d(kernel_size=kernel_size, stride=stride_size)
        l_shape = calculate_output_length(l_shape, kernel_size, stride=stride_size)
        
        # Second convolution layer
        self.conv2 = nn.Conv1d(hidden_size, 1, kernel_size=kernel_size, padding=0)
        l_shape = calculate_output_length(l_shape, kernel_size)
        self.pool2 = nn.MaxPool1d(kernel_size=kernel_size, stride=stride_size)
        l_shape = calculate_output_length(l_shape, kernel_size, stride=stride_size)
        
        # CNN fully connected layer
        self.fc1 = nn.Linear(l_shape, 84)
        
        # MLP branch for extracted features
        self.mlp_linear1 = nn.Linear(n_extr, l1_mlp)
        self.mlp_activation = act_fun_mlp()
        self.mlp_linear2 = nn.Linear(l1_mlp, int(l1_mlp / 2))
        
        # Fusion layers
        fusion_input_size = 84 + int(l1_mlp / 2)
        self.fusion1 = nn.Linear(fusion_input_size, 40)
        self.fusion2 = nn.Linear(40, n_output)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        
        self.logger = None  # Will be set by parent class
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, n_abs + n_extr)
            
        Returns:
            Output tensor of shape (batch_size, n_output)
        """
        # Split input into absorption spectra and extracted features
        x_abs = x[:, :self.n_abs]
        x_extr = x[:, self.n_abs:]
        
        # CNN branch for absorption spectra
        x_abs = x_abs.view(x_abs.size(0), 1, -1)  # (batch, 1, n_abs)
        x_abs = self.pool1(F.relu(self.conv1(x_abs)))
        x_abs = self.batch_norm1(x_abs)
        x_abs = self.pool2(F.relu(self.conv2(x_abs)))
        x_abs = F.relu(self.fc1(x_abs))
        
        # MLP branch for extracted features
        x_extr = x_extr.view(x_extr.size(0), 1, -1)  # (batch, 1, n_extr)
        x_extr = self.mlp_linear1(x_extr)
        x_extr = self.mlp_activation(x_extr)
        x_extr = self.dropout(x_extr)
        x_extr = self.mlp_linear2(x_extr)
        
        # Flatten both branches
        x_abs_flat = x_abs.view(x.size(0), -1)
        x_extr_flat = x_extr.view(x.size(0), -1)
        
        # Fusion
        x_combined = torch.hstack((x_abs_flat, x_extr_flat))
        x = self.mlp_activation(self.fusion1(x_combined))
        x = self.dropout(x)
        x = self.fusion2(x)
        
        return x


class CNNMLPRegressor(BaseModel):
    """
    CNN-MLP Hybrid regressor for UV-Vis analysis.
    
    This model combines convolutional layers for spectral data processing
    with MLP layers for extracted features, providing a powerful hybrid approach.
    """
    
    def __init__(
        self, 
        config: Config,
        n_abs: int = 212,
        n_extr: int = 10,
        hidden_size: int = 200,
        kernel_size: int = 3,
        stride_size: int = 2,
        l1_mlp: int = 100,
        activation: str = 'sigmoid',
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
        Initialize CNN-MLP hybrid regressor.
        
        Args:
            config: Configuration object
            n_abs: Number of absorption spectral features
            n_extr: Number of extracted features
            hidden_size: Number of hidden channels for CNN
            kernel_size: Kernel size for convolutions
            stride_size: Stride size for pooling
            l1_mlp: Number of hidden units in MLP
            activation: Activation function name
            learning_rate: Learning rate
            weight_decay: Weight decay (L2 regularization)
            dropout_rate: Dropout rate
            epochs: Number of training epochs
            batch_size: Batch size (None for full batch)
            early_stopping_patience: Early stopping patience
            early_stopping_min_delta: Early stopping minimum delta
            device: Device to use ('cpu', 'cuda', or None for auto)
        """
        super().__init__(config, model_name="CNN_MLP")
        
        # Model parameters
        self.n_abs = n_abs
        self.n_extr = n_extr
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.stride_size = stride_size
        self.l1_mlp = l1_mlp
        self.activation = activation
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
        self.model: Optional[CNN_MLP] = None
        self.optimizer: Optional[optim.Optimizer] = None
        self.criterion = nn.MSELoss()
        
        # Store parameters
        self.model_params = {
            'n_abs': n_abs,
            'n_extr': n_extr,
            'hidden_size': hidden_size,
            'kernel_size': kernel_size,
            'stride_size': stride_size,
            'l1_mlp': l1_mlp,
            'activation': activation,
            'learning_rate': learning_rate,
            'weight_decay': weight_decay,
            'dropout_rate': dropout_rate,
            'epochs': epochs,
            'batch_size': batch_size,
            'early_stopping_patience': early_stopping_patience,
            'early_stopping_min_delta': early_stopping_min_delta,
            'device': str(self.device)
        }
        
        self.logger.info(f"CNN-MLP initialized on device: {self.device}")
    
    def _get_activation(self, activation_name: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            'sigmoid': nn.Sigmoid,
            'relu': nn.ReLU,
            'tanh': nn.Tanh,
            'leaky_relu': nn.LeakyReLU
        }
        
        if activation_name not in activations:
            raise ValueError(f"Unknown activation: {activation_name}")
        
        return activations[activation_name]
    
    def _create_model(self) -> None:
        """Create the CNN-MLP hybrid model."""
        if self.n_features is None or self.n_targets is None:
            raise ValueError("Model dimensions not set. Call fit() first.")
        
        # Validate input dimensions
        if self.n_features != (self.n_abs + self.n_extr):
            raise ValueError(
                f"Expected {self.n_abs + self.n_extr} features "
                f"(n_abs={self.n_abs} + n_extr={self.n_extr}), "
                f"got {self.n_features}"
            )
        
        activation_fn = self._get_activation(self.activation)
        
        self.model = CNN_MLP(
            n_abs=self.n_abs,
            n_extr=self.n_extr,
            hidden_size=self.hidden_size,
            kernel_size=self.kernel_size,
            stride_size=self.stride_size,
            l1_mlp=self.l1_mlp,
            act_fun_mlp=activation_fn,
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
        
        self.logger.info(
            f"Created CNN-MLP: {self.n_features} ({self.n_abs} + {self.n_extr}) -> "
            f"{self.hidden_size} -> {self.n_targets}"
        )
    
    def fit(
        self, 
        X: Union[np.ndarray, pd.DataFrame], 
        y: Union[np.ndarray, pd.DataFrame],
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        **kwargs
    ) -> 'CNNMLPRegressor':
        """
        Fit the CNN-MLP hybrid model.
        
        Args:
            X: Training features (absorption + extracted features)
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
            # Training phase
            self.model.train()
            self.optimizer.zero_grad()
            
            output = self.model(X_tensor)
            train_loss = self.criterion(output, y_tensor)
            train_loss.backward()
            self.optimizer.step()
            
            # Validation phase
            if X_val_tensor is not None:
                self.model.eval()
                with torch.no_grad():
                    val_output = self.model(X_val_tensor)
                    val_loss = self.criterion(val_output, y_val_tensor)
                
                # Early stopping
                if early_stopper.early_stop(val_loss.item()):
                    self.logger.info(f"Early stopping at epoch {epoch}")
                    break
                
                # Store history
                self.training_history['train_loss'].append(train_loss.item())
                self.training_history['val_loss'].append(val_loss.item())
                
                if epoch % 100 == 0:
                    self.logger.info(
                        f"Epoch {epoch}: Train Loss: {train_loss.item():.6f}, "
                        f"Val Loss: {val_loss.item():.6f}"
                    )
            else:
                # No validation data
                self.training_history['train_loss'].append(train_loss.item())
                
                if epoch % 100 == 0:
                    self.logger.info(f"Epoch {epoch}: Train Loss: {train_loss.item():.6f}")
        
        self.is_fitted = True
        self.logger.info("Training completed")
        
        return self
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Input features
            
        Returns:
            Predictions
        """
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be fitted before making predictions")
        
        # Validate and preprocess data
        X, _ = self.validate_data(X, None)
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        # Make predictions
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_tensor)
        
        return predictions.cpu().numpy()
    
    def get_feature_importance(
        self,
        X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
    ) -> Optional[Dict[str, np.ndarray]]:
        """
        Get feature importance for both spectral and extracted features.

        Args:
            X: Input features used to compute gradient-based importance. If
                None, returns the most recently computed importance (or None).

        Returns:
            Dictionary with feature importance for both branches, or None if
            no importance has been computed yet.
        """
        if X is None:
            return self.feature_importance

        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be fitted before computing feature importance")

        X, _ = self.validate_data(X, None)

        X_tensor = torch.FloatTensor(X).to(self.device)
        X_tensor.requires_grad_(True)

        self.model.eval()
        output = self.model(X_tensor)

        importance_abs = []
        importance_extr = []

        for i in range(self.n_targets):
            self.model.zero_grad()
            output[:, i].backward(torch.ones_like(output[:, i]), retain_graph=True)

            grad_abs = X_tensor.grad[:, :self.n_abs].abs().mean(dim=0).cpu().numpy()
            grad_extr = X_tensor.grad[:, self.n_abs:].abs().mean(dim=0).cpu().numpy()

            importance_abs.append(grad_abs)
            importance_extr.append(grad_extr)

        result = {
            'absorption_features': np.array(importance_abs),
            'extracted_features': np.array(importance_extr),
        }
        self.feature_importance = result
        return result

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
                'cnn_layers': 2,
                'mlp_layers': 2,
                'fusion_layers': 2,
                'absorption_features': self.n_abs,
                'extracted_features': self.n_extr
            })
        
        return summary 