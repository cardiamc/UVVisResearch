"""
Multi-Layer Perceptron (MLP) model for UV-Vis analysis.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Union, List, Tuple
import joblib
from pathlib import Path

from .base import BaseModel
from ..utils.config import Config
from ..utils.early_stopping import EarlyStopper


class MLP(nn.Module):
    """
    PyTorch MLP implementation.
    """
    
    def __init__(
        self, 
        n_inputs: int, 
        hidden_size: int, 
        n_output: int,
        activation: nn.Module = nn.Sigmoid,
        dropout_rate: float = 0.0
    ):
        """
        Initialize MLP.
        
        Args:
            n_inputs: Number of input features
            hidden_size: Number of hidden units
            n_output: Number of output units
            activation: Activation function
            dropout_rate: Dropout rate
        """
        super(MLP, self).__init__()
        
        self.linear1 = nn.Linear(n_inputs, hidden_size)
        self.activation = activation()
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.linear2 = nn.Linear(hidden_size, n_output)
        
    def forward(self, x):
        """Forward pass."""
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class MLPRegressor(BaseModel):
    """
    Multi-Layer Perceptron regressor for UV-Vis analysis.
    """
    
    def __init__(
        self, 
        config: Config,
        hidden_size: int = 100,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0,
        activation: str = 'sigmoid',
        dropout_rate: float = 0.0,
        epochs: int = 1000,
        batch_size: Optional[int] = None,
        early_stopping_patience: int = 50,
        early_stopping_min_delta: float = 0.001,
        device: Optional[str] = None
    ):
        """
        Initialize MLP regressor.
        
        Args:
            config: Configuration object
            hidden_size: Number of hidden units
            learning_rate: Learning rate
            weight_decay: Weight decay (L2 regularization)
            activation: Activation function ('sigmoid', 'relu', 'tanh')
            dropout_rate: Dropout rate
            epochs: Number of training epochs
            batch_size: Batch size (None for full batch)
            early_stopping_patience: Early stopping patience
            early_stopping_min_delta: Early stopping minimum delta
            device: Device to use ('cpu', 'cuda', or None for auto)
        """
        super().__init__(config, model_name="MLP")
        
        # Model parameters
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.activation = activation
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
        self.model: Optional[MLP] = None
        self.optimizer: Optional[optim.Optimizer] = None
        self.criterion = nn.MSELoss()
        
        # Store parameters
        self.model_params = {
            'hidden_size': hidden_size,
            'learning_rate': learning_rate,
            'weight_decay': weight_decay,
            'activation': activation,
            'dropout_rate': dropout_rate,
            'epochs': epochs,
            'batch_size': batch_size,
            'early_stopping_patience': early_stopping_patience,
            'early_stopping_min_delta': early_stopping_min_delta,
            'device': str(self.device)
        }
        
        self.logger.info(f"MLP initialized on device: {self.device}")
    
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
        """Create the MLP model."""
        if self.n_features is None or self.n_targets is None:
            raise ValueError("Model dimensions not set. Call fit() first.")
        
        activation_fn = self._get_activation(self.activation)
        
        self.model = MLP(
            n_inputs=self.n_features,
            hidden_size=self.hidden_size,
            n_output=self.n_targets,
            activation=activation_fn,
            dropout_rate=self.dropout_rate
        ).to(self.device)
        
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        self.logger.info(f"Created MLP: {self.n_features} -> {self.hidden_size} -> {self.n_targets}")
    
    def fit(
        self, 
        X: Union[np.ndarray, pd.DataFrame], 
        y: Union[np.ndarray, pd.DataFrame],
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        **kwargs
    ) -> 'MLPRegressor':
        """
        Fit the MLP model.
        
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
        Get feature importance based on first layer weights.
        
        Returns:
            Feature importance array
        """
        if not self.is_fitted or self.model is None:
            return None
        
        # Use L1 norm of first layer weights as feature importance
        weights = self.model.linear1.weight.data.cpu().numpy()
        importance = np.mean(np.abs(weights), axis=0)
        
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
                'model_architecture': str(self.model)
            })
        
        return summary 

MLPModel = MLPRegressor 