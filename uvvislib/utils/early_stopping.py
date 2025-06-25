"""
Early stopping utilities for neural network training.
"""

from typing import Optional, Callable
import numpy as np


class EarlyStopper:
    """
    Early stopping mechanism for neural network training.
    
    Monitors validation loss and stops training when the loss doesn't improve
    for a specified number of epochs (patience).
    """
    
    def __init__(
        self, 
        patience: int = 100, 
        min_delta: float = 0.0,
        mode: str = 'min',
        restore_best_weights: bool = True
    ):
        """
        Initialize early stopper.
        
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change in monitored quantity to qualify as improvement
            mode: One of 'min' or 'max'. In 'min' mode, training stops when the quantity
                  monitored has stopped decreasing; in 'max' mode it stops when the
                  quantity monitored has stopped increasing.
            restore_best_weights: Whether to restore model weights from the best epoch
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
        
        # For weight restoration
        self.best_weights = None
        self.model = None
    
    def __call__(self, val_loss: float, model=None) -> bool:
        """
        Check if training should stop.
        
        Args:
            val_loss: Current validation loss
            model: Model instance for weight restoration
            
        Returns:
            True if training should stop, False otherwise
        """
        if self.mode == 'min':
            score = -val_loss
        else:
            score = val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = 0
            if self.restore_best_weights and model is not None:
                self.best_weights = self._get_model_weights(model)
        elif score > self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
            self.best_epoch = 0
            if self.restore_best_weights and model is not None:
                self.best_weights = self._get_model_weights(model)
        
        return self.early_stop
    
    def _get_model_weights(self, model):
        """Get model weights for restoration."""
        if hasattr(model, 'state_dict'):
            return model.state_dict().copy()
        return None
    
    def restore_best_weights(self, model):
        """Restore model to best weights."""
        if self.best_weights is not None and model is not None:
            if hasattr(model, 'load_state_dict'):
                model.load_state_dict(self.best_weights)
                return True
        return False
    
    def reset(self):
        """Reset early stopper state."""
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
        self.best_weights = None
    
    def get_best_epoch(self) -> int:
        """Get the epoch with the best performance."""
        return self.best_epoch
    
    def get_best_score(self) -> Optional[float]:
        """Get the best score achieved."""
        return self.best_score


class LearningRateScheduler:
    """
    Learning rate scheduler with early stopping capabilities.
    """
    
    def __init__(
        self,
        optimizer,
        mode: str = 'min',
        factor: float = 0.1,
        patience: int = 10,
        min_lr: float = 1e-6,
        verbose: bool = False
    ):
        """
        Initialize learning rate scheduler.
        
        Args:
            optimizer: PyTorch optimizer
            mode: One of 'min' or 'max'
            factor: Factor by which to reduce learning rate
            patience: Number of epochs with no improvement after which LR will be reduced
            min_lr: Lower bound on the learning rate
            verbose: If True, prints a message to stdout for each update
        """
        self.optimizer = optimizer
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.verbose = verbose
        
        self.best = None
        self.num_bad_epochs = 0
        self.cooldown_counter = 0
        self.cooldown_reset = 0
    
    def __call__(self, val_loss: float) -> None:
        """
        Step the scheduler based on validation loss.
        
        Args:
            val_loss: Current validation loss
        """
        if self.mode == 'min':
            if self.best is None or val_loss < self.best:
                self.best = val_loss
                self.num_bad_epochs = 0
            else:
                self.num_bad_epochs += 1
        else:
            if self.best is None or val_loss > self.best:
                self.best = val_loss
                self.num_bad_epochs = 0
            else:
                self.num_bad_epochs += 1
        
        if self.num_bad_epochs >= self.patience:
            self._reduce_lr()
            self.num_bad_epochs = 0
    
    def _reduce_lr(self):
        """Reduce learning rate for all parameter groups."""
        for param_group in self.optimizer.param_groups:
            old_lr = param_group['lr']
            new_lr = max(old_lr * self.factor, self.min_lr)
            param_group['lr'] = new_lr
            
            if self.verbose and old_lr != new_lr:
                print(f'Reducing learning rate from {old_lr:.6f} to {new_lr:.6f}')
    
    def get_last_lr(self):
        """Get the current learning rate."""
        return [group['lr'] for group in self.optimizer.param_groups]
    
    def state_dict(self):
        """Get scheduler state."""
        return {
            'best': self.best,
            'num_bad_epochs': self.num_bad_epochs,
            'cooldown_counter': self.cooldown_counter,
            'cooldown_reset': self.cooldown_reset
        }
    
    def load_state_dict(self, state_dict):
        """Load scheduler state."""
        self.best = state_dict['best']
        self.num_bad_epochs = state_dict['num_bad_epochs']
        self.cooldown_counter = state_dict['cooldown_counter']
        self.cooldown_reset = state_dict['cooldown_reset'] 