"""
Visualization module for UV-Vis spectral analysis.

This module provides comprehensive plotting functions for:
- Spectral data visualization
- Model evaluation plots
- Feature importance visualization
- Cross-validation results
- Training history plots
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
from pathlib import Path
import warnings

# Set style
plt.style.use('default')
sns.set_palette("husl")


class Plotter:
    """
    Main plotting class for UV-Vis spectral analysis.
    
    This class provides methods for creating various types of plots
    commonly used in spectral data analysis and machine learning.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (10, 6), dpi: int = 100):
        """
        Initialize Plotter.
        
        Args:
            figsize: Default figure size (width, height)
            dpi: DPI for figures
        """
        self.figsize = figsize
        self.dpi = dpi
        self.colors = sns.color_palette("husl", 10)
        
    def plot_spectra(
        self, 
        wavelengths: np.ndarray, 
        spectra: np.ndarray,
        labels: Optional[List[str]] = None,
        title: str = "UV-Vis Spectra",
        xlabel: str = "Wavelength (nm)",
        ylabel: str = "Absorbance",
        figsize: Optional[Tuple[int, int]] = None,
        save_path: Optional[Union[str, Path]] = None,
        show: bool = True
    ) -> plt.Figure:
        """
        Plot UV-Vis spectra.
        
        Args:
            wavelengths: Wavelength values
            spectra: Spectral data (n_samples, n_wavelengths)
            labels: Sample labels
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            figsize: Figure size
            save_path: Path to save figure
            show: Whether to display the plot
            
        Returns:
            Matplotlib figure
        """
        if figsize is None:
            figsize = self.figsize
            
        fig, ax = plt.subplots(figsize=figsize, dpi=self.dpi)
        
        if spectra.ndim == 1:
            spectra = spectra.reshape(1, -1)
        
        for i, spectrum in enumerate(spectra):
            label = labels[i] if labels and i < len(labels) else f'Sample {i+1}'
            ax.plot(wavelengths, spectrum, label=label, alpha=0.8)
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        if show:
            plt.show()
        
        return fig
    
    def plot_spectra_comparison(
        self,
        wavelengths: np.ndarray,
        spectra_dict: Dict[str, np.ndarray],
        title: str = "Spectra Comparison",
        xlabel: str = "Wavelength (nm)",
        ylabel: str = "Absorbance",
        figsize: Optional[Tuple[int, int]] = None,
        save_path: Optional[Union[str, Path]] = None,
        show: bool = True
    ) -> plt.Figure:
        """
        Plot comparison of different spectra.
        
        Args:
            wavelengths: Wavelength values
            spectra_dict: Dictionary of spectra with labels as keys
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            figsize: Figure size
            save_path: Path to save figure
            show: Whether to display the plot
            
        Returns:
            Matplotlib figure
        """
        if figsize is None:
            figsize = self.figsize
            
        fig, ax = plt.subplots(figsize=figsize, dpi=self.dpi)
        
        for i, (label, spectrum) in enumerate(spectra_dict.items()):
            color = self.colors[i % len(self.colors)]
            ax.plot(wavelengths, spectrum, label=label, color=color, alpha=0.8)
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        if show:
            plt.show()
        
        return fig
    
    def plot_correlation_matrix(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        title: str = "Correlation Matrix",
        figsize: Optional[Tuple[int, int]] = None,
        save_path: Optional[Union[str, Path]] = None,
        show: bool = True,
        annotate: bool = True
    ) -> plt.Figure:
        """
        Plot correlation matrix heatmap.
        
        Args:
            data: Data matrix or DataFrame
            title: Plot title
            figsize: Figure size
            save_path: Path to save figure
            show: Whether to display the plot
            annotate: Whether to show correlation values
            
        Returns:
            Matplotlib figure
        """
        if figsize is None:
            figsize = (12, 10)
            
        # Calculate correlation matrix
        if isinstance(data, pd.DataFrame):
            corr_matrix = data.corr()
        else:
            corr_matrix = pd.DataFrame(data).corr()
        
        fig, ax = plt.subplots(figsize=figsize, dpi=self.dpi)
        
        # Create heatmap
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(
            corr_matrix, 
            mask=mask,
            annot=annotate, 
            cmap='coolwarm', 
            center=0,
            square=True,
            ax=ax,
            fmt='.2f' if annotate else None
        )
        
        ax.set_title(title)
        
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        if show:
            plt.show()
        
        return fig
    
    def plot_prediction_vs_actual(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        target_names: Optional[List[str]] = None,
        title: str = "Prediction vs Actual",
        figsize: Optional[Tuple[int, int]] = None,
        save_path: Optional[Union[str, Path]] = None,
        show: bool = True
    ) -> plt.Figure:
        """
        Plot predicted vs actual values.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            target_names: Names of targets
            title: Plot title
            figsize: Figure size
            save_path: Path to save figure
            show: Whether to display the plot
            
        Returns:
            Matplotlib figure
        """
        if y_true.ndim == 1:
            y_true = y_true.reshape(-1, 1)
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)
        
        n_targets = y_true.shape[1]
        
        if figsize is None:
            figsize = (5 * n_targets, 5)
        
        fig, axes = plt.subplots(1, n_targets, figsize=figsize, dpi=self.dpi)
        if n_targets == 1:
            axes = [axes]
        
        for i in range(n_targets):
            ax = axes[i]
            
            # Plot points
            ax.scatter(y_true[:, i], y_pred[:, i], alpha=0.6, color=self.colors[i])
            
            # Plot diagonal line
            min_val = min(y_true[:, i].min(), y_pred[:, i].min())
            max_val = max(y_true[:, i].max(), y_pred[:, i].max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
            
            # Calculate R²
            from ..evaluation.metrics import r2_score_single
            r2 = r2_score_single(y_pred[:, i], y_true[:, i])
            
            target_name = target_names[i] if target_names and i < len(target_names) else f'Target {i+1}'
            ax.set_xlabel(f'Actual {target_name}')
            ax.set_ylabel(f'Predicted {target_name}')
            ax.set_title(f'{target_name} (R² = {r2:.3f})')
            ax.grid(True, alpha=0.3)
        
        plt.suptitle(title)
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        if show:
            plt.show()
        
        return fig
    
    def plot_feature_importance(
        self,
        feature_names: List[str],
        importance_scores: np.ndarray,
        target_names: Optional[List[str]] = None,
        title: str = "Feature Importance",
        figsize: Optional[Tuple[int, int]] = None,
        save_path: Optional[Union[str, Path]] = None,
        show: bool = True,
        top_n: Optional[int] = None
    ) -> plt.Figure:
        """
        Plot feature importance.
        
        Args:
            feature_names: Names of features
            importance_scores: Importance scores (n_targets, n_features)
            target_names: Names of targets
            title: Plot title
            figsize: Figure size
            save_path: Path to save figure
            show: Whether to display the plot
            top_n: Number of top features to show
            
        Returns:
            Matplotlib figure
        """
        if importance_scores.ndim == 1:
            importance_scores = importance_scores.reshape(1, -1)
        
        n_targets, n_features = importance_scores.shape
        
        if figsize is None:
            figsize = (12, 6 * n_targets)
        
        fig, axes = plt.subplots(n_targets, 1, figsize=figsize, dpi=self.dpi)
        if n_targets == 1:
            axes = [axes]
        
        for i in range(n_targets):
            ax = axes[i]
            
            # Sort features by importance
            sorted_indices = np.argsort(importance_scores[i])[::-1]
            sorted_scores = importance_scores[i][sorted_indices]
            sorted_names = [feature_names[j] for j in sorted_indices]
            
            # Select top N features if specified
            if top_n is not None:
                sorted_scores = sorted_scores[:top_n]
                sorted_names = sorted_names[:top_n]
            
            # Create horizontal bar plot
            y_pos = np.arange(len(sorted_scores))
            ax.barh(y_pos, sorted_scores, color=self.colors[i])
            ax.set_yticks(y_pos)
            ax.set_yticklabels(sorted_names)
            ax.set_xlabel('Importance Score')
            
            target_name = target_names[i] if target_names and i < len(target_names) else f'Target {i+1}'
            ax.set_title(f'{target_name} - Feature Importance')
            ax.grid(True, alpha=0.3, axis='x')
        
        plt.suptitle(title)
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        if show:
            plt.show()
        
        return fig
    
    def plot_training_history(
        self,
        history: Dict[str, List[float]],
        title: str = "Training History",
        figsize: Optional[Tuple[int, int]] = None,
        save_path: Optional[Union[str, Path]] = None,
        show: bool = True
    ) -> plt.Figure:
        """
        Plot training history.
        
        Args:
            history: Dictionary with training metrics
            title: Plot title
            figsize: Figure size
            save_path: Path to save figure
            show: Whether to display the plot
            
        Returns:
            Matplotlib figure
        """
        if figsize is None:
            figsize = self.figsize
        
        fig, axes = plt.subplots(1, len(history), figsize=figsize, dpi=self.dpi)
        if len(history) == 1:
            axes = [axes]
        
        for i, (metric, values) in enumerate(history.items()):
            ax = axes[i]
            epochs = range(1, len(values) + 1)
            ax.plot(epochs, values, 'b-', label=f'{metric}')
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric)
            ax.set_title(f'{metric} vs Epoch')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.suptitle(title)
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        if show:
            plt.show()
        
        return fig
    
    def plot_cv_results(
        self,
        cv_results: Dict[str, Any],
        title: str = "Cross-Validation Results",
        figsize: Optional[Tuple[int, int]] = None,
        save_path: Optional[Union[str, Path]] = None,
        show: bool = True
    ) -> plt.Figure:
        """
        Plot cross-validation results.
        
        Args:
            cv_results: Cross-validation results dictionary
            title: Plot title
            figsize: Figure size
            save_path: Path to save figure
            show: Whether to display the plot
            
        Returns:
            Matplotlib figure
        """
        if figsize is None:
            figsize = (15, 10)
        
        fig, axes = plt.subplots(2, 2, figsize=figsize, dpi=self.dpi)
        axes = axes.flatten()
        
        # Plot 1: Test scores across folds
        if 'test_scores' in cv_results:
            test_scores = cv_results['test_scores']
            if test_scores:
                # Extract RMSE values
                rmse_values = [score['RMSE'] for score in test_scores]
                if isinstance(rmse_values[0], np.ndarray):
                    rmse_values = np.array(rmse_values)
                    for i in range(rmse_values.shape[1]):
                        axes[0].plot(range(1, len(rmse_values) + 1), 
                                   rmse_values[:, i], 
                                   marker='o', 
                                   label=f'Target {i+1}')
                else:
                    axes[0].plot(range(1, len(rmse_values) + 1), rmse_values, marker='o')
                axes[0].set_xlabel('Fold')
                axes[0].set_ylabel('RMSE')
                axes[0].set_title('Test RMSE Across Folds')
                axes[0].legend()
                axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Train vs Validation scores
        if 'train_scores' in cv_results and 'val_scores' in cv_results:
            train_scores = cv_results['train_scores']
            val_scores = cv_results['val_scores']
            if train_scores and val_scores:
                train_rmse = [score['RMSE'] for score in train_scores]
                val_rmse = [score['RMSE'] for score in val_scores]
                axes[1].plot(range(1, len(train_rmse) + 1), train_rmse, 
                           marker='o', label='Train RMSE')
                axes[1].plot(range(1, len(val_rmse) + 1), val_rmse, 
                           marker='s', label='Val RMSE')
                axes[1].set_xlabel('Fold')
                axes[1].set_ylabel('RMSE')
                axes[1].set_title('Train vs Validation RMSE')
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Predictions vs Actual for all folds
        if 'predictions' in cv_results and 'true_values' in cv_results:
            all_pred = np.vstack(cv_results['predictions'])
            all_true = np.vstack(cv_results['true_values'])
            
            if all_pred.ndim == 1:
                all_pred = all_pred.reshape(-1, 1)
            if all_true.ndim == 1:
                all_true = all_true.reshape(-1, 1)
            
            for i in range(all_pred.shape[1]):
                axes[2].scatter(all_true[:, i], all_pred[:, i], 
                              alpha=0.6, label=f'Target {i+1}')
            
            # Add diagonal line
            min_val = min(all_true.min(), all_pred.min())
            max_val = max(all_true.max(), all_pred.max())
            axes[2].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
            
            axes[2].set_xlabel('Actual Values')
            axes[2].set_ylabel('Predicted Values')
            axes[2].set_title('All Folds: Predicted vs Actual')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
        
        # Plot 4: Score distribution
        if 'test_scores' in cv_results:
            test_scores = cv_results['test_scores']
            if test_scores:
                rmse_values = [score['RMSE'] for score in test_scores]
                if isinstance(rmse_values[0], np.ndarray):
                    rmse_values = np.array(rmse_values)
                    for i in range(rmse_values.shape[1]):
                        axes[3].hist(rmse_values[:, i], alpha=0.7, 
                                   label=f'Target {i+1}', bins=10)
                else:
                    axes[3].hist(rmse_values, bins=10, alpha=0.7)
                axes[3].set_xlabel('RMSE')
                axes[3].set_ylabel('Frequency')
                axes[3].set_title('RMSE Distribution')
                axes[3].legend()
                axes[3].grid(True, alpha=0.3)
        
        plt.suptitle(title)
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        if show:
            plt.show()
        
        return fig
    
    def plot_spectral_feature_importance(
        self,
        wavelengths: np.ndarray,
        importance_scores: np.ndarray,
        target_names: Optional[List[str]] = None,
        title: str = "Spectral Feature Importance",
        figsize: Optional[Tuple[int, int]] = None,
        save_path: Optional[Union[str, Path]] = None,
        show: bool = True,
        threshold: Optional[float] = None
    ) -> plt.Figure:
        """
        Plot feature importance for spectral data.
        
        Args:
            wavelengths: Wavelength values
            importance_scores: Importance scores (n_targets, n_wavelengths)
            target_names: Names of targets
            title: Plot title
            figsize: Figure size
            save_path: Path to save figure
            show: Whether to display the plot
            threshold: Threshold for highlighting important features
            
        Returns:
            Matplotlib figure
        """
        if importance_scores.ndim == 1:
            importance_scores = importance_scores.reshape(1, -1)
        
        n_targets = importance_scores.shape[0]
        
        if figsize is None:
            figsize = (12, 4 * n_targets)
        
        fig, axes = plt.subplots(n_targets, 1, figsize=figsize, dpi=self.dpi)
        if n_targets == 1:
            axes = [axes]
        
        for i in range(n_targets):
            ax = axes[i]
            
            # Plot importance scores
            ax.plot(wavelengths, importance_scores[i], 
                   color=self.colors[i], linewidth=2)
            
            # Highlight important features if threshold is provided
            if threshold is not None:
                important_mask = importance_scores[i] > threshold
                ax.scatter(wavelengths[important_mask], 
                          importance_scores[i][important_mask],
                          color='red', s=50, alpha=0.7, 
                          label=f'Important (> {threshold})')
                ax.legend()
            
            ax.set_xlabel('Wavelength (nm)')
            ax.set_ylabel('Importance Score')
            
            target_name = target_names[i] if target_names and i < len(target_names) else f'Target {i+1}'
            ax.set_title(f'{target_name} - Spectral Feature Importance')
            ax.grid(True, alpha=0.3)
        
        plt.suptitle(title)
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        if show:
            plt.show()
        
        return fig
    
    def plot_model_comparison(
        self,
        model_names: List[str],
        metrics: Dict[str, List[float]],
        title: str = "Model Comparison",
        figsize: Optional[Tuple[int, int]] = None,
        save_path: Optional[Union[str, Path]] = None,
        show: bool = True
    ) -> plt.Figure:
        """
        Plot model comparison.
        
        Args:
            model_names: Names of models
            metrics: Dictionary with metric names as keys and lists of values
            title: Plot title
            figsize: Figure size
            save_path: Path to save figure
            show: Whether to display the plot
            
        Returns:
            Matplotlib figure
        """
        if figsize is None:
            figsize = (12, 6)
        
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=figsize, dpi=self.dpi)
        if n_metrics == 1:
            axes = [axes]
        
        for i, (metric_name, values) in enumerate(metrics.items()):
            ax = axes[i]
            
            # Create bar plot
            bars = ax.bar(model_names, values, color=self.colors[:len(model_names)])
            ax.set_xlabel('Models')
            ax.set_ylabel(metric_name)
            ax.set_title(f'{metric_name} Comparison')
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.3f}', ha='center', va='bottom')
            
            ax.grid(True, alpha=0.3, axis='y')
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        plt.suptitle(title)
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        if show:
            plt.show()
        
        return fig
    
    def plot_residuals(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        target_names: Optional[List[str]] = None,
        title: str = "Residual Analysis",
        figsize: Optional[Tuple[int, int]] = None,
        save_path: Optional[Union[str, Path]] = None,
        show: bool = True
    ) -> plt.Figure:
        """
        Plot residual analysis.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            target_names: Names of targets
            title: Plot title
            figsize: Figure size
            save_path: Path to save figure
            show: Whether to display the plot
            
        Returns:
            Matplotlib figure
        """
        if y_true.ndim == 1:
            y_true = y_true.reshape(-1, 1)
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)
        
        n_targets = y_true.shape[1]
        residuals = y_true - y_pred
        
        if figsize is None:
            figsize = (15, 5 * n_targets)
        
        fig, axes = plt.subplots(n_targets, 2, figsize=figsize, dpi=self.dpi)
        if n_targets == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(n_targets):
            # Residuals vs Predicted
            axes[i, 0].scatter(y_pred[:, i], residuals[:, i], alpha=0.6)
            axes[i, 0].axhline(y=0, color='r', linestyle='--', alpha=0.8)
            axes[i, 0].set_xlabel('Predicted Values')
            axes[i, 0].set_ylabel('Residuals')
            axes[i, 0].set_title(f'Residuals vs Predicted')
            axes[i, 0].grid(True, alpha=0.3)
            
            # Residuals histogram
            axes[i, 1].hist(residuals[:, i], bins=20, alpha=0.7, edgecolor='black')
            axes[i, 1].set_xlabel('Residuals')
            axes[i, 1].set_ylabel('Frequency')
            axes[i, 1].set_title(f'Residuals Distribution')
            axes[i, 1].grid(True, alpha=0.3)
            
            target_name = target_names[i] if target_names and i < len(target_names) else f'Target {i+1}'
            axes[i, 0].set_title(f'{target_name} - Residuals vs Predicted')
            axes[i, 1].set_title(f'{target_name} - Residuals Distribution')
        
        plt.suptitle(title)
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        if show:
            plt.show()
        
        return fig


# Convenience functions
def plot_spectra(*args, **kwargs):
    """Convenience function for plotting spectra."""
    plotter = Plotter()
    return plotter.plot_spectra(*args, **kwargs)


def plot_prediction_vs_actual(*args, **kwargs):
    """Convenience function for plotting predictions vs actual."""
    plotter = Plotter()
    return plotter.plot_prediction_vs_actual(*args, **kwargs)


def plot_feature_importance(*args, **kwargs):
    """Convenience function for plotting feature importance."""
    plotter = Plotter()
    return plotter.plot_feature_importance(*args, **kwargs)


def plot_training_history(*args, **kwargs):
    """Convenience function for plotting training history."""
    plotter = Plotter()
    return plotter.plot_training_history(*args, **kwargs)


def plot_cv_results(*args, **kwargs):
    """Convenience function for plotting cross-validation results."""
    plotter = Plotter()
    return plotter.plot_cv_results(*args, **kwargs) 