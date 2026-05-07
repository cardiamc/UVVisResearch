"""
Visualization module for UV-Vis spectral analysis.

This module provides comprehensive plotting functions for:
- Spectral data visualization
- Model evaluation plots
- Feature importance visualization
- Cross-validation results
- Training history plots
"""

import math
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
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


# ---------------------------------------------------------------------------
# Generative-model plotting helpers
# ---------------------------------------------------------------------------

def _confidence_ellipse(
    x: np.ndarray,
    y: np.ndarray,
    ax,
    n_std: float = 2.0,
    facecolor: str = "none",
    **kwargs,
):
    """Draw a covariance confidence ellipse on *ax*."""
    from matplotlib.patches import Ellipse
    import matplotlib.transforms as transforms

    if x.size != y.size:
        raise ValueError("x and y must be the same size")
    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    ell_rx = np.sqrt(1 + pearson)
    ell_ry = np.sqrt(1 - pearson)
    ellipse = Ellipse(
        (0, 0), width=ell_rx * 2, height=ell_ry * 2, facecolor=facecolor, **kwargs
    )
    scale_x = np.sqrt(cov[0, 0]) * n_std
    scale_y = np.sqrt(cov[1, 1]) * n_std
    transf = (
        transforms.Affine2D()
        .rotate_deg(45)
        .scale(scale_x, scale_y)
        .translate(np.mean(x), np.mean(y))
    )
    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def plot_gan_losses(
    history: Dict[str, list],
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False,
    figsize: Tuple[int, int] = (10, 4),
    dpi: int = 100,
) -> "plt.Figure":
    """
    Plot discriminator and generator loss curves from a CGAN training history.

    Parameters
    ----------
    history : dict
        ``CGAN.training_history`` — must contain ``"D_loss"`` and ``"G_loss"`` keys.
    save_path : str or Path, optional
    show : bool
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    epochs = range(1, len(history["D_loss"]) + 1)
    ax.plot(epochs, history["D_loss"], label="D_loss", color="steelblue")
    ax.plot(epochs, history["G_loss"], label="G_loss", color="tomato")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("GAN Training Losses")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def plot_generative_samples(
    X_real: np.ndarray,
    X_synth: np.ndarray,
    wavelengths: Optional[np.ndarray] = None,
    title: str = "Real vs Synthetic Spectra",
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False,
    figsize: Tuple[int, int] = (12, 5),
    dpi: int = 100,
) -> "plt.Figure":
    """
    Overlay mean ± 1 SD for real and synthetic spectra.

    Parameters
    ----------
    X_real : ndarray of shape (n_real, n_wavelengths)
    X_synth : ndarray of shape (n_synth, n_wavelengths)
    wavelengths : ndarray of shape (n_wavelengths,), optional
        Defaults to 200–727.5 nm at 2.5 nm step.
    """
    if wavelengths is None:
        wavelengths = np.arange(200, 730, 2.5)

    real_mean = np.mean(X_real, axis=0)
    real_std = np.std(X_real, axis=0)
    synth_mean = np.mean(X_synth, axis=0)
    synth_std = np.std(X_synth, axis=0)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.plot(wavelengths, real_mean, "b-", label="Real (mean)", linewidth=2)
    ax.fill_between(
        wavelengths, real_mean - real_std, real_mean + real_std,
        color="blue", alpha=0.15, label="Real (±1 SD)"
    )
    ax.plot(wavelengths, synth_mean, "r-", label="Synthetic (mean)", linewidth=2)
    ax.fill_between(
        wavelengths, synth_mean - synth_std, synth_mean + synth_std,
        color="red", alpha=0.15, label="Synthetic (±1 SD)"
    )
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Absorbance (MinMax-scaled)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def plot_pca_comparison(
    X_real: np.ndarray,
    X_synth: np.ndarray,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False,
    figsize: Tuple[int, int] = (12, 10),
    dpi: int = 100,
) -> "plt.Figure":
    """
    PCA scatter plot (PC1 vs PC2) with confidence ellipses, plus loading curves.

    PCA is fit on the combined (real + synthetic) data standardised together,
    matching the approach in ``reliability_synthetic_data.py``.
    """
    from sklearn.preprocessing import StandardScaler

    combined = np.vstack([X_real, X_synth])
    scaled = StandardScaler().fit_transform(combined)
    pca = PCA(n_components=2).fit(scaled)
    proj = pca.transform(scaled)
    real_proj = proj[: len(X_real)]
    synth_proj = proj[len(X_real) :]

    wavelengths = np.arange(200, 730, 2.5)[: X_real.shape[1]]

    fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=dpi)

    # --- Scatter ---
    ax = axes[0]
    ax.scatter(real_proj[:, 0], real_proj[:, 1], c="blue", alpha=0.6,
               label="Real", s=50, edgecolors="none")
    ax.scatter(synth_proj[:, 0], synth_proj[:, 1], c="red", alpha=0.6,
               label="Synthetic", s=50, edgecolors="none")
    _confidence_ellipse(real_proj[:, 0], real_proj[:, 1], ax,
                        n_std=2.0, edgecolor="blue", linewidth=2)
    _confidence_ellipse(synth_proj[:, 0], synth_proj[:, 1], ax,
                        n_std=2.0, edgecolor="red", linewidth=2)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.set_title("PCA Projection")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Loadings ---
    ax2 = axes[1]
    ax2.plot(wavelengths, pca.components_[0], "b-", label="PC1", linewidth=2)
    ax2.plot(wavelengths, pca.components_[1], "r-", label="PC2", linewidth=2)
    ax2.set_xlabel("Wavelength (nm)")
    ax2.set_ylabel("Loading value")
    ax2.set_title("PCA Loadings (combined)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def plot_wasserstein_profile(
    X_real: np.ndarray,
    X_synth: np.ndarray,
    wavelengths: Optional[np.ndarray] = None,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False,
    figsize: Tuple[int, int] = (12, 4),
    dpi: int = 100,
) -> "plt.Figure":
    """
    Per-wavelength Wasserstein distance between real and synthetic distributions.
    """
    from scipy.stats import wasserstein_distance as _wd

    if wavelengths is None:
        wavelengths = np.arange(200, 730, 2.5)[: X_real.shape[1]]

    wd = np.array([
        _wd(X_real[:, i], X_synth[:, i]) for i in range(X_real.shape[1])
    ])
    mean_wd = float(np.mean(wd))

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.bar(wavelengths, wd, width=2.0, color="purple", alpha=0.7)
    ax.axhline(mean_wd, color="red", linewidth=2, label=f"Mean {mean_wd:.4f}")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Wasserstein Distance")
    ax.set_title("Per-wavelength Wasserstein Distance (Real vs Synthetic)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def plot_cod_qq(
    real_y: np.ndarray,
    synth_y: np.ndarray,
    log_space: bool = True,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False,
    figsize: Tuple[int, int] = (6, 6),
    dpi: int = 100,
) -> "plt.Figure":
    """
    Q-Q plot comparing COD distributions of real and synthetic data.

    Parameters
    ----------
    real_y, synth_y : ndarray
        If ``log_space=True`` (default), values are assumed to be log-COD;
        if False they are treated as raw COD (mg/L).
    """
    from scipy import stats

    q = np.linspace(0.01, 0.99, 100)
    rq = np.quantile(real_y, q)
    sq = np.quantile(synth_y, q)

    ks_stat, p_value = stats.ks_2samp(real_y, synth_y)
    unit = "log(COD)" if log_space else "COD (mg/L)"

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.scatter(rq, sq, s=40, alpha=0.7, color="purple")
    lo, hi = min(rq.min(), sq.min()), max(rq.max(), sq.max())
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=2, label="Identity line")
    ax.set_xlabel(f"Real {unit} quantiles")
    ax.set_ylabel(f"Synthetic {unit} quantiles")
    ax.set_title("Q-Q Plot: COD Distribution")
    ax.text(
        0.05, 0.95,
        f"KS stat: {ks_stat:.4f}\np-value: {p_value:.4f}",
        transform=ax.transAxes, va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def plot_comparison_results(
    results_df: "pd.DataFrame",
    metric_groups: Optional[List[str]] = None,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False,
    figsize: Tuple[int, int] = (16, 10),
    dpi: int = 100,
) -> "plt.Figure":
    """
    Bar-chart grid across generators and metrics from ``compare_generators``.

    Parameters
    ----------
    results_df : pd.DataFrame
        Returned by ``uvvislib.generative.evaluation.compare_generators``.
    metric_groups : list of str, optional
        Column prefixes to include (e.g. ``["spectral", "pca"]``).
        Defaults to all numeric columns.
    """
    import pandas as pd

    df = results_df.select_dtypes(include=[np.number])
    if metric_groups is not None:
        cols = [c for c in df.columns if any(c.startswith(g) for g in metric_groups)]
        df = df[cols]

    n_metrics = len(df.columns)
    if n_metrics == 0:
        raise ValueError("No numeric columns to plot.")

    ncols = 3
    nrows = math.ceil(n_metrics / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=dpi)
    axes = np.array(axes).ravel()

    for i, col in enumerate(df.columns):
        ax = axes[i]
        methods = df.index.tolist()
        values = df[col].tolist()
        ax.bar(methods, values, color=sns.color_palette("husl", len(methods)))
        ax.set_title(col, fontsize=9)
        ax.tick_params(axis="x", rotation=30)
        ax.grid(True, alpha=0.3, axis="y")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Generator Comparison", fontsize=12)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    return fig


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------
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