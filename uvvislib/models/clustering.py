"""
Clustering analysis module for UV-Vis spectral data.

This module provides unsupervised learning methods for clustering spectral data,
including K-means, hierarchical clustering, and DBSCAN.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Union, List, Tuple
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import logging
from pathlib import Path

from .base import BaseModel
from ..utils.config import Config


class SpectralClusterer(BaseModel):
    """
    Clustering analysis for spectral data.
    
    This class provides various clustering algorithms specifically designed
    for UV-Vis and NIR spectral data analysis.
    """
    
    def __init__(
        self, 
        config: Config,
        algorithm: str = 'kmeans',
        n_clusters: int = 3,
        random_state: int = 42,
        **kwargs
    ):
        """
        Initialize spectral clusterer.
        
        Args:
            config: Configuration object
            algorithm: Clustering algorithm ('kmeans', 'hierarchical', 'dbscan', 'gmm')
            n_clusters: Number of clusters (for algorithms that require it)
            random_state: Random state for reproducibility
            **kwargs: Additional algorithm-specific parameters
        """
        super().__init__(config, model_name=f"SpectralClusterer_{algorithm}")
        
        self.algorithm = algorithm
        self.n_clusters = n_clusters
        self.random_state = random_state
        
        # Initialize clustering model
        self.model = self._create_model(**kwargs)
        
        # Clustering results
        self.labels: Optional[np.ndarray] = None
        self.cluster_centers: Optional[np.ndarray] = None
        self.cluster_sizes: Optional[Dict[int, int]] = None
        
        # Evaluation metrics
        self.silhouette_score: Optional[float] = None
        self.calinski_harabasz_score: Optional[float] = None
        self.davies_bouldin_score: Optional[float] = None
        
        # Store parameters
        self.model_params = {
            'algorithm': algorithm,
            'n_clusters': n_clusters,
            'random_state': random_state,
            **kwargs
        }
        
        self.logger.info(f"Initialized {algorithm} clusterer with {n_clusters} clusters")
    
    def _create_model(self, **kwargs) -> Any:
        """Create the clustering model based on algorithm."""
        if self.algorithm == 'kmeans':
            return KMeans(
                n_clusters=self.n_clusters,
                random_state=self.random_state,
                **kwargs
            )
        elif self.algorithm == 'hierarchical':
            return AgglomerativeClustering(
                n_clusters=self.n_clusters,
                **kwargs
            )
        elif self.algorithm == 'dbscan':
            return DBSCAN(**kwargs)
        elif self.algorithm == 'gmm':
            return GaussianMixture(
                n_components=self.n_clusters,
                random_state=self.random_state,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown clustering algorithm: {self.algorithm}")
    
    def fit(
        self, 
        X: Union[np.ndarray, pd.DataFrame], 
        y: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        **kwargs
    ) -> 'SpectralClusterer':
        """
        Fit the clustering model.
        
        Args:
            X: Training features (spectral data)
            y: Not used for clustering (kept for compatibility)
            **kwargs: Additional parameters
            
        Returns:
            Self for method chaining
        """
        # Validate and preprocess data
        X, _ = self.validate_data(X, None)
        
        # Update model info
        self._update_model_info(X, None)
        
        self.logger.info(f"Fitting {self.algorithm} clustering model")
        
        # Fit the model
        if self.algorithm == 'gmm':
            self.model.fit(X)
            self.labels = self.model.predict(X)
        else:
            self.labels = self.model.fit_predict(X)
        
        # Store cluster centers if available
        if hasattr(self.model, 'cluster_centers_'):
            self.cluster_centers = self.model.cluster_centers_
        elif hasattr(self.model, 'means_'):
            self.cluster_centers = self.model.means_
        
        # Calculate cluster sizes
        unique_labels, counts = np.unique(self.labels, return_counts=True)
        self.cluster_sizes = dict(zip(unique_labels, counts))
        
        # Calculate evaluation metrics
        self._calculate_evaluation_metrics(X)
        
        self.is_fitted = True
        self.logger.info(f"Clustering completed. Found {len(unique_labels)} clusters")
        
        return self
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Predict cluster labels for new data.
        
        Args:
            X: Features to cluster
            
        Returns:
            Cluster labels
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Validate data
        X, _ = self.validate_data(X, None)
        
        # Make predictions
        if self.algorithm == 'gmm':
            return self.model.predict(X)
        else:
            return self.model.predict(X)
    
    def _calculate_evaluation_metrics(self, X: np.ndarray) -> None:
        """Calculate clustering evaluation metrics."""
        if len(np.unique(self.labels)) < 2:
            self.logger.warning("Cannot calculate metrics with less than 2 clusters")
            return
        
        try:
            self.silhouette_score = silhouette_score(X, self.labels)
            self.calinski_harabasz_score = calinski_harabasz_score(X, self.labels)
            self.davies_bouldin_score = davies_bouldin_score(X, self.labels)
        except Exception as e:
            self.logger.warning(f"Could not calculate some evaluation metrics: {e}")
    
    def get_cluster_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the clusters.
        
        Returns:
            Dictionary containing cluster statistics
        """
        if not self.is_fitted:
            return {}
        
        stats = {
            'n_clusters': len(np.unique(self.labels)),
            'cluster_sizes': self.cluster_sizes,
            'silhouette_score': self.silhouette_score,
            'calinski_harabasz_score': self.calinski_harabasz_score,
            'davies_bouldin_score': self.davies_bouldin_score,
            'algorithm': self.algorithm
        }
        
        return stats
    
    def get_cluster_centers(self) -> Optional[np.ndarray]:
        """
        Get cluster centers.
        
        Returns:
            Cluster centers array or None
        """
        return self.cluster_centers
    
    def get_cluster_labels(self) -> Optional[np.ndarray]:
        """
        Get cluster labels for training data.
        
        Returns:
            Cluster labels array or None
        """
        return self.labels
    
    def _save_model_impl(self, filepath: Path) -> None:
        """Save the clustering model."""
        import joblib
        joblib.dump(self.model, filepath)
    
    def _load_model_impl(self, filepath: Path) -> None:
        """Load the clustering model."""
        import joblib
        if filepath.exists():
            self.model = joblib.load(filepath)
            self.is_fitted = True


class ClusteringAnalyzer:
    """
    Comprehensive clustering analysis for spectral data.
    
    This class provides methods for comparing different clustering algorithms
    and finding optimal parameters.
    """
    
    def __init__(self, config: Config):
        """
        Initialize clustering analyzer.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Analysis results
        self.results: Dict[str, Any] = {}
        self.best_model: Optional[SpectralClusterer] = None
    
    def compare_algorithms(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        algorithms: List[str] = ['kmeans', 'hierarchical', 'gmm'],
        n_clusters_range: List[int] = [2, 3, 4, 5, 6],
        random_state: int = 42
    ) -> Dict[str, Any]:
        """
        Compare different clustering algorithms.
        
        Args:
            X: Spectral data
            algorithms: List of algorithms to compare
            n_clusters_range: Range of cluster numbers to test
            random_state: Random state for reproducibility
            
        Returns:
            Dictionary containing comparison results
        """
        self.logger.info("Comparing clustering algorithms")
        
        results = {}
        
        for algorithm in algorithms:
            algorithm_results = {}
            
            for n_clusters in n_clusters_range:
                try:
                    # Create and fit clusterer
                    clusterer = SpectralClusterer(
                        self.config,
                        algorithm=algorithm,
                        n_clusters=n_clusters,
                        random_state=random_state
                    )
                    clusterer.fit(X)
                    
                    # Get statistics
                    stats = clusterer.get_cluster_statistics()
                    algorithm_results[n_clusters] = stats
                    
                except Exception as e:
                    self.logger.warning(f"Failed to fit {algorithm} with {n_clusters} clusters: {e}")
                    algorithm_results[n_clusters] = None
            
            results[algorithm] = algorithm_results
        
        self.results = results
        return results
    
    def find_optimal_clusters(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        algorithm: str = 'kmeans',
        max_clusters: int = 10,
        random_state: int = 42
    ) -> Dict[str, Any]:
        """
        Find optimal number of clusters using elbow method and silhouette analysis.
        
        Args:
            X: Spectral data
            algorithm: Clustering algorithm to use
            max_clusters: Maximum number of clusters to test
            random_state: Random state for reproducibility
            
        Returns:
            Dictionary containing optimal cluster analysis
        """
        self.logger.info(f"Finding optimal number of clusters for {algorithm}")
        
        n_clusters_range = range(2, max_clusters + 1)
        inertias = []
        silhouette_scores = []
        calinski_scores = []
        
        for n_clusters in n_clusters_range:
            try:
                clusterer = SpectralClusterer(
                    self.config,
                    algorithm=algorithm,
                    n_clusters=n_clusters,
                    random_state=random_state
                )
                clusterer.fit(X)
                
                stats = clusterer.get_cluster_statistics()
                
                # Store metrics
                if algorithm == 'kmeans' and hasattr(clusterer.model, 'inertia_'):
                    inertias.append(clusterer.model.inertia_)
                else:
                    inertias.append(None)
                
                silhouette_scores.append(stats.get('silhouette_score'))
                calinski_scores.append(stats.get('calinski_harabasz_score'))
                
            except Exception as e:
                self.logger.warning(f"Failed to fit with {n_clusters} clusters: {e}")
                inertias.append(None)
                silhouette_scores.append(None)
                calinski_scores.append(None)
        
        # Find optimal number of clusters
        valid_silhouette = [s for s in silhouette_scores if s is not None]
        if valid_silhouette:
            optimal_silhouette = n_clusters_range[np.argmax(valid_silhouette)]
        else:
            optimal_silhouette = None
        
        valid_calinski = [c for c in calinski_scores if c is not None]
        if valid_calinski:
            optimal_calinski = n_clusters_range[np.argmax(valid_calinski)]
        else:
            optimal_calinski = None
        
        analysis = {
            'n_clusters_range': list(n_clusters_range),
            'inertias': inertias,
            'silhouette_scores': silhouette_scores,
            'calinski_scores': calinski_scores,
            'optimal_silhouette': optimal_silhouette,
            'optimal_calinski': optimal_calinski,
            'algorithm': algorithm
        }
        
        return analysis
    
    def get_best_model(self) -> Optional[SpectralClusterer]:
        """
        Get the best performing clustering model.
        
        Returns:
            Best clustering model or None
        """
        return self.best_model 