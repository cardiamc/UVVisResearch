"""
Comprehensive example demonstrating all UV-Vis analysis library functionalities.

This script shows how to use all the major components of the library:
- Data loading and preprocessing
- Model training and evaluation
- Clustering analysis
- Visualization
- Experiment management
- Model persistence
"""

import numpy as np
import pandas as pd
from pathlib import Path
import logging

# Import library components
from uvvislib import (
    Config, setup_logging, DataLoader, Preprocessor,
    MLPRegressor, CNNRegressor, CNNMLPRegressor, RandomForestRegressor,
    SpectralClusterer, ClusteringAnalyzer,
    r2_score, rmse, compute_evaluation,
    DoubleKFoldCV, NestedCrossValidation,
    Plotter,
    ModelManager, ExperimentManager, DataPersistence
)


def main():
    """Main function demonstrating all library functionalities."""
    
    # Setup logging
    setup_logging(log_level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("Starting comprehensive UV-Vis analysis example")
    
    # Initialize configuration
    config = Config(
        data_path="./Data/",
        uv_vis_data_file="Abs_100mm_completo_ARCHA.csv",
        chemical_data_file="Analisi_chimiche_complete_ARCHA.csv",
        wavelength_start=210,
        wavelength_end=622,
        wavelength_step=2,
        target_variables=["CONDUCIBILITA'", "POTENZIALE REDOX", "NITRITI", "NITRATI", "FLUORURI", "SODIO", "CALCIO",
                   "MAGNESIO", "SOLFATI", "CLORURI", "DUREZZA (da calcolo)", "pH",
                   "AMMONIO", "CLORATI", "ARSENICO", "ANTIMONIO",
                   "ALLUMINIO", "CADMIO", "CROMO TOTALE", "FERRO", "MANGANESE", "NICHEL", "PIOMBO",
                   "RAME", "SELENIO", "MERCURIO", "BORO", "BROMODICLOROMETANO", "BROMOFORMIO",
                   "CLOROFORMIO", "cis-1,2-DICLOROETILENE", "DIBROMOCLOROMETANO", "TETRACLOROETILENE",
                #    "trans-1,2-DICLOROETILENE", 
                   "TRICLOROETILENE", "1,1-DICLOROETANO", "1,1-DICLOROETILENE",
                   "1,1,1-TRICLOROETANO", "1,2-DIBROMOETANO", "1,2-DICLOROETANO",
                   "1,2-DICLOROPROPANO", "ESACLOROBUTADIENE",
                   "CLORURO DI VINILE", "METILTERZIARBUTILETERE (MTBE)",
                   "RESIDUO FISSO A 180°C", "1,2,4-TRICLOROBENZENE", "1,2,3-TRICLOROBENZENE",
                   "1,3-DICLOROBENZENE",
                   "1,3,5-TRIMETILBENZENE", "n-PROPILBENZENE",
                   "iso-PROPILBENZENE", "STIRENE", "o-XILENE", "(m+p)-XILENE", "ETILBENZENE", "TOLUENE",
                   "CLORITI", "TOC", "CONTA DI MICRORGANISMI VITALI A 36°C", "CONTA DI MICRORGANISMI VITALI A 22°C"],
        log_target=False,
        apply_pca=False,
        k_fold_splits=5,
        random_state=42
    )
    
    # Initialize managers
    experiment_manager = ExperimentManager(config)
    model_manager = ModelManager(config)
    data_persistence = DataPersistence(config)
    
    # Start experiment
    experiment_id = experiment_manager.start_experiment(
        experiment_name="comprehensive_analysis",
        description="Comprehensive demonstration of all library functionalities",
        tags=["demo", "comprehensive", "uv-vis"]
    )
    
    try:
        # 1. DATA LOADING AND PREPROCESSING
        logger.info("=== Data Loading and Preprocessing ===")
        
        # Load data
        data_loader = DataLoader(config)
        features_df, targets_df = data_loader.load_and_prepare_data()
        
        logger.info(f"Loaded data: {features_df.shape[0]} samples, {features_df.shape[1]} features")
        logger.info(f"Target variables: {targets_df.shape[1]}")
        
        # Preprocess data
        preprocessor = Preprocessor(config)
        processed_features, processed_targets = preprocessor.preprocess_pipeline(
            features_df, targets_df,
            clean_data=True,
            normalize_features=True,
            apply_pca=False,
            engineer_features=True
        )
        
        logger.info(f"Preprocessed data: {processed_features.shape}")
        
        # Save processed dataset
        data_persistence.save_dataset(
            processed_features, processed_targets,
            dataset_name="processed_uv_vis_data",
            metadata={"description": "Preprocessed UV-Vis dataset"}
        )
        
        # 2. MODEL TRAINING AND EVALUATION
        logger.info("=== Model Training and Evaluation ===")
        
        # Prepare data for training
        X = processed_features.values
        y = processed_targets.values
        
        # Initialize models
        models = {
            'MLP': MLPRegressor(config, hidden_size=100, epochs=100),
            'CNN': CNNRegressor(config, hidden_size=200, epochs=100),
            'RandomForest': RandomForestRegressor(config, n_estimators=50)
        }
        
        # Train and evaluate models
        results = {}
        for name, model in models.items():
            logger.info(f"Training {name} model...")
            
            # Simple train-test split for demonstration
            split_idx = int(0.8 * len(X))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            metrics = compute_evaluation(y_test, y_pred)
            results[name] = {
                'model': model,
                'metrics': metrics,
                'predictions': y_pred,
                'true_values': y_test
            }

            logger.info(
                f"{name} - R²: {np.array2string(metrics['r2_score'], precision=4)}, "
                f"RMSE: {np.array2string(metrics['rmse'], precision=4)}"
            )

            # Save per-target evaluation metrics to CSV
            experiment_manager.save_evaluation_csv(
                metrics,
                target_names=targets_df.columns.tolist(),
                filename=f"{name.lower()}_evaluation.csv"
            )

            # Save model
            model_manager.save_model(
                model, f"{name.lower()}_model",
                experiment_name=experiment_id
            )
        
        # 3. CROSS-VALIDATION
        logger.info("=== Cross-Validation ===")
        
        # Perform cross-validation on best model
        best_model_name = max(
            results.keys(),
            key=lambda k: np.mean(results[k]["metrics"]["r2_score"])
        )
        best_model = results[best_model_name]['model']
        
        cv = DoubleKFoldCV(
            outer_splits=5,
            inner_splits=4,
            random_state=config.random_state,
            stratify=False,  # multi-output continuous y is incompatible with StratifiedKFold
        )

        rf_param_grid = {
            'n_estimators' : [5, 10, 15, 20, 23, 25, 30],
            'max_depth' : [2, 5, 6, 7, 8, 12, 15, 19, None],
            'min_samples_leaf' : [1, 3, 5, 10],
            'min_samples_split' : [2, 3, 5, 8, 10, 15, 20, 30],
            'criterion' : ['poisson', 'friedman_mse', 'squared_error', 'absolute_error'],
            'max_features': ['sqrt', 'log2'],
        }
        # DoubleKFoldCV uses RandomizedSearchCV and requires a sklearn-compatible estimator
        rf_estimator = results['RandomForest']['model'].model
        cv_scores = cv.fit(X, y, rf_estimator, rf_param_grid, n_iter=64)

        logger.info("Cross-validation results for RandomForest:")
        test_r2 = np.array([np.mean(s['r2_score']) for s in cv_scores['test_scores']])
        test_rmse = np.array([np.mean(s['rmse']) for s in cv_scores['test_scores']])
        logger.info(f"R²: {np.mean(test_r2):.4f} ± {np.std(test_r2):.4f}")
        logger.info(f"RMSE: {np.mean(test_rmse):.4f} ± {np.std(test_rmse):.4f}")

        # Save per-target CV metrics to CSV
        experiment_manager.save_cv_results_csv(
            cv_scores,
            target_names=targets_df.columns.tolist(),
            filename="cv_results_randomforest.csv"
        )
        
        # 4. CLUSTERING ANALYSIS
        logger.info("=== Clustering Analysis ===")
        
        # Perform clustering on spectral data
        clusterer = SpectralClusterer(
            config, algorithm='kmeans', n_clusters=3
        )
        clusterer.fit(X)
        
        cluster_stats = clusterer.get_cluster_statistics()
        logger.info(f"Clustering results: {cluster_stats}")
        
        # Compare clustering algorithms
        clustering_analyzer = ClusteringAnalyzer(config)
        clustering_comparison = clustering_analyzer.compare_algorithms(
            X, algorithms=['kmeans', 'hierarchical', 'gmm'],
            n_clusters_range=[2, 3, 4, 5]
        )
        
        logger.info("Clustering algorithm comparison completed")
        
        # 5. VISUALIZATION
        logger.info("=== Visualization ===")
        
        plotter = Plotter()
        
        # Plot predictions vs actual for best model
        best_result = results[best_model_name]
        fig = plotter.plot_prediction_vs_actual(
            best_result['true_values'],
            best_result['predictions'],
            target_names=targets_df.columns.tolist(),
            title=f"{best_model_name} - Predictions vs Actual"
        )
        
        experiment_manager.save_plot(fig, f"{best_model_name}_predictions.png")
        
        # Plot feature importance for Random Forest
        if 'RandomForest' in results:
            rf_model = results['RandomForest']['model']
            feature_importance = rf_model.get_feature_importance()
            
            if feature_importance is not None:
                fig = plotter.plot_feature_importance(
                    processed_features.columns.tolist(),
                    feature_importance.reshape(1, -1),
                    target_names=targets_df.columns.tolist(),
                    title="Random Forest Feature Importance",
                    top_n=20
                )
                experiment_manager.save_plot(fig, "random_forest_feature_importance.png")
        
        # Plot clustering results (Plotter has no clustering scatter method; use matplotlib directly)
        cluster_labels = clusterer.get_cluster_labels()
        if cluster_labels is not None:
            import matplotlib.pyplot as plt
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X)
            fig, ax = plt.subplots(figsize=(8, 6))
            scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis', alpha=0.7)
            plt.colorbar(scatter, ax=ax, label='Cluster')
            ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
            ax.set_title('Spectral Data Clustering Results')
            plt.tight_layout()
            experiment_manager.save_plot(fig, "clustering_results.png")
            plt.close(fig)
        
        # 6. SAVE RESULTS
        logger.info("=== Saving Results ===")
        
        # Save all results
        all_results = {
            'model_results': {name: result['metrics'] for name, result in results.items()},
            'cross_validation': cv_scores,
            'clustering': cluster_stats,
            'clustering_comparison': clustering_comparison,
            'data_info': {
                'n_samples': len(X),
                'n_features': X.shape[1],
                'n_targets': y.shape[1],
                'feature_names': processed_features.columns.tolist(),
                'target_names': targets_df.columns.tolist()
            }
        }
        
        experiment_manager.save_results(all_results, "comprehensive_results.json")
        
        # Save predictions
        for name, result in results.items():
            experiment_manager.save_predictions(
                result['predictions'],
                result['true_values'],
                target_names=targets_df.columns.tolist(),
                filename=f"{name.lower()}_predictions.csv"
            )
        
        logger.info("All results saved successfully")
        
        # 7. SUMMARY
        logger.info("=== Analysis Summary ===")
        logger.info(f"Experiment ID: {experiment_id}")
        logger.info(f"Best model: {best_model_name}")
        logger.info(f"Best R² score: {np.mean(results[best_model_name]['metrics']['r2_score']):.4f}")
        logger.info(f"Number of clusters found: {cluster_stats['n_clusters']}")
        logger.info(f"Clustering silhouette score: {cluster_stats['silhouette_score']:.4f}")
        
    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        experiment_manager.end_experiment(status="failed")
        raise
    else:
        # End experiment successfully
        experiment_manager.end_experiment(status="completed")
        logger.info("Comprehensive analysis completed successfully!")


if __name__ == "__main__":
    main() 