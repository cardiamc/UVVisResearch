"""
Logging utilities for UV-Vis analysis library.
"""

import logging
import os
from datetime import datetime
from typing import Optional, Dict, Any
from pathlib import Path


def setup_logging(
    output_dir: str = "./LOG/",
    log_level: int = logging.DEBUG,
    log_format: str = "%(asctime)s %(levelname)s %(message)s",
    disable_matplotlib_logging: bool = True
) -> logging.Logger:
    """
    Set up logging configuration for the UV-Vis analysis library.
    
    Args:
        output_dir: Directory to save log files
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Format string for log messages
        disable_matplotlib_logging: Whether to disable matplotlib logging
    
    Returns:
        Configured logger instance
    """
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Create timestamp for log file
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = f"{output_dir}/{timestamp}.log"
    
    # Configure logging
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()  # Also log to console
        ]
    )
    
    # Disable matplotlib logging if requested
    if disable_matplotlib_logging:
        logging.getLogger('matplotlib.font_manager').disabled = True
        logging.getLogger('matplotlib').setLevel(logging.WARNING)
    
    # Get logger instance
    logger = logging.getLogger(__name__)
    
    # Log initial setup information
    logger.info("=" * 50)
    logger.info("UV-Vis Analysis Library Logging Setup")
    logger.info("=" * 50)
    logger.info(f"Log file: {log_filename}")
    logger.info(f"Log level: {logging.getLevelName(log_level)}")
    logger.info(f"Timestamp: {timestamp}")
    
    return logger


class ExperimentLogger:
    """
    Logger class for tracking experiment parameters and results.
    """
    
    def __init__(self, experiment_name: str, output_dir: str = "./LOG/"):
        """
        Initialize experiment logger.
        
        Args:
            experiment_name: Name of the experiment
            output_dir: Directory to save experiment logs
        """
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir)
        self.experiment_dir = self.output_dir / f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging for this experiment
        self.logger = setup_logging(
            output_dir=str(self.experiment_dir),
            log_level=logging.DEBUG
        )
        
        # Store experiment parameters
        self.parameters: Dict[str, Any] = {}
        self.results: Dict[str, Any] = {}
    
    def log_parameter(self, name: str, value: Any) -> None:
        """Log a parameter value."""
        self.parameters[name] = value
        self.logger.info(f"Parameter - {name}: {value}")
    
    def log_parameters(self, params: Dict[str, Any]) -> None:
        """Log multiple parameters at once."""
        for name, value in params.items():
            self.log_parameter(name, value)
    
    def log_result(self, name: str, value: Any) -> None:
        """Log a result value."""
        self.results[name] = value
        self.logger.info(f"Result - {name}: {value}")
    
    def log_results(self, results: Dict[str, Any]) -> None:
        """Log multiple results at once."""
        for name, value in results.items():
            self.log_result(name, value)
    
    def log_model_info(self, model_name: str, model_params: Dict[str, Any]) -> None:
        """Log model information."""
        self.logger.info(f"Model: {model_name}")
        self.log_parameters(model_params)
    
    def log_training_start(self, model_name: str) -> None:
        """Log the start of model training."""
        self.logger.info(f"Starting training for {model_name}")
    
    def log_training_end(self, model_name: str, training_time: float) -> None:
        """Log the end of model training."""
        self.logger.info(f"Training completed for {model_name} in {training_time:.2f} seconds")
    
    def log_evaluation_results(self, metrics: Dict[str, float]) -> None:
        """Log evaluation metrics."""
        self.logger.info("Evaluation Results:")
        for metric_name, value in metrics.items():
            self.logger.info(f"  {metric_name}: {value:.4f}")
    
    def log_error(self, error_message: str, exception: Optional[Exception] = None) -> None:
        """Log an error message."""
        self.logger.error(f"ERROR: {error_message}")
        if exception:
            self.logger.error(f"Exception: {str(exception)}")
    
    def log_warning(self, warning_message: str) -> None:
        """Log a warning message."""
        self.logger.warning(f"WARNING: {warning_message}")
    
    def save_experiment_summary(self) -> None:
        """Save experiment summary to file."""
        summary_file = self.experiment_dir / "experiment_summary.txt"
        
        with open(summary_file, 'w') as f:
            f.write("=" * 50 + "\n")
            f.write(f"Experiment: {self.experiment_name}\n")
            f.write(f"Timestamp: {datetime.now()}\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("PARAMETERS:\n")
            f.write("-" * 20 + "\n")
            for name, value in self.parameters.items():
                f.write(f"{name}: {value}\n")
            
            f.write("\nRESULTS:\n")
            f.write("-" * 20 + "\n")
            for name, value in self.results.items():
                f.write(f"{name}: {value}\n")
        
        self.logger.info(f"Experiment summary saved to: {summary_file}")
    
    def get_experiment_path(self) -> Path:
        """Get the experiment directory path."""
        return self.experiment_dir 