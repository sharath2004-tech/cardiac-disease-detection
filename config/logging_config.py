"""
Module 4: Logging Module
ACRMF-Net Logging Configuration
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional


class ColoredFormatter(logging.Formatter):
    """Custom colored formatter for console output"""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }
    
    def format(self, record):
        # Add color to level name
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"
        return super().format(record)


def setup_logging(
    log_dir: Optional[Path] = None,
    log_level: str = "INFO",
    experiment_name: str = "acrmf_net",
    console_output: bool = True,
    file_output: bool = True
) -> logging.Logger:
    """
    Setup comprehensive logging for ACRMF-Net
    
    Args:
        log_dir: Directory to save log files
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        experiment_name: Name of the experiment for log filename
        console_output: Enable console logging
        file_output: Enable file logging
        
    Returns:
        Configured logger instance
    """
    
    # Create logger
    logger = logging.getLogger("ACRMF-Net")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        fmt='[%(asctime)s] [%(name)s] [%(levelname)s] [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    colored_formatter = ColoredFormatter(
        fmt='[%(asctime)s] [%(levelname)s] - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_handler.setFormatter(colored_formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if file_output and log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create timestamped log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"{experiment_name}_{timestamp}.log"
        
        file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)  # Save all logs to file
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"📝 Log file created: {log_file}")
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger


class TrainingLogger:
    """
    Custom logger for tracking training progress
    Integrates with Module 23: Training Engine
    """
    
    def __init__(self, logger: logging.Logger, log_interval: int = 10):
        self.logger = logger
        self.log_interval = log_interval
        self.metrics_history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
    def log_epoch_start(self, epoch: int, total_epochs: int):
        """Log epoch start"""
        self.logger.info("=" * 80)
        self.logger.info(f"🚀 Epoch {epoch}/{total_epochs} Started")
        self.logger.info("=" * 80)
        
    def log_epoch_end(self, epoch: int, metrics: dict):
        """Log epoch end with metrics"""
        self.logger.info("-" * 80)
        self.logger.info(f"✓ Epoch {epoch} Completed")
        
        for key, value in metrics.items():
            self.logger.info(f"  {key}: {value:.4f}")
            if key in self.metrics_history:
                self.metrics_history[key].append(value)
        
        self.logger.info("-" * 80)
        
    def log_batch(self, batch_idx: int, total_batches: int, loss: float, metrics: dict = None):
        """Log batch progress"""
        if batch_idx % self.log_interval == 0:
            msg = f"Batch [{batch_idx}/{total_batches}] Loss: {loss:.4f}"
            if metrics:
                for key, value in metrics.items():
                    msg += f" | {key}: {value:.4f}"
            self.logger.info(msg)
            
    def log_validation(self, metrics: dict):
        """Log validation results"""
        self.logger.info("🔍 Validation Results:")
        for key, value in metrics.items():
            self.logger.info(f"  {key}: {value:.4f}")
            
    def log_best_model(self, epoch: int, metric: str, value: float):
        """Log best model checkpoint"""
        self.logger.info(f"🏆 New best model at epoch {epoch}! {metric}: {value:.4f}")
        
    def log_early_stopping(self, epoch: int, patience: int):
        """Log early stopping trigger"""
        self.logger.warning(f"⚠️  Early stopping triggered at epoch {epoch} (patience: {patience})")
        
    def log_training_complete(self, total_time: float, best_metrics: dict):
        """Log training completion"""
        self.logger.info("=" * 80)
        self.logger.info("🎉 Training Complete!")
        self.logger.info(f"⏱️  Total Time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
        self.logger.info("🏆 Best Metrics:")
        for key, value in best_metrics.items():
            self.logger.info(f"  {key}: {value:.4f}")
        self.logger.info("=" * 80)


class StageLogger:
    """
    Logger for tracking implementation stages
    Useful for the 13-stage roadmap
    """
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        
    def log_stage_start(self, stage_num: int, stage_name: str):
        """Log stage start"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info(f"📋 Stage {stage_num}: {stage_name}")
        self.logger.info("=" * 80 + "\n")
        
    def log_module_start(self, module_num: int, module_name: str):
        """Log module start"""
        self.logger.info(f"⚙️  Module {module_num}: {module_name}")
        
    def log_module_complete(self, module_num: int, module_name: str):
        """Log module completion"""
        self.logger.info(f"✓ Module {module_num}: {module_name} - Complete\n")
        
    def log_stage_complete(self, stage_num: int, stage_name: str):
        """Log stage completion"""
        self.logger.info("=" * 80)
        self.logger.info(f"✅ Stage {stage_num}: {stage_name} - Complete")
        self.logger.info("=" * 80 + "\n")


def get_logger(name: str = "ACRMF-Net") -> logging.Logger:
    """
    Get logger instance by name
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


# ============================================================================
# Example Usage
# ============================================================================
if __name__ == "__main__":
    # Setup logging
    log_dir = Path(__file__).parent.parent / "results" / "logs"
    logger = setup_logging(
        log_dir=log_dir,
        log_level="INFO",
        experiment_name="test_logging"
    )
    
    # Test basic logging
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")
    
    # Test training logger
    training_logger = TrainingLogger(logger, log_interval=5)
    training_logger.log_epoch_start(1, 10)
    training_logger.log_batch(5, 100, loss=0.532, metrics={'acc': 0.85})
    training_logger.log_epoch_end(1, {'train_loss': 0.532, 'val_loss': 0.612, 'val_acc': 0.82})
    
    # Test stage logger
    stage_logger = StageLogger(logger)
    stage_logger.log_stage_start(1, "Project Initialization")
    stage_logger.log_module_start(1, "Project Folder Structure")
    stage_logger.log_module_complete(1, "Project Folder Structure")
    stage_logger.log_stage_complete(1, "Project Initialization")
    
    print(f"\n✓ Logging test complete. Check log file in: {log_dir}")
