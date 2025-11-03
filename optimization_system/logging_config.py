"""
Logging Configuration with Rotation

Prevents disk space issues during long optimization runs.
"""

import logging
import logging.handlers
import os
from typing import Optional
from datetime import datetime


def setup_logging(log_dir: str = "/tmp/optimization_logs",
                 log_level: int = logging.INFO,
                 max_bytes: int = 10 * 1024 * 1024,  # 10 MB
                 backup_count: int = 5,
                 console_output: bool = True) -> logging.Logger:
    """
    Setup logging with rotation

    Args:
        log_dir: Directory to store log files
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        max_bytes: Maximum size of each log file before rotation
        backup_count: Number of backup files to keep
        console_output: Whether to also output to console

    Returns:
        Configured logger
    """
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)

    # Create logger
    logger = logging.getLogger('optimization')
    logger.setLevel(log_level)

    # Clear any existing handlers
    logger.handlers = []

    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )

    # 1. Rotating file handler for general logs
    general_log = os.path.join(log_dir, 'optimization.log')
    file_handler = logging.handlers.RotatingFileHandler(
        general_log,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)  # Capture all levels in file
    file_handler.setFormatter(detailed_formatter)
    logger.addHandler(file_handler)

    # 2. Separate rotating handler for errors only
    error_log = os.path.join(log_dir, 'errors.log')
    error_handler = logging.handlers.RotatingFileHandler(
        error_log,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding='utf-8'
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(detailed_formatter)
    logger.addHandler(error_handler)

    # 3. Timed rotating handler for daily logs
    daily_log = os.path.join(log_dir, 'daily.log')
    daily_handler = logging.handlers.TimedRotatingFileHandler(
        daily_log,
        when='midnight',
        interval=1,
        backupCount=7,  # Keep 7 days
        encoding='utf-8'
    )
    daily_handler.setLevel(logging.INFO)
    daily_handler.setFormatter(detailed_formatter)
    logger.addHandler(daily_handler)

    # 4. Console handler (optional)
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(simple_formatter)
        logger.addHandler(console_handler)

    logger.info("=" * 60)
    logger.info("Logging configured successfully")
    logger.info(f"Log directory: {log_dir}")
    logger.info(f"Log level: {logging.getLevelName(log_level)}")
    logger.info(f"Max file size: {max_bytes / (1024 * 1024):.1f} MB")
    logger.info(f"Backup count: {backup_count}")
    logger.info("=" * 60)

    return logger


def setup_performance_logging(log_dir: str = "/tmp/optimization_logs",
                              max_bytes: int = 50 * 1024 * 1024,  # 50 MB
                              backup_count: int = 3) -> logging.Logger:
    """
    Setup separate logger for performance metrics

    Args:
        log_dir: Directory to store log files
        max_bytes: Maximum size before rotation
        backup_count: Number of backups

    Returns:
        Performance logger
    """
    os.makedirs(log_dir, exist_ok=True)

    perf_logger = logging.getLogger('optimization.performance')
    perf_logger.setLevel(logging.INFO)
    perf_logger.handlers = []

    # CSV-style formatter for easy parsing
    csv_formatter = logging.Formatter(
        '%(asctime)s,%(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Performance metrics log
    perf_log = os.path.join(log_dir, 'performance.csv')
    perf_handler = logging.handlers.RotatingFileHandler(
        perf_log,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding='utf-8'
    )
    perf_handler.setFormatter(csv_formatter)
    perf_logger.addHandler(perf_handler)

    # Write CSV header
    perf_logger.info("timestamp,generation,trial,fitness,duration,parameters")

    return perf_logger


def cleanup_old_logs(log_dir: str = "/tmp/optimization_logs",
                    days_to_keep: int = 7):
    """
    Clean up old log files

    Args:
        log_dir: Log directory
        days_to_keep: Number of days of logs to retain
    """
    import time

    if not os.path.exists(log_dir):
        return

    current_time = time.time()
    cutoff_time = current_time - (days_to_keep * 24 * 60 * 60)

    removed_count = 0
    total_size = 0

    for filename in os.listdir(log_dir):
        filepath = os.path.join(log_dir, filename)

        if os.path.isfile(filepath):
            file_time = os.path.getmtime(filepath)

            if file_time < cutoff_time:
                file_size = os.path.getsize(filepath)
                os.remove(filepath)
                removed_count += 1
                total_size += file_size

    if removed_count > 0:
        logger = logging.getLogger('optimization')
        logger.info(f"Cleaned up {removed_count} old log files "
                   f"({total_size / (1024 * 1024):.1f} MB freed)")


def log_trial(perf_logger: logging.Logger,
             generation: int,
             trial: int,
             fitness: float,
             duration: float,
             parameters: dict):
    """
    Log a single trial to performance log

    Args:
        perf_logger: Performance logger instance
        generation: Generation number
        trial: Trial number within generation
        fitness: Fitness value
        duration: Trial duration in seconds
        parameters: Parameter dictionary
    """
    # Serialize parameters as JSON string
    import json
    param_str = json.dumps(parameters)

    perf_logger.info(f"{generation},{trial},{fitness:.4f},{duration:.2f},{param_str}")


class OptimizationLogger:
    """
    Convenience wrapper for optimization logging

    Handles both general logging and performance metrics.
    """

    def __init__(self, log_dir: str = "/tmp/optimization_logs",
                log_level: int = logging.INFO):
        """
        Initialize optimization logger

        Args:
            log_dir: Log directory
            log_level: Logging level
        """
        self.log_dir = log_dir
        self.logger = setup_logging(log_dir, log_level)
        self.perf_logger = setup_performance_logging(log_dir)

        # Clean up old logs on startup
        cleanup_old_logs(log_dir)

    def info(self, message: str):
        """Log info message"""
        self.logger.info(message)

    def warning(self, message: str):
        """Log warning message"""
        self.logger.warning(message)

    def error(self, message: str):
        """Log error message"""
        self.logger.error(message)

    def debug(self, message: str):
        """Log debug message"""
        self.logger.debug(message)

    def log_trial(self, generation: int, trial: int, fitness: float,
                 duration: float, parameters: dict):
        """Log trial performance"""
        log_trial(self.perf_logger, generation, trial, fitness, duration, parameters)

    def log_generation_summary(self, generation: int, best_fitness: float,
                              avg_fitness: float, worst_fitness: float,
                              duration: float):
        """Log generation summary"""
        self.logger.info(
            f"Generation {generation} complete: "
            f"Best={best_fitness:.2f}, Avg={avg_fitness:.2f}, "
            f"Worst={worst_fitness:.2f}, Time={duration:.1f}s"
        )

    def get_log_path(self, log_type: str = 'general') -> str:
        """
        Get path to log file

        Args:
            log_type: 'general', 'errors', 'daily', or 'performance'

        Returns:
            Path to log file
        """
        log_files = {
            'general': 'optimization.log',
            'errors': 'errors.log',
            'daily': 'daily.log',
            'performance': 'performance.csv'
        }

        filename = log_files.get(log_type, 'optimization.log')
        return os.path.join(self.log_dir, filename)


# Global logger instance
_global_logger: Optional[OptimizationLogger] = None


def get_logger() -> OptimizationLogger:
    """
    Get global logger instance

    Returns:
        Global OptimizationLogger
    """
    global _global_logger

    if _global_logger is None:
        _global_logger = OptimizationLogger()

    return _global_logger
