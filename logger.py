"""
Unified logging system for adversarial patch training.

This module provides a consistent logging interface across all components,
replacing scattered print statements with proper structured logging.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime
import colorlog


def setup_logger(
    name: str,
    level: str = "INFO",
    log_dir: Optional[Path] = None,
    console: bool = True,
    file: bool = False,
) -> logging.Logger:
    """
    Configure and return a logger with consistent formatting.

    Args:
        name: Logger name (usually __name__)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files (if file=True)
        console: Enable console output
        file: Enable file output

    Returns:
        Configured logger instance

    Examples:
        >>> # Basic console logging
        >>> logger = setup_logger(__name__)
        >>> logger.info("Training started")

        >>> # With file output
        >>> logger = setup_logger(__name__, file=True, log_dir=Path("logs"))

        >>> # Debug level
        >>> logger = setup_logger(__name__, level="DEBUG")
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    # Clear existing handlers to avoid duplicate logs
    logger.handlers.clear()

    # Prevent propagation to root logger
    logger.propagate = False

    # Define colored formatter for console output
    color_formatter = colorlog.ColoredFormatter(
        fmt='%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s%(reset)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        },
        secondary_log_colors={},
        style='%'
    )

    # Define plain formatter for file output
    plain_formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler - output to stdout with colors
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(color_formatter)
        logger.addHandler(console_handler)

    # File handler - write to file with timestamp (no colors in file)
    if file and log_dir:
        log_dir_path = Path(log_dir)
        log_dir_path.mkdir(parents=True, exist_ok=True)

        # Create log file with timestamp and logger name
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        # Sanitize logger name for filename (replace dots with underscores)
        safe_name = name.replace('.', '_')
        log_file = log_dir_path / f"{safe_name}_{timestamp}.log"

        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(plain_formatter)
        logger.addHandler(file_handler)

        logger.debug(f"Log file created: {log_file}")

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get an existing logger by name.

    This is useful for getting loggers that were already configured
    via setup_logger() in other modules.

    Args:
        name: Logger name

    Returns:
        Existing logger instance

    Examples:
        >>> logger = get_logger(__name__)
        >>> logger.info("Using existing logger")
    """
    return logging.getLogger(name)


class LoggerMixin:
    """
    Mixin class to add logging capabilities to any class.

    Usage:
        class MyTrainer(LoggerMixin):
            def __init__(self, config):
                self._setup_logger(config.logging)
                self.logger.info("Trainer initialized")
    """

    def _setup_logger(
        self,
        logging_config,
        name: Optional[str] = None
    ) -> None:
        """
        Setup logger for this class instance.

        Args:
            logging_config: LoggingConfig object with level, console, file, log_dir
            name: Optional custom logger name (defaults to class name)
        """
        if name is None:
            name = self.__class__.__name__

        self.logger = setup_logger(
            name=name,
            level=logging_config.level,
            log_dir=Path(logging_config.log_dir) if logging_config.log_dir else None,
            console=logging_config.console,
            file=logging_config.file,
        )


# Module-level convenience functions
def debug(msg: str, *args, **kwargs) -> None:
    """Log debug message to root logger."""
    logging.debug(msg, *args, **kwargs)


def info(msg: str, *args, **kwargs) -> None:
    """Log info message to root logger."""
    logging.info(msg, *args, **kwargs)


def warning(msg: str, *args, **kwargs) -> None:
    """Log warning message to root logger."""
    logging.warning(msg, *args, **kwargs)


def error(msg: str, *args, **kwargs) -> None:
    """Log error message to root logger."""
    logging.error(msg, *args, **kwargs)


def critical(msg: str, *args, **kwargs) -> None:
    """Log critical message to root logger."""
    logging.critical(msg, *args, **kwargs)
