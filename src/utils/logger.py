"""Logging Setup Utility

Thiết lập logging cho toàn bộ application với file và console handlers.
"""

import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from typing import Optional


def setup_logger(
    name: str = "vo",
    level: str = "INFO",
    log_file: Optional[str] = None,
    console: bool = True,
    rotation: bool = True,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 3
) -> logging.Logger:
    """
    Thiết lập logger với file và console handlers.

    Args:
        name: Tên logger
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path tới log file (None = không log vào file)
        console: Log ra console hay không
        rotation: Enable log rotation
        max_bytes: Max size của mỗi log file (bytes)
        backup_count: Số backup files giữ lại

    Returns:
        Configured logger instance

    Example:
        >>> logger = setup_logger("vo", "INFO", "logs/vo.log")
        >>> logger.info("VO pipeline started")
    """
    # Tạo logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    # Xóa handlers cũ nếu có (tránh duplicate)
    if logger.hasHandlers():
        logger.handlers.clear()

    # Format
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper()))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler
    if log_file is not None:
        # Tạo log directory nếu chưa có
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        if rotation:
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding='utf-8'
            )
        else:
            file_handler = logging.FileHandler(log_file, encoding='utf-8')

        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "vo") -> logging.Logger:
    """
    Lấy logger đã được setup.

    Args:
        name: Tên logger

    Returns:
        Logger instance
    """
    return logging.getLogger(name)
