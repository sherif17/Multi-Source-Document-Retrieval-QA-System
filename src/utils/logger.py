"""
Centralized logging configuration using Loguru.

Why Loguru over stdlib logging:
- Zero-config structured output
- Automatic exception formatting with traceback
- Rotation and retention built-in
- Cleaner API (no getLogger/handlers boilerplate)
"""

import sys
from pathlib import Path

from loguru import logger

# Remove default handler to avoid duplicate output
logger.remove()

# Console output: INFO level, concise format
logger.add(
    sys.stderr,
    level="INFO",
    format="<green>{time:HH:mm:ss}</green> | <level>{level:<7}</level> | <cyan>{module}</cyan>:<cyan>{function}</cyan> | <level>{message}</level>",
    colorize=True,
)

# File output: DEBUG level, full details, rotation
LOG_DIR = Path(__file__).parent.parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

logger.add(
    LOG_DIR / "app.log",
    level="DEBUG",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level:<7} | {module}:{function}:{line} | {message}",
    rotation="10 MB",
    retention="7 days",
    compression="zip",
)

__all__ = ["logger"]
