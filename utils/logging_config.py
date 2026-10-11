"""Centralized structured logging. Call configure_logging() once at app startup."""

import logging
import sys
from pathlib import Path


def configure_logging(level: str = "INFO", log_dir: str = "logs") -> logging.Logger:
    """Configure root logger with console + rotating file output. Idempotent."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    if root.handlers:
        return logging.getLogger("customer_intelligence")
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    root.addHandler(console)
    file_handler = logging.FileHandler(Path(log_dir) / "app.log", encoding="utf-8")
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)
    return logging.getLogger("customer_intelligence")
