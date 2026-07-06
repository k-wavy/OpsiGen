"""Logging helpers for command-line and notebook workflows."""

from __future__ import annotations

import logging
from pathlib import Path


def configure_logging(log_dir: Path | None = None, *, verbose: bool = False) -> logging.Logger:
    """Configure root logging and optionally write a pipeline log file."""

    level = logging.DEBUG if verbose else logging.INFO
    logger = logging.getLogger("opsigen")
    logger.setLevel(level)
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if not any(getattr(handler, "_opsigen_console", False) for handler in logger.handlers):
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(level)
        console_handler._opsigen_console = True  # type: ignore[attr-defined]
        logger.addHandler(console_handler)

    for handler in logger.handlers:
        handler.setLevel(level)

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "opsigen.log"
        if not any(getattr(handler, "_opsigen_log_path", None) == log_path for handler in logger.handlers):
            file_handler = logging.FileHandler(log_path)
            file_handler.setFormatter(formatter)
            file_handler.setLevel(level)
            file_handler._opsigen_log_path = log_path  # type: ignore[attr-defined]
            logger.addHandler(file_handler)

    return logger
