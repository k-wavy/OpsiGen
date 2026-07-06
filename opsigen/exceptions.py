"""Custom exceptions used by the OpsiGen pipeline."""

from __future__ import annotations


class OpsiGenError(RuntimeError):
    """Base class for user-facing OpsiGen errors."""


class ConfigError(OpsiGenError):
    """Raised when a configuration file is missing required values."""


class InputValidationError(OpsiGenError):
    """Raised when an input file or record is invalid."""


class ExternalToolError(OpsiGenError):
    """Raised when an external executable fails."""
