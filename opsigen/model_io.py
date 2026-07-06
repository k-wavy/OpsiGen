"""Model loading and saving helpers."""

from __future__ import annotations

import pickle
import sys
from pathlib import Path
from typing import Any

from .exceptions import InputValidationError


def import_model_definitions() -> Any:
    """Import model definitions and register legacy pickle aliases."""

    try:
        from . import models
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise InputValidationError(
            "Torch and torch-geometric are required for OpsiGen model loading. "
            "Install the project requirements before training or prediction."
        ) from exc

    sys.modules.setdefault("models", models)
    return models


def load_pickle_model(path: str | Path, *, device: Any | None = None) -> Any:
    """Load a pickled model with legacy module aliases registered."""

    model_path = Path(path)
    if not model_path.exists():
        raise InputValidationError(f"Model file does not exist: {model_path}")
    import_model_definitions()
    with model_path.open("rb") as handle:
        model = pickle.load(handle)
    if device is not None and hasattr(model, "to"):
        model = model.to(device)
    return model


def save_pickle_model(model: Any, path: str | Path) -> Path:
    """Save a model pickle and return its path."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        pickle.dump(model, handle)
    return output_path
