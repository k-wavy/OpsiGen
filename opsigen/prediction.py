"""Central prediction pipeline for OpsiGen."""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import (
    GraphDataConfig,
    ModelConfig,
    OpsinRecord,
    PredictionConfig,
    load_config_mapping,
    resolve_path,
)
from .data import GraphArrays, normalize_features, read_graph
from .exceptions import ConfigError, InputValidationError
from .logging_utils import configure_logging
from .model_io import load_pickle_model
from .preprocessing import PreprocessedRecord, preprocess_opsins

LOGGER = logging.getLogger("opsigen.prediction")


@dataclass(frozen=True)
class PredictionRow:
    """Prediction output for one opsin graph."""

    id: str
    prediction: float
    features_path: Path
    dists_path: Path


@dataclass(frozen=True)
class PredictionResult:
    """Summary of a prediction run."""

    output_dir: Path
    predictions_path: Path
    metadata_path: Path
    rows: tuple[PredictionRow, ...]


def _json_default(value: Any) -> str:
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _select_device(requested: str):
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError("Prediction requires torch and torch-geometric.") from exc

    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def load_normalization(means_path: Path, stds_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load feature normalization arrays."""

    if not means_path.exists():
        raise InputValidationError(f"Missing means array: {means_path}")
    if not stds_path.exists():
        raise InputValidationError(f"Missing stds array: {stds_path}")
    means = np.load(means_path)
    stds = np.load(stds_path)
    if means.shape != stds.shape:
        raise InputValidationError(f"Normalization arrays have different shapes: {means.shape} vs {stds.shape}")
    return means, stds


def load_prediction_graph(
    *,
    features_path: Path,
    dists_path: Path,
    indexes_to_keep: tuple[int, ...],
    means: np.ndarray,
    stds: np.ndarray,
) -> GraphArrays:
    """Load and normalize one prediction graph."""

    graph = read_graph(dists_path, features_path, indexes=indexes_to_keep)
    if graph is None:
        raise InputValidationError(f"Missing prediction graph files: features={features_path}, dists={dists_path}")
    return GraphArrays(
        features=normalize_features(graph.features, means, stds),
        distances=graph.distances,
    )


def predict_graph(model: Any, graph: GraphArrays, *, graph_threshold: float) -> float:
    """Run prediction on a normalized graph."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError("Prediction requires torch and torch-geometric.") from exc

    model.eval()
    with torch.no_grad():
        prediction = model.double().forward(
            torch.tensor(graph.features),
            torch.tensor(graph.distances),
            graph_threshold,
        )
    return float(prediction.detach().cpu().reshape(-1)[0])


def _record_graph_paths(
    record: OpsinRecord,
    preprocessed_by_id: dict[str, PreprocessedRecord],
) -> tuple[Path, Path]:
    if record.features_path and record.dists_path:
        return record.features_path, record.dists_path
    safe_id = next((key for key in preprocessed_by_id if key == record.id), None)
    if safe_id is None:
        from .preprocessing.pipeline import sanitize_id

        safe_id = sanitize_id(record.id)
    if safe_id in preprocessed_by_id:
        item = preprocessed_by_id[safe_id]
        return item.features_path, item.dists_path
    raise InputValidationError(
        f"Record {record.id} has no features/dists paths and was not preprocessed. "
        "Provide graph files or enable preprocessing in the prediction config."
    )


def run_predictions(config: PredictionConfig, *, verbose: bool = False) -> PredictionResult:
    """Run batch prediction from a normalized config."""

    logs_dir = config.output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(logs_dir, verbose=verbose)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Starting prediction for %s record(s)", len(config.records))
    needs_preprocessing = [
        record
        for record in config.records
        if not (record.features_path and record.dists_path)
    ]
    preprocessed: list[PreprocessedRecord] = []
    if needs_preprocessing:
        if config.preprocessing is None:
            missing = ", ".join(record.id for record in needs_preprocessing)
            raise ConfigError(
                "Prediction records without features_path/dists_path require a preprocessing config. "
                f"Missing graph files for: {missing}"
            )
        preprocessed = preprocess_opsins(needs_preprocessing, config.preprocessing)

    preprocessed_by_id = {record.id: record for record in preprocessed}
    means, stds = load_normalization(config.means_path, config.stds_path)
    device = _select_device(config.device)
    model = load_pickle_model(config.model_path, device=device)

    rows: list[PredictionRow] = []
    for record in config.records:
        features_path, dists_path = _record_graph_paths(record, preprocessed_by_id)
        graph = load_prediction_graph(
            features_path=features_path,
            dists_path=dists_path,
            indexes_to_keep=config.indexes_to_keep,
            means=means,
            stds=stds,
        )
        prediction = predict_graph(model, graph, graph_threshold=config.graph_threshold)
        rows.append(
            PredictionRow(
                id=record.id,
                prediction=prediction,
                features_path=features_path,
                dists_path=dists_path,
            )
        )
        LOGGER.info("Predicted %s: %.3f", record.id, prediction)

    predictions_path = config.output_dir / "predictions.csv"
    with predictions_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["id", "prediction", "features_path", "dists_path"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "id": row.id,
                    "prediction": row.prediction,
                    "features_path": row.features_path,
                    "dists_path": row.dists_path,
                }
            )

    metadata_path = config.output_dir / "metadata.json"
    metadata = {
        "config": json.loads(json.dumps(asdict(config), default=_json_default)),
        "predictions_path": str(predictions_path),
        "prediction_count": len(rows),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))
    LOGGER.info("Prediction complete. Results: %s", predictions_path)
    return PredictionResult(
        output_dir=config.output_dir,
        predictions_path=predictions_path,
        metadata_path=metadata_path,
        rows=tuple(rows),
    )


def predict_from_config(path: str | Path, *, verbose: bool = False) -> PredictionResult:
    """Load a prediction config and run batch prediction."""

    return run_predictions(PredictionConfig.from_file(path), verbose=verbose)


def predict_one_graph(
    *,
    model_path: Path,
    means_path: Path,
    stds_path: Path,
    features_path: Path,
    dists_path: Path,
    indexes_to_keep: tuple[int, ...],
    graph_threshold: float,
    device: str = "auto",
) -> float:
    """Predict one already-preprocessed graph."""

    means, stds = load_normalization(means_path, stds_path)
    graph = load_prediction_graph(
        features_path=features_path,
        dists_path=dists_path,
        indexes_to_keep=indexes_to_keep,
        means=means,
        stds=stds,
    )
    torch_device = _select_device(device)
    model = load_pickle_model(model_path, device=torch_device)
    return predict_graph(model, graph, graph_threshold=graph_threshold)


def legacy_predict_one(
    *,
    config_path: str | Path,
    model_path: str | Path,
    output_file: str | Path,
    features_path: str | Path,
    dists_path: str | Path,
) -> float:
    """Compatibility shim for ``predict/calculate_one_rhodopsin.py`` arguments."""

    mapping, base_dir = load_config_mapping(config_path)
    data_config = GraphDataConfig.from_mapping(mapping, base_dir)
    model_config = ModelConfig.from_mapping(mapping, base_dir)
    means_path = resolve_path(mapping.get("means_path"), base_dir) or (base_dir / "means.npy")
    stds_path = resolve_path(mapping.get("stds_path"), base_dir) or (base_dir / "stds.npy")
    model_resolved = _resolve_legacy_cli_path(model_path, base_dir)
    features_resolved = _resolve_legacy_cli_path(features_path, Path.cwd())
    dists_resolved = _resolve_legacy_cli_path(dists_path, Path.cwd())
    prediction = predict_one_graph(
        model_path=model_resolved,
        means_path=means_path,
        stds_path=stds_path,
        features_path=features_resolved,
        dists_path=dists_resolved,
        indexes_to_keep=data_config.indexes_to_keep,
        graph_threshold=model_config.graph_threshold,
    )
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(f"absorption wavelength is {prediction}")
    return prediction


def _resolve_legacy_cli_path(value: str | Path, fallback_base: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    cwd_candidate = (Path.cwd() / path).resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    return (fallback_base / path).resolve()
