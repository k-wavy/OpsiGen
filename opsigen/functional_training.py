"""Training pipeline for functional/non-functional opsin classification."""

from __future__ import annotations

import csv
import json
import logging
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import FunctionalTrainingConfig
from .data import FunctionalGraphDataset, validate_graph_paths
from .exceptions import InputValidationError
from .logging_utils import configure_logging
from .model_io import import_model_definitions, load_pickle_model, save_pickle_model
from .training import calculate_l1_reg

LOGGER = logging.getLogger("opsigen.functional_training")


@dataclass(frozen=True)
class FunctionalTrainingResult:
    """Summary of a completed binary classifier run."""

    output_dir: Path
    best_checkpoint_path: Path | None
    final_checkpoint_path: Path
    means_path: Path
    stds_path: Path
    metrics_path: Path
    validation_predictions_path: Path
    metadata_path: Path
    epochs_completed: int
    best_metric: float | None


def _json_default(value: Any) -> str:
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _select_device(requested: str):
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError("Torch is required for training.") from exc

    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _set_seed(seed: int) -> None:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError("Torch is required for training.") from exc

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_model(config: FunctionalTrainingConfig, device: Any) -> Any:
    if config.model.initial_model_path:
        LOGGER.info("Loading initial model from %s", config.model.initial_model_path)
        return load_pickle_model(config.model.initial_model_path, device=device)

    model_defs = import_model_definitions()
    try:
        model_factory = getattr(model_defs, config.model.name)
    except AttributeError as exc:
        raise ValueError(f"Unknown model class: {config.model.name}") from exc

    return model_factory(
        config.model.number_features,
        config.model.hidden_layer_size,
        config.model.out_layer_size,
        device,
        dp_rate=config.model.dropout,
    ).to(device)


def sigmoid(values: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid for numpy arrays."""

    return np.where(
        values >= 0,
        1 / (1 + np.exp(-values)),
        np.exp(values) / (1 + np.exp(values)),
    )


def _safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def roc_auc_score_no_sklearn(labels: np.ndarray, probabilities: np.ndarray) -> float:
    """Compute ROC AUC with average ranks; returns NaN when one class is absent."""

    positives = int(np.sum(labels == 1))
    negatives = int(np.sum(labels == 0))
    if positives == 0 or negatives == 0:
        return float("nan")

    order = np.argsort(probabilities)
    sorted_scores = probabilities[order]
    ranks = np.empty(len(probabilities), dtype=float)
    start = 0
    while start < len(probabilities):
        end = start + 1
        while end < len(probabilities) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        average_rank = (start + 1 + end) / 2
        ranks[order[start:end]] = average_rank
        start = end

    positive_rank_sum = float(np.sum(ranks[labels == 1]))
    return (positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def average_precision_score_no_sklearn(labels: np.ndarray, probabilities: np.ndarray) -> float:
    """Compute average precision for the positive class."""

    positives = int(np.sum(labels == 1))
    if positives == 0:
        return float("nan")
    order = np.argsort(-probabilities)
    sorted_labels = labels[order]
    true_positive_count = 0
    precision_sum = 0.0
    for rank, label in enumerate(sorted_labels, start=1):
        if label == 1:
            true_positive_count += 1
            precision_sum += true_positive_count / rank
    return precision_sum / positives


def classification_metrics(
    labels: np.ndarray,
    logits: np.ndarray,
    *,
    threshold: float,
) -> dict[str, float]:
    """Return threshold and ranking metrics for binary logits."""

    labels = labels.astype(int)
    probabilities = sigmoid(logits)
    predictions = (probabilities >= threshold).astype(int)

    tp = int(np.sum((predictions == 1) & (labels == 1)))
    tn = int(np.sum((predictions == 0) & (labels == 0)))
    fp = int(np.sum((predictions == 1) & (labels == 0)))
    fn = int(np.sum((predictions == 0) & (labels == 1)))

    recall = _safe_divide(tp, tp + fn)
    specificity = _safe_divide(tn, tn + fp)
    precision = _safe_divide(tp, tp + fp)
    negative_predictive_value = _safe_divide(tn, tn + fn)
    f1 = _safe_divide(2 * precision * recall, precision + recall)
    accuracy = _safe_divide(tp + tn, len(labels))

    return {
        "accuracy": accuracy,
        "balanced_accuracy": (recall + specificity) / 2,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "negative_predictive_value": negative_predictive_value,
        "f1": f1,
        "auroc": roc_auc_score_no_sklearn(labels, probabilities),
        "average_precision": average_precision_score_no_sklearn(labels, probabilities),
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "positive_count": float(np.sum(labels == 1)),
        "negative_count": float(np.sum(labels == 0)),
    }


def _binary_loss(logits: Any, labels: Any, weights: Any, *, use_weights: bool):
    try:
        import torch.nn.functional as F
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError("Torch is required for training.") from exc

    loss_values = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    if use_weights:
        loss_values = loss_values * weights
    return loss_values.mean()


def _run_epoch(
    model: Any,
    dataloader: Any,
    optimizer: Any,
    config: FunctionalTrainingConfig,
    device: Any,
) -> tuple[float, dict[str, float]]:
    model.train()
    losses: list[float] = []
    logits_out: list[float] = []
    labels_out: list[float] = []

    for features, distances, labels, weights in dataloader:
        optimizer.zero_grad()
        labels = labels.to(device).double().reshape(-1)
        weights = weights.to(device).double().reshape(-1)
        logits = model.double().forward(features, distances, config.model.graph_threshold).reshape(-1)
        loss = _binary_loss(
            logits,
            labels,
            weights,
            use_weights=config.classifier.class_weighted_loss,
        )
        if config.fit.l1_lambda:
            loss = loss + calculate_l1_reg(model, config.fit.l1_lambda)
        loss.backward()
        optimizer.step()

        losses.append(float(loss.item()))
        logits_out.extend(float(value) for value in logits.detach().cpu().reshape(-1))
        labels_out.extend(float(value) for value in labels.detach().cpu().reshape(-1))

    metrics = classification_metrics(
        np.asarray(labels_out, dtype=float),
        np.asarray(logits_out, dtype=float),
        threshold=config.classifier.decision_threshold,
    )
    return float(np.mean(losses)) if losses else float("nan"), metrics


def _evaluate(
    model: Any,
    dataloader: Any,
    config: FunctionalTrainingConfig,
    device: Any,
) -> tuple[float, float, dict[str, float], np.ndarray, np.ndarray]:
    try:
        import torch
        import torch.nn.functional as F
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError("Torch is required for training.") from exc

    model.eval()
    losses: list[float] = []
    weighted_losses: list[float] = []
    logits_out: list[float] = []
    labels_out: list[float] = []

    with torch.no_grad():
        for features, distances, labels, weights in dataloader:
            labels = labels.to(device).double().reshape(-1)
            weights = weights.to(device).double().reshape(-1)
            logits = model.double().forward(features, distances, config.model.graph_threshold).reshape(-1)
            raw_loss = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
            losses.append(float(raw_loss.mean().item()))
            weighted_losses.append(float((raw_loss * weights).mean().item()))
            logits_out.extend(float(value) for value in logits.detach().cpu().reshape(-1))
            labels_out.extend(float(value) for value in labels.detach().cpu().reshape(-1))

    labels_array = np.asarray(labels_out, dtype=float)
    logits_array = np.asarray(logits_out, dtype=float)
    metrics = classification_metrics(
        labels_array,
        logits_array,
        threshold=config.classifier.decision_threshold,
    )
    return (
        float(np.mean(losses)) if losses else float("nan"),
        float(np.mean(weighted_losses)) if weighted_losses else float("nan"),
        metrics,
        labels_array,
        logits_array,
    )


def _prefixed(prefix: str, values: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in values.items()}


def _metric_is_better(value: float, best: float | None, mode: str) -> bool:
    if np.isnan(value):
        return False
    if best is None:
        return True
    return value < best if mode == "min" else value > best


def train_functional_classifier(
    config: FunctionalTrainingConfig,
    *,
    verbose: bool = False,
) -> FunctionalTrainingResult:
    """Train a class-imbalanced functional/non-functional graph classifier."""

    try:
        import torch
        from torch import optim
        from torch.utils.data import DataLoader, WeightedRandomSampler
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "Functional classifier training requires torch and torch-geometric."
        ) from exc

    if config.fit.batch_size != 1:
        LOGGER.warning(
            "The bundled graph models were written for batch_size=1; got batch_size=%s.",
            config.fit.batch_size,
        )

    output_dir = config.output_dir
    logs_dir = output_dir / "logs"
    checkpoints_dir = output_dir / "checkpoints"
    artifacts_dir = output_dir / "artifacts"
    for directory in (logs_dir, checkpoints_dir, artifacts_dir):
        directory.mkdir(parents=True, exist_ok=True)
    configure_logging(logs_dir, verbose=verbose)

    LOGGER.info("Starting functional classifier run '%s' in %s", config.run_name, output_dir)
    LOGGER.info("Positive class: %s", config.classifier.positive_class)
    validate_graph_paths(config.data)
    _set_seed(config.fit.seed)
    device = _select_device(config.fit.device)
    LOGGER.info("Using device: %s", device)

    model = _build_model(config, device)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.fit.learning_rate,
        weight_decay=config.fit.weight_decay,
    )

    train_dataset = FunctionalGraphDataset(config.data, config.data.train_wildtypes_list, config.classifier)
    train_counts = train_dataset.class_counts()
    if train_counts["positive"] == 0 or train_counts["negative"] == 0:
        raise InputValidationError(
            "Functional classifier training requires both classes in the training split. "
            f"Found positive={train_counts['positive']}, negative={train_counts['negative']}. "
            "Create a stratified split or move at least one non-functional and one functional opsin into training."
        )
    test_dataset = FunctionalGraphDataset(
        config.data,
        config.data.test_wildtypes_list,
        config.classifier,
        means=train_dataset.means,
        stds=train_dataset.stds,
    )

    means_path = artifacts_dir / "means.npy"
    stds_path = artifacts_dir / "stds.npy"
    np.save(means_path, train_dataset.means)
    np.save(stds_path, train_dataset.stds)

    if config.fit.weighted_sampler:
        sampler = WeightedRandomSampler(
            weights=torch.tensor(train_dataset.sample_weights(), dtype=torch.double),
            num_samples=config.fit.sampler_samples or len(train_dataset),
            replacement=True,
        )
        train_dataloader = DataLoader(train_dataset, batch_size=config.fit.batch_size, sampler=sampler)
    else:
        train_dataloader = DataLoader(train_dataset, batch_size=config.fit.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=config.fit.batch_size, shuffle=False)

    wandb_run = None
    if config.fit.use_wandb:
        try:
            import wandb
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise RuntimeError("W&B logging was requested but wandb is not installed.") from exc
        wandb_run = wandb.init(
            project=config.fit.wandb_project,
            name=config.run_name,
            config=json.loads(json.dumps(asdict(config), default=_json_default)),
        )

    metrics_path = logs_dir / "metrics.jsonl"
    validation_predictions_path = output_dir / "validation_predictions.csv"
    metadata_path = output_dir / "metadata.json"
    best_checkpoint_path: Path | None = None
    best_metric: float | None = None
    epochs_completed = 0
    final_labels = np.asarray([], dtype=float)
    final_logits = np.asarray([], dtype=float)

    with metrics_path.open("w") as metrics_file:
        for epoch in range(1, config.fit.epochs + 1):
            train_loss, train_metrics = _run_epoch(model, train_dataloader, optimizer, config, device)
            test_loss, weighted_test_loss, test_metrics, labels, logits = _evaluate(
                model,
                test_dataloader,
                config,
                device,
            )
            final_labels = labels
            final_logits = logits
            epochs_completed = epoch

            metrics = {
                "epoch": epoch,
                "train_loss": train_loss,
                "test_loss": test_loss,
                "weighted_test_loss": weighted_test_loss,
                **_prefixed("train", train_metrics),
                **_prefixed("test", test_metrics),
            }
            metrics_file.write(json.dumps(metrics) + "\n")
            metrics_file.flush()
            LOGGER.info(
                "Epoch %s/%s train_loss=%.4f test_loss=%.4f test_balanced_accuracy=%.4f "
                "test_precision=%.4f test_recall=%.4f",
                epoch,
                config.fit.epochs,
                train_loss,
                test_loss,
                test_metrics["balanced_accuracy"],
                test_metrics["precision"],
                test_metrics["recall"],
            )
            if wandb_run is not None:
                wandb_run.log(metrics)

            monitored_name = config.classifier.monitor_metric
            monitored_value = metrics.get(f"test_{monitored_name}", metrics.get(monitored_name))
            if monitored_value is None:
                raise ValueError(
                    f"Monitor metric {monitored_name!r} was not found. "
                    f"Available metrics include: {sorted(metrics)}"
                )
            if _metric_is_better(float(monitored_value), best_metric, config.classifier.monitor_mode):
                best_metric = float(monitored_value)
                best_checkpoint_path = save_pickle_model(model, checkpoints_dir / f"best_{config.fit.checkpoint_name}")

    probabilities = sigmoid(final_logits)
    predictions = (probabilities >= config.classifier.decision_threshold).astype(int)
    with validation_predictions_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["label", "logit", "probability_positive", "prediction"],
        )
        writer.writeheader()
        for label, logit, probability, prediction in zip(final_labels, final_logits, probabilities, predictions):
            writer.writerow(
                {
                    "label": int(label),
                    "logit": logit,
                    "probability_positive": probability,
                    "prediction": int(prediction),
                }
            )

    final_checkpoint_path = save_pickle_model(model, checkpoints_dir / config.fit.checkpoint_name)
    metadata = {
        "run_name": config.run_name,
        "config": json.loads(json.dumps(asdict(config), default=_json_default)),
        "positive_class": config.classifier.positive_class,
        "functional_threshold_nm": config.classifier.functional_threshold_nm,
        "decision_threshold": config.classifier.decision_threshold,
        "train_class_counts": train_counts,
        "test_class_counts": test_dataset.class_counts(),
        "outputs": {
            "best_checkpoint_path": str(best_checkpoint_path) if best_checkpoint_path else None,
            "final_checkpoint_path": str(final_checkpoint_path),
            "means_path": str(means_path),
            "stds_path": str(stds_path),
            "metrics_path": str(metrics_path),
            "validation_predictions_path": str(validation_predictions_path),
        },
        "epochs_completed": epochs_completed,
        "best_metric": best_metric,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))
    if wandb_run is not None:
        wandb_run.finish()

    LOGGER.info("Functional classifier training complete. Final checkpoint: %s", final_checkpoint_path)
    return FunctionalTrainingResult(
        output_dir=output_dir,
        best_checkpoint_path=best_checkpoint_path,
        final_checkpoint_path=final_checkpoint_path,
        means_path=means_path,
        stds_path=stds_path,
        metrics_path=metrics_path,
        validation_predictions_path=validation_predictions_path,
        metadata_path=metadata_path,
        epochs_completed=epochs_completed,
        best_metric=best_metric,
    )


def train_functional_classifier_from_config(
    path: str | Path,
    *,
    verbose: bool = False,
) -> FunctionalTrainingResult:
    """Load a config file and train a functional/non-functional classifier."""

    return train_functional_classifier(FunctionalTrainingConfig.from_file(path), verbose=verbose)
