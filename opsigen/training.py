"""Central training pipeline for OpsiGen."""

from __future__ import annotations

import json
import logging
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import TrainingConfig
from .data import GraphDataset, validate_graph_paths
from .logging_utils import configure_logging
from .model_io import import_model_definitions, load_pickle_model, save_pickle_model

LOGGER = logging.getLogger("opsigen.training")
EV_TO_NM = 1239.8


@dataclass(frozen=True)
class TrainingResult:
    """Summary of a completed training run."""

    output_dir: Path
    best_checkpoint_path: Path | None
    final_checkpoint_path: Path
    means_path: Path
    stds_path: Path
    metrics_path: Path
    metadata_path: Path
    epochs_completed: int
    best_test_loss: float | None


def _json_default(value: Any) -> str:
    if isinstance(value, Path):
        return str(value)
    return str(value)


def calculate_l1_reg(model: Any, l1_lambda: float):
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError("Torch is required for training.") from exc

    return l1_lambda * sum(torch.norm(parameter, 1) for parameter in model.parameters())


def calculate_energy_loss(prediction: Any, target: Any) -> float:
    """Calculate the historical energy-space error used by legacy scripts."""

    pred_value = float(prediction.detach().cpu().reshape(-1)[0])
    target_value = float(target.detach().cpu().reshape(-1)[0])
    if pred_value < 3 or target_value < 3:
        return 7.0
    return (EV_TO_NM / pred_value) - (EV_TO_NM / target_value)


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


def _build_model(config: TrainingConfig, device: Any) -> Any:
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


def _valid_target(target: Any) -> bool:
    try:
        return float(target.reshape(-1)[0]) != 0.0
    except AttributeError:
        return float(target) != 0.0


def _run_epoch(model: Any, dataloader: Any, optimizer: Any, config: TrainingConfig, device: Any) -> tuple[float, int]:
    model.train()
    loss_sum = 0.0
    sample_count = 0
    for features, distances, target in dataloader:
        if not _valid_target(target):
            continue
        optimizer.zero_grad()
        target = target.to(device).double()
        prediction = model.double().forward(features, distances, config.model.graph_threshold)
        loss = torch_norm(prediction - target) + calculate_l1_reg(model, config.fit.l1_lambda)
        loss.backward()
        optimizer.step()
        loss_sum += float(loss.item())
        sample_count += 1
    return loss_sum / sample_count if sample_count else float("nan"), sample_count


def torch_norm(value: Any) -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError("Torch is required for training.") from exc

    return torch.norm(value)


def _evaluate(model: Any, dataloader: Any, config: TrainingConfig, device: Any) -> tuple[float, float, int]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError("Torch is required for training.") from exc

    model.eval()
    loss_sum = 0.0
    energy_loss_sum = 0.0
    sample_count = 0
    with torch.no_grad():
        for features, distances, target in dataloader:
            if not _valid_target(target):
                continue
            target = target.to(device).double()
            prediction = model.double().forward(features, distances, config.model.graph_threshold)
            loss = torch_norm(prediction - target)
            loss_sum += float(loss.item())
            energy_loss_sum += calculate_energy_loss(prediction, target)
            sample_count += 1
    if not sample_count:
        return float("nan"), float("nan"), 0
    return loss_sum / sample_count, energy_loss_sum / sample_count, sample_count


def train_model(config: TrainingConfig, *, verbose: bool = False) -> TrainingResult:
    """Run the full training workflow from a normalized config."""

    try:
        import torch
        from torch import optim
        from torch.utils.data import DataLoader, WeightedRandomSampler
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "Training requires torch and torch-geometric. Install the project requirements first."
        ) from exc

    output_dir = config.output_dir
    logs_dir = output_dir / "logs"
    checkpoints_dir = output_dir / "checkpoints"
    artifacts_dir = output_dir / "artifacts"
    for directory in (logs_dir, checkpoints_dir, artifacts_dir):
        directory.mkdir(parents=True, exist_ok=True)
    configure_logging(logs_dir, verbose=verbose)

    LOGGER.info("Starting training run '%s' in %s", config.run_name, output_dir)
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

    train_dataset = GraphDataset(config.data, config.data.train_wildtypes_list)
    test_dataset = GraphDataset(
        config.data,
        config.data.test_wildtypes_list,
        means=train_dataset.means,
        stds=train_dataset.stds,
    )

    means_path = artifacts_dir / "means.npy"
    stds_path = artifacts_dir / "stds.npy"
    np.save(means_path, train_dataset.means)
    np.save(stds_path, train_dataset.stds)

    if config.fit.weighted_sampler:
        weights = train_dataset.calculate_weights()
        sampler = WeightedRandomSampler(
            weights=weights,
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
    metadata_path = output_dir / "metadata.json"
    best_checkpoint_path: Path | None = None
    best_test_loss: float | None = None
    epochs_completed = 0

    with metrics_path.open("w") as metrics_file:
        for epoch in range(1, config.fit.epochs + 1):
            train_loss, train_count = _run_epoch(model, train_dataloader, optimizer, config, device)
            test_loss, energy_loss, test_count = _evaluate(model, test_dataloader, config, device)
            epochs_completed = epoch

            metrics = {
                "epoch": epoch,
                "train_loss": train_loss,
                "test_loss": test_loss,
                "energy_loss": energy_loss,
                "train_samples": train_count,
                "test_samples": test_count,
            }
            metrics_file.write(json.dumps(metrics) + "\n")
            metrics_file.flush()
            LOGGER.info(
                "Epoch %s/%s train_loss=%.4f test_loss=%.4f energy_loss=%.4f",
                epoch,
                config.fit.epochs,
                train_loss,
                test_loss,
                energy_loss,
            )
            if wandb_run is not None:
                wandb_run.log(metrics)

            if not np.isnan(test_loss) and (best_test_loss is None or test_loss < best_test_loss):
                best_test_loss = test_loss
                best_checkpoint_path = save_pickle_model(model, checkpoints_dir / f"best_{config.fit.checkpoint_name}")

            if config.fit.test_goal is not None and not np.isnan(test_loss) and test_loss < config.fit.test_goal:
                LOGGER.info("Stopping early because test_loss %.4f < test_goal %.4f", test_loss, config.fit.test_goal)
                break

    final_checkpoint_path = save_pickle_model(model, checkpoints_dir / config.fit.checkpoint_name)
    metadata = {
        "run_name": config.run_name,
        "config": json.loads(json.dumps(asdict(config), default=_json_default)),
        "outputs": {
            "best_checkpoint_path": str(best_checkpoint_path) if best_checkpoint_path else None,
            "final_checkpoint_path": str(final_checkpoint_path),
            "means_path": str(means_path),
            "stds_path": str(stds_path),
            "metrics_path": str(metrics_path),
        },
        "epochs_completed": epochs_completed,
        "best_test_loss": best_test_loss,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))
    if wandb_run is not None:
        wandb_run.finish()

    LOGGER.info("Training complete. Final checkpoint: %s", final_checkpoint_path)
    return TrainingResult(
        output_dir=output_dir,
        best_checkpoint_path=best_checkpoint_path,
        final_checkpoint_path=final_checkpoint_path,
        means_path=means_path,
        stds_path=stds_path,
        metrics_path=metrics_path,
        metadata_path=metadata_path,
        epochs_completed=epochs_completed,
        best_test_loss=best_test_loss,
    )


def train_from_config(path: str | Path, *, verbose: bool = False) -> TrainingResult:
    """Load a config file and run training."""

    return train_model(TrainingConfig.from_file(path), verbose=verbose)
