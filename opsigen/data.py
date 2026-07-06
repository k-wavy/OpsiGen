"""Graph dataset utilities for OpsiGen training and prediction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .config import GraphDataConfig
from .exceptions import InputValidationError


@dataclass(frozen=True)
class GraphArrays:
    """Feature and distance arrays for one graph."""

    features: np.ndarray
    distances: np.ndarray


def validate_graph_paths(config: GraphDataConfig) -> None:
    """Validate dataset paths before training starts."""

    required_files = {
        "excel_path": config.excel_path,
        "train_wildtypes_list": config.train_wildtypes_list,
        "test_wildtypes_list": config.test_wildtypes_list,
    }
    for key, path in required_files.items():
        if not path.exists():
            raise InputValidationError(f"Missing {key}: {path}")
    for key, path in {
        "graph_features_path": config.graph_features_path,
        "graph_dists_path": config.graph_dists_path,
    }.items():
        if not path.exists() or not path.is_dir():
            raise InputValidationError(f"Missing {key} directory: {path}")


def read_graph(
    dists_file_name: str | Path,
    features_file_name: str | Path,
    indexes: Sequence[int],
) -> GraphArrays | None:
    """Read one graph feature/distance pair."""

    dists_path = Path(dists_file_name)
    features_path = Path(features_file_name)
    if not dists_path.exists() or not features_path.exists():
        return None

    distances = np.load(dists_path)
    features = np.load(features_path)
    if len(features.shape) != 2:
        raise InputValidationError(f"Expected a 2D feature array at {features_path}; got {features.shape}")
    if max(indexes, default=-1) >= features.shape[1]:
        raise InputValidationError(
            f"Feature index {max(indexes)} is out of bounds for {features_path} with shape {features.shape}"
        )
    return GraphArrays(features=features[:, indexes], distances=distances)


def normalize_features(
    features: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
) -> np.ndarray:
    """Normalize feature columns while dropping zero-variance columns."""

    if features.shape[1] != means.shape[0] or means.shape != stds.shape:
        raise InputValidationError(
            "Feature normalization shape mismatch: "
            f"features={features.shape}, means={means.shape}, stds={stds.shape}"
        )
    mask = stds != 0
    return (features[:, mask] - means[mask]) / stds[mask]


class GraphDataset:
    """Torch Dataset-compatible graph dataset.

    The class intentionally avoids importing torch at module import time so that
    configs and notebooks can be inspected without a full training environment.
    """

    feature_length = 36

    def __init__(
        self,
        config: GraphDataConfig,
        wildtypes_file: Path,
        *,
        means: np.ndarray | None = None,
        stds: np.ndarray | None = None,
    ) -> None:
        validate_graph_paths(config)
        self.config = config
        self.excel_data = pd.read_excel(config.excel_path)
        self.wildtypes_names = {
            line.strip()
            for line in wildtypes_file.read_text().splitlines()
            if line.strip()
        }
        self.means = means
        self.stds = stds
        if self.means is None or self.stds is None:
            self.means, self.stds = self.calculate_stats()

    def __len__(self) -> int:
        length = self.excel_data.shape[0]
        return max(0, length - 1) if self.config.drop_last_row else length

    def get_category(self, category: str) -> list:
        if category not in self.excel_data.columns:
            raise InputValidationError(f"Excel dataset is missing required column: {category}")
        return list(self.excel_data[category])

    def graph_paths_for_index(self, idx: int) -> tuple[Path, Path]:
        return (
            self.config.graph_dists_path / f"cutted_parts{idx}_dists.npy",
            self.config.graph_features_path / f"cutted_parts{idx}.npz",
        )

    def calculate_weights(self) -> np.ndarray:
        from collections import Counter

        wildtypes = self.get_category("Wildtype")[: len(self)]
        wildtype_counts = Counter(wildtypes)
        return np.asarray([1 / wildtype_counts[wildtype] for wildtype in wildtypes], dtype=float)

    def calculate_stats(self) -> tuple[np.ndarray, np.ndarray]:
        values: list[np.ndarray] = []
        for idx in range(len(self)):
            graph_paths = self.graph_paths_for_index(idx)
            graph = read_graph(*graph_paths, indexes=self.config.indexes_to_keep)
            if graph is None:
                continue
            wildtype = self.get_category("Wildtype")[idx]
            if wildtype not in self.wildtypes_names:
                continue
            values.append(graph.features)

        if not values:
            raise InputValidationError("No training graphs were found for normalization statistics.")

        stacked = np.vstack(values)
        means = np.mean(stacked, axis=0)
        stds = np.std(stacked, axis=0)
        if not self.config.dataset_normalize_last:
            means[-3:] = 0
            stds[-3:] = 1
        return means, stds

    def get_specific_item(
        self,
        dists_path: str | Path,
        features_path: str | Path,
    ) -> GraphArrays:
        graph = read_graph(dists_path, features_path, indexes=self.config.indexes_to_keep)
        if graph is None:
            raise InputValidationError(f"Missing graph files: features={features_path}, dists={dists_path}")
        return GraphArrays(
            features=normalize_features(graph.features, self.means, self.stds),
            distances=graph.distances,
        )

    def __getitem__(self, idx: int):
        graph_paths = self.graph_paths_for_index(idx)
        graph = read_graph(*graph_paths, indexes=self.config.indexes_to_keep)
        if graph is None:
            return [], [], 0

        wildtype = self.get_category("Wildtype")[idx]
        target = self.get_category("lmax")[idx]
        if wildtype not in self.wildtypes_names:
            return [], [], 0

        normalized = normalize_features(graph.features, self.means, self.stds)
        return normalized, graph.distances, target
