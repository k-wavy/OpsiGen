"""Graph dataset utilities for OpsiGen training and prediction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .config import FunctionalClassifierSettings, GraphDataConfig
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
        if (config.features_column or config.dists_column) and not path.exists():
            continue
        if not path.exists() or not path.is_dir():
            raise InputValidationError(f"Missing {key} directory: {path}")


def load_graph_table(path: Path) -> pd.DataFrame:
    """Load a graph metadata table from Excel, CSV, or TSV."""

    if not path.exists():
        raise InputValidationError(f"Graph metadata table does not exist: {path}")
    suffix = path.suffix.lower()
    if suffix in {".csv"}:
        return pd.read_csv(path)
    if suffix in {".tsv", ".tab"}:
        return pd.read_csv(path, sep="\t")
    return pd.read_excel(path)


def load_name_set(path: Path) -> set[str]:
    """Load non-empty line-delimited names."""

    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


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
        self.excel_data = load_graph_table(config.excel_path)
        self.wildtypes_names = load_name_set(wildtypes_file)
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
        entry = self.excel_data.iloc[idx]
        if self.config.features_column and self.config.dists_column:
            for column in (self.config.features_column, self.config.dists_column):
                if column not in self.excel_data.columns:
                    raise InputValidationError(f"Excel dataset is missing graph path column: {column}")
            return (
                self._resolve_table_path(entry[self.config.dists_column]),
                self._resolve_table_path(entry[self.config.features_column]),
            )
        return (
            self.config.graph_dists_path / f"cutted_parts{idx}_dists.npy",
            self.config.graph_features_path / f"cutted_parts{idx}.npz",
        )

    def _resolve_table_path(self, value: object) -> Path:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            raise InputValidationError("Encountered an empty graph path in the training table.")
        path = Path(str(value)).expanduser()
        return path if path.is_absolute() else (self.config.excel_path.parent / path).resolve()

    def calculate_weights(self) -> np.ndarray:
        from collections import Counter

        wildtypes = self.get_category(self.config.wildtype_column)[: len(self)]
        wildtype_counts = Counter(wildtypes)
        return np.asarray([1 / wildtype_counts[wildtype] for wildtype in wildtypes], dtype=float)

    def calculate_stats(self) -> tuple[np.ndarray, np.ndarray]:
        values: list[np.ndarray] = []
        for idx in range(len(self)):
            graph_paths = self.graph_paths_for_index(idx)
            graph = read_graph(*graph_paths, indexes=self.config.indexes_to_keep)
            if graph is None:
                continue
            wildtype = self.get_category(self.config.wildtype_column)[idx]
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

        wildtype = self.get_category(self.config.wildtype_column)[idx]
        target = self.get_category(self.config.target_column)[idx]
        if wildtype not in self.wildtypes_names:
            return [], [], 0

        normalized = normalize_features(graph.features, self.means, self.stds)
        return normalized, graph.distances, target


class FunctionalGraphDataset:
    """Graph dataset for binary functional/non-functional classification."""

    def __init__(
        self,
        config: GraphDataConfig,
        wildtypes_file: Path,
        classifier: FunctionalClassifierSettings,
        *,
        means: np.ndarray | None = None,
        stds: np.ndarray | None = None,
    ) -> None:
        validate_graph_paths(config)
        self.config = config
        self.classifier = classifier
        self.excel_data = load_graph_table(config.excel_path)
        self.wildtypes_names = load_name_set(wildtypes_file)
        self.row_indices = self._select_rows()
        if not self.row_indices:
            raise InputValidationError(f"No rows matched split file: {wildtypes_file}")
        self.labels = np.asarray([self._label_for_index(idx) for idx in self.row_indices], dtype=np.float32)
        self._sample_weights = self._calculate_sample_weights()
        self.means = means
        self.stds = stds
        if self.means is None or self.stds is None:
            self.means, self.stds = self.calculate_stats()

    def _select_rows(self) -> list[int]:
        if self.config.wildtype_column not in self.excel_data.columns:
            raise InputValidationError(f"Excel dataset is missing required column: {self.config.wildtype_column}")
        length = self.excel_data.shape[0]
        if self.config.drop_last_row:
            length = max(0, length - 1)
        return [
            idx
            for idx in range(length)
            if str(self.excel_data.iloc[idx][self.config.wildtype_column]) in self.wildtypes_names
        ]

    def __len__(self) -> int:
        return len(self.row_indices)

    def _resolve_table_path(self, value: object) -> Path:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            raise InputValidationError("Encountered an empty graph path in the training table.")
        path = Path(str(value)).expanduser()
        return path if path.is_absolute() else (self.config.excel_path.parent / path).resolve()

    def graph_paths_for_table_index(self, table_idx: int) -> tuple[Path, Path]:
        entry = self.excel_data.iloc[table_idx]
        if self.config.features_column and self.config.dists_column:
            for column in (self.config.features_column, self.config.dists_column):
                if column not in self.excel_data.columns:
                    raise InputValidationError(f"Excel dataset is missing graph path column: {column}")
            return (
                self._resolve_table_path(entry[self.config.dists_column]),
                self._resolve_table_path(entry[self.config.features_column]),
            )
        return (
            self.config.graph_dists_path / f"cutted_parts{table_idx}_dists.npy",
            self.config.graph_features_path / f"cutted_parts{table_idx}.npz",
        )

    def _label_for_index(self, table_idx: int) -> float:
        if self.config.target_column not in self.excel_data.columns:
            raise InputValidationError(f"Excel dataset is missing target column: {self.config.target_column}")
        value = self.excel_data.iloc[table_idx][self.config.target_column]
        if pd.isna(value):
            raise InputValidationError(f"Missing lmax target at row {table_idx}.")
        functional = float(value) > self.classifier.functional_threshold_nm
        if self.classifier.positive_class == "functional":
            return float(functional)
        return float(not functional)

    def class_counts(self) -> dict[str, int]:
        positives = int(np.sum(self.labels == 1))
        negatives = int(np.sum(self.labels == 0))
        return {
            "positive": positives,
            "negative": negatives,
            self.classifier.positive_class: positives,
            ("functional" if self.classifier.positive_class == "nonfunctional" else "nonfunctional"): negatives,
        }

    def class_weights(self) -> dict[int, float]:
        counts = self.class_counts()
        positives = counts["positive"]
        negatives = counts["negative"]
        if positives == 0 or negatives == 0:
            return {0: 1.0, 1: 1.0}
        total = positives + negatives
        return {
            0: total / (2 * negatives),
            1: total / (2 * positives),
        }

    def _calculate_sample_weights(self) -> np.ndarray:
        weights = self.class_weights()
        return np.asarray([weights[int(label)] for label in self.labels], dtype=np.float64)

    def sample_weights(self) -> np.ndarray:
        return self._sample_weights.copy()

    def calculate_stats(self) -> tuple[np.ndarray, np.ndarray]:
        values: list[np.ndarray] = []
        for table_idx in self.row_indices:
            graph_paths = self.graph_paths_for_table_index(table_idx)
            graph = read_graph(*graph_paths, indexes=self.config.indexes_to_keep)
            if graph is not None:
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

    def __getitem__(self, idx: int):
        table_idx = self.row_indices[idx]
        graph_paths = self.graph_paths_for_table_index(table_idx)
        graph = read_graph(*graph_paths, indexes=self.config.indexes_to_keep)
        if graph is None:
            raise InputValidationError(f"Missing graph files for table row {table_idx}: {graph_paths}")
        normalized = normalize_features(graph.features, self.means, self.stds)
        label = self.labels[idx]
        weight = self._sample_weights[idx]
        return normalized, graph.distances, label, weight
