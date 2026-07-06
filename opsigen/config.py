"""Configuration loading and normalization for OpsiGen pipelines."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .exceptions import ConfigError


DEFAULT_ALIGNED_POSITIONS: tuple[int, ...] = (
    429,
    465,
    469,
    597,
    599,
    600,
    603,
    604,
    607,
    651,
    652,
    655,
    687,
    690,
    691,
    694,
    774,
    777,
    778,
    781,
    828,
    832,
    835,
    836,
)


def load_config_mapping(path: str | Path) -> tuple[dict[str, Any], Path]:
    """Load a JSON or YAML config file and return its mapping plus base directory."""

    config_path = Path(path).expanduser().resolve()
    if not config_path.exists():
        raise ConfigError(f"Configuration file does not exist: {config_path}")

    text = config_path.read_text()
    if config_path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ConfigError("YAML config support requires PyYAML to be installed.") from exc
        loaded = yaml.safe_load(text)
    else:
        loaded = json.loads(text)

    if not isinstance(loaded, dict):
        raise ConfigError(f"Configuration file must contain a mapping: {config_path}")
    return loaded, config_path.parent


def resolve_path(value: str | Path | None, base_dir: Path) -> Path | None:
    """Resolve a config path relative to the config file that contained it."""

    if value in (None, ""):
        return None
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base_dir / path).resolve()


def require_path(value: str | Path | None, base_dir: Path, key: str) -> Path:
    """Resolve a required path field."""

    path = resolve_path(value, base_dir)
    if path is None:
        raise ConfigError(f"Missing required config value: {key}")
    return path


def _mapping_at(mapping: Mapping[str, Any], *keys: str) -> Mapping[str, Any]:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, Mapping):
            return {}
        current = current.get(key, {})
    return current if isinstance(current, Mapping) else {}


def _get_nested(mapping: Mapping[str, Any], keys: Iterable[str], default: Any = None) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return default


@dataclass(frozen=True)
class GraphDataConfig:
    """Paths and feature-selection settings for graph datasets."""

    excel_path: Path
    graph_features_path: Path
    graph_dists_path: Path
    train_wildtypes_list: Path
    test_wildtypes_list: Path
    indexes_to_keep: tuple[int, ...] = tuple(range(36))
    dataset_normalize_last: bool = True
    drop_last_row: bool = True
    id_column: str = "Name"
    target_column: str = "lmax"
    wildtype_column: str = "Wildtype"
    features_column: str | None = None
    dists_column: str | None = None

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any], base_dir: Path) -> "GraphDataConfig":
        data = _mapping_at(mapping, "data")
        source = data or mapping
        return cls(
            excel_path=require_path(source.get("excel_path"), base_dir, "data.excel_path"),
            graph_features_path=require_path(
                source.get("graph_features_path"), base_dir, "data.graph_features_path"
            ),
            graph_dists_path=require_path(source.get("graph_dists_path"), base_dir, "data.graph_dists_path"),
            train_wildtypes_list=require_path(
                source.get("train_wildtypes_list"), base_dir, "data.train_wildtypes_list"
            ),
            test_wildtypes_list=require_path(
                source.get("test_wildtypes_list"), base_dir, "data.test_wildtypes_list"
            ),
            indexes_to_keep=tuple(int(i) for i in source.get("indexes_to_keep", range(36))),
            dataset_normalize_last=bool(source.get("dataset_normalize_last", True)),
            drop_last_row=bool(source.get("drop_last_row", True)),
            id_column=str(source.get("id_column", "Name")),
            target_column=str(source.get("target_column", "lmax")),
            wildtype_column=str(source.get("wildtype_column", "Wildtype")),
            features_column=source.get("features_column"),
            dists_column=source.get("dists_column"),
        )


@dataclass(frozen=True)
class ModelConfig:
    """Model architecture settings."""

    name: str
    number_features: int
    hidden_layer_size: int
    out_layer_size: int
    graph_threshold: float
    dropout: float = 0.1
    initial_model_path: Path | None = None

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any], base_dir: Path) -> "ModelConfig":
        model = _mapping_at(mapping, "model")
        source = model or mapping
        return cls(
            name=str(source.get("name", source.get("model_name", "GAT21Model"))),
            number_features=int(source.get("number_features", source.get("c_in", 34))),
            hidden_layer_size=int(source.get("hidden_layer_size", source.get("c_hidden", 40))),
            out_layer_size=int(source.get("out_layer_size", source.get("c_out", 30))),
            graph_threshold=float(source.get("graph_threshold", source.get("graph_th", 2))),
            dropout=float(source.get("dropout", source.get("dp_rate", 0.1))),
            initial_model_path=resolve_path(source.get("initial_model_path"), base_dir),
        )


@dataclass(frozen=True)
class FitConfig:
    """Training loop settings."""

    epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    l1_lambda: float = 1e-4
    batch_size: int = 1
    weighted_sampler: bool = True
    sampler_samples: int | None = None
    test_goal: float | None = None
    seed: int = 7
    device: str = "auto"
    use_wandb: bool = False
    wandb_project: str | None = None
    checkpoint_name: str = "model.pkl"

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "FitConfig":
        fit = _mapping_at(mapping, "training")
        source = fit or mapping
        return cls(
            epochs=int(source.get("epochs", 100)),
            learning_rate=float(source.get("learning_rate", source.get("lr", 1e-4))),
            weight_decay=float(source.get("weight_decay", 1e-5)),
            l1_lambda=float(source.get("l1_lambda", 1e-4)),
            batch_size=int(source.get("batch_size", 1)),
            weighted_sampler=bool(source.get("weighted_sampler", True)),
            sampler_samples=(
                int(source["sampler_samples"]) if source.get("sampler_samples") is not None else None
            ),
            test_goal=(
                float(source["test_goal"]) if source.get("test_goal") not in (None, "") else None
            ),
            seed=int(source.get("seed", 7)),
            device=str(source.get("device", "auto")),
            use_wandb=bool(source.get("use_wandb", False)),
            wandb_project=source.get("wandb_project"),
            checkpoint_name=str(source.get("checkpoint_name", "model.pkl")),
        )


@dataclass(frozen=True)
class TrainingConfig:
    """Full training pipeline configuration."""

    data: GraphDataConfig
    model: ModelConfig
    fit: FitConfig
    output_dir: Path
    run_name: str = "opsigen-training"

    @classmethod
    def from_file(cls, path: str | Path) -> "TrainingConfig":
        mapping, base_dir = load_config_mapping(path)
        outputs = _mapping_at(mapping, "outputs")
        return cls(
            data=GraphDataConfig.from_mapping(mapping, base_dir),
            model=ModelConfig.from_mapping(mapping, base_dir),
            fit=FitConfig.from_mapping(mapping),
            output_dir=require_path(
                outputs.get("output_dir", mapping.get("output_dir", "runs/train")),
                base_dir,
                "outputs.output_dir",
            ),
            run_name=str(outputs.get("run_name", mapping.get("run_name", "opsigen-training"))),
        )


@dataclass(frozen=True)
class OpsinRecord:
    """One opsin to preprocess and/or predict."""

    id: str
    fasta_path: Path | None = None
    pdb_path: Path | None = None
    features_path: Path | None = None
    dists_path: Path | None = None

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any], base_dir: Path, index: int) -> "OpsinRecord":
        record_id = str(mapping.get("id") or mapping.get("name") or f"opsin_{index:04d}")
        return cls(
            id=record_id,
            fasta_path=resolve_path(mapping.get("fasta_path"), base_dir),
            pdb_path=resolve_path(mapping.get("pdb_path"), base_dir),
            features_path=resolve_path(mapping.get("features_path"), base_dir),
            dists_path=resolve_path(mapping.get("dists_path"), base_dir),
        )


def load_opsin_manifest(path: Path) -> list[dict[str, Any]]:
    """Load opsin records from CSV, TSV, or JSON."""

    if not path.exists():
        raise ConfigError(f"Opsin manifest does not exist: {path}")
    suffix = path.suffix.lower()
    if suffix == ".json":
        loaded = json.loads(path.read_text())
        if isinstance(loaded, dict):
            loaded = loaded.get("opsins", loaded.get("records"))
        if not isinstance(loaded, list):
            raise ConfigError(f"JSON manifest must be a list or contain an 'opsins' list: {path}")
        return [dict(row) for row in loaded]

    delimiter = "\t" if suffix in {".tsv", ".tab"} else ","
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        return [dict(row) for row in reader]


@dataclass(frozen=True)
class PreprocessConfig:
    """Configuration for FASTA/PDB to graph preprocessing."""

    output_dir: Path
    reference_alignment: Path
    reference_alignment_input: Path | None = None
    mafft_executable: str = "mafft"
    feature_maker_binary: Path = Path("feature_maker/interface2grid")
    chem_lib_path: Path | None = None
    amino_mapping_path: Path = Path("feature_maker/add_amino_acid_features/amino_mapping")
    aligned_positions: tuple[int, ...] = DEFAULT_ALIGNED_POSITIONS
    reference_sequence_id: str | None = None
    reference_sequence_path: Path | None = None
    reference_residue_sites: tuple[int, ...] | None = None
    site_gap_strategy: str = "next"
    original_feature_length: int = 18

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any], base_dir: Path) -> "PreprocessConfig":
        preprocessing = _mapping_at(mapping, "preprocessing")
        source = preprocessing or mapping
        reference_value = _get_nested(source, ("reference_alignment", "sequences"))
        reference_alignment_input = resolve_path(source.get("reference_alignment_input"), base_dir)
        output_value = _get_nested(source, ("output_dir", "preprocessed_dir"), "runs/preprocess")
        feature_maker_binary = require_path(
            source.get(
                "feature_maker_binary",
                "../feature_maker/interface2grid"
                if "sequences" in source
                else "feature_maker/interface2grid",
            ),
            base_dir,
            "preprocessing.feature_maker_binary",
        )
        chem_lib_value = source.get("chem_lib_path")
        chem_lib_path = (
            require_path(chem_lib_value, base_dir, "preprocessing.chem_lib_path")
            if chem_lib_value
            else feature_maker_binary.parent / "chem.lib"
        )
        return cls(
            output_dir=require_path(output_value, base_dir, "preprocessing.output_dir"),
            reference_alignment=require_path(reference_value, base_dir, "preprocessing.reference_alignment"),
            reference_alignment_input=reference_alignment_input,
            mafft_executable=str(source.get("mafft_executable", "mafft")),
            feature_maker_binary=feature_maker_binary,
            chem_lib_path=chem_lib_path,
            amino_mapping_path=require_path(
                source.get(
                    "amino_mapping_path",
                    "../feature_maker/add_amino_acid_features/amino_mapping"
                    if "sequences" in source
                    else "feature_maker/add_amino_acid_features/amino_mapping",
                ),
                base_dir,
                "preprocessing.amino_mapping_path",
            ),
            aligned_positions=tuple(int(i) for i in source.get("aligned_positions", DEFAULT_ALIGNED_POSITIONS)),
            reference_sequence_id=source.get("reference_sequence_id"),
            reference_sequence_path=resolve_path(source.get("reference_sequence_path"), base_dir),
            reference_residue_sites=(
                tuple(int(i) for i in source["reference_residue_sites"])
                if source.get("reference_residue_sites")
                else None
            ),
            site_gap_strategy=str(source.get("site_gap_strategy", "next")),
            original_feature_length=int(source.get("original_feature_length", 18)),
        )


@dataclass(frozen=True)
class PredictionConfig:
    """Full prediction pipeline configuration."""

    model_path: Path
    means_path: Path
    stds_path: Path
    output_dir: Path
    records: tuple[OpsinRecord, ...]
    indexes_to_keep: tuple[int, ...] = tuple(range(36))
    graph_threshold: float = 2
    device: str = "auto"
    preprocessing: PreprocessConfig | None = None

    @classmethod
    def from_file(cls, path: str | Path) -> "PredictionConfig":
        mapping, base_dir = load_config_mapping(path)
        model = _mapping_at(mapping, "model")
        normalization = _mapping_at(mapping, "normalization")
        data = _mapping_at(mapping, "data")
        outputs = _mapping_at(mapping, "outputs")

        manifest_value = mapping.get("input_manifest") or data.get("input_manifest")
        raw_records: list[dict[str, Any]]
        if manifest_value:
            manifest_path = require_path(manifest_value, base_dir, "input_manifest")
            raw_records = load_opsin_manifest(manifest_path)
            record_base = manifest_path.parent
        else:
            raw_records = list(mapping.get("opsins", mapping.get("records", [])))
            record_base = base_dir

        records = tuple(
            OpsinRecord.from_mapping(record, record_base, index)
            for index, record in enumerate(raw_records)
        )
        if not records:
            raise ConfigError("Prediction config must define at least one opsin record or input_manifest.")

        preprocessing = None
        if "preprocessing" in mapping:
            preprocessing = PreprocessConfig.from_mapping(mapping, base_dir)

        return cls(
            model_path=require_path(model.get("model_path", mapping.get("model_path")), base_dir, "model.model_path"),
            means_path=require_path(
                normalization.get("means_path", mapping.get("means_path")), base_dir, "normalization.means_path"
            ),
            stds_path=require_path(
                normalization.get("stds_path", mapping.get("stds_path")), base_dir, "normalization.stds_path"
            ),
            output_dir=require_path(
                outputs.get("output_dir", mapping.get("output_dir", "runs/predict")),
                base_dir,
                "outputs.output_dir",
            ),
            records=records,
            indexes_to_keep=tuple(
                int(i) for i in data.get("indexes_to_keep", mapping.get("indexes_to_keep", range(36)))
            ),
            graph_threshold=float(
                data.get("graph_threshold", mapping.get("graph_th", mapping.get("graph_threshold", 2)))
            ),
            device=str(data.get("device", mapping.get("device", "auto"))),
            preprocessing=preprocessing,
        )


@dataclass(frozen=True)
class PreprocessRunConfig:
    """Standalone preprocessing configuration."""

    preprocessing: PreprocessConfig
    records: tuple[OpsinRecord, ...]

    @classmethod
    def from_file(cls, path: str | Path) -> "PreprocessRunConfig":
        mapping, base_dir = load_config_mapping(path)
        preprocessing = PreprocessConfig.from_mapping(mapping, base_dir)
        if "opsins" in mapping or "records" in mapping or "input_manifest" in mapping:
            manifest_value = mapping.get("input_manifest")
            if manifest_value:
                manifest_path = require_path(manifest_value, base_dir, "input_manifest")
                raw_records = load_opsin_manifest(manifest_path)
                record_base = manifest_path.parent
            else:
                raw_records = list(mapping.get("opsins", mapping.get("records", [])))
                record_base = base_dir
            records = tuple(
                OpsinRecord.from_mapping(record, record_base, index)
                for index, record in enumerate(raw_records)
            )
        else:
            cut_path = resolve_path(mapping.get("cutted_result_pdb_path"), base_dir)
            record_id = cut_path.stem if cut_path else "cutted_parts0"
            records = (
                OpsinRecord(
                    id=record_id,
                    fasta_path=require_path(mapping.get("fasta_path"), base_dir, "fasta_path"),
                    pdb_path=require_path(mapping.get("pdb_path"), base_dir, "pdb_path"),
                ),
            )
        if not records:
            raise ConfigError("Preprocessing config must define at least one opsin record.")
        return cls(preprocessing=preprocessing, records=records)
