"""Dataset preparation helpers for OpsiGen."""

from __future__ import annotations

import csv
import json
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from .exceptions import InputValidationError
from .preprocessing.pipeline import parse_fasta, sanitize_id

STANDARD_AMINO_ACIDS = set("ARNDCQEGHILKMFPSTWYV")
HAGEN_FIGURE3_VERTEBRATE_SITES = [
    46,
    49,
    52,
    83,
    86,
    90,
    93,
    118,
    122,
    124,
    132,
    180,
    197,
    230,
    233,
    277,
    285,
    292,
    298,
    299,
    300,
    308,
]


@dataclass(frozen=True)
class PreparedWDSDataset:
    """Paths produced by WDS dataset preparation."""

    output_dir: Path
    training_excel: Path
    training_csv: Path
    preprocess_manifest: Path
    predict_manifest_template: Path
    train_split: Path
    test_split: Path
    preprocess_config: Path
    train_config: Path
    predict_config: Path
    validation_report: Path


def _relative_path(path: Path, base: Path) -> str:
    return Path(os.path.relpath(path.resolve(), base.resolve())).as_posix()


def _relative_from_file(path: Path, file_path: Path) -> str:
    return _relative_path(path, file_path.parent)


def _write_fasta(path: Path, record_id: str, sequence: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    wrapped = "\n".join(sequence[i : i + 80] for i in range(0, len(sequence), 80))
    path.write_text(f">{record_id}\n{wrapped}\n")


def _build_file_index(directory: Path, suffix: str) -> dict[str, Path]:
    exact: dict[str, Path] = {}
    if not directory.exists():
        raise InputValidationError(f"Directory does not exist: {directory}")
    suffix = suffix.lower()
    for path in directory.rglob("*"):
        if path.is_file() and path.suffix.lower() == suffix:
            exact.setdefault(path.stem.lower(), path.resolve())
    return exact


def _find_by_id(index: dict[str, Path], record_id: str) -> Path | None:
    key = record_id.lower()
    if key in index:
        return index[key]
    prefix_matches = [
        path
        for stem, path in index.items()
        if stem.startswith(f"{key}_") or stem.startswith(f"{key}-") or stem.startswith(f"{key}.")
    ]
    if len(prefix_matches) == 1:
        return prefix_matches[0]
    return None


def _write_lines(path: Path, values: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(values) + "\n")


def prepare_wds_dataset(
    *,
    meta_path: str | Path,
    fasta_path: str | Path,
    output_dir: str | Path,
    configs_dir: str | Path,
    pdb_dir: str | Path | None = None,
    reference_alignment: str | Path | None = None,
    feature_maker_binary: str | Path = "feature_maker/interface2grid",
    chem_lib_path: str | Path = "feature_maker/chem.lib",
    amino_mapping_path: str | Path = "feature_maker/add_amino_acid_features/amino_mapping",
    reference_sequence_id: str = "Bovine",
    reference_residue_sites: list[int] | None = None,
    train_fraction: float = 0.8,
    seed: int = 7,
) -> PreparedWDSDataset:
    """Prepare WDS metadata/FASTA for OpsiGen preprocessing and training."""

    meta_path = Path(meta_path).expanduser().resolve()
    fasta_path = Path(fasta_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    configs_dir = Path(configs_dir).expanduser().resolve()
    repo_root = configs_dir.parent

    if not meta_path.exists():
        raise InputValidationError(f"Metadata TSV does not exist: {meta_path}")
    if not fasta_path.exists():
        raise InputValidationError(f"FASTA file does not exist: {fasta_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    configs_dir.mkdir(parents=True, exist_ok=True)
    fastas_dir = output_dir / "fastas"
    splits_dir = output_dir / "splits"
    copied_dir = output_dir / "source"
    reference_dir = output_dir / "reference"
    copied_dir.mkdir(parents=True, exist_ok=True)
    reference_dir.mkdir(parents=True, exist_ok=True)
    if reference_residue_sites is None:
        reference_residue_sites = HAGEN_FIGURE3_VERTEBRATE_SITES
    if reference_alignment is None:
        reference_alignment_path = reference_dir / "wds_reference_alignment.fasta"
        reference_alignment_input_path = copied_dir / fasta_path.name
    else:
        reference_alignment_path = (repo_root / reference_alignment).resolve()
        reference_alignment_input_path = copied_dir / fasta_path.name

    meta = pd.read_csv(meta_path, sep="\t")
    required_columns = {"Seq_Id", "Lambda_Max", "Species", "Opsin_Family", "Accession"}
    missing = required_columns - set(meta.columns)
    if missing:
        raise InputValidationError(f"Metadata TSV is missing required columns: {sorted(missing)}")

    fasta_records = {record.name: record.sequence for record in parse_fasta(fasta_path)}
    missing_sequences = sorted(set(meta["Seq_Id"]) - set(fasta_records))
    extra_sequences = sorted(set(fasta_records) - set(meta["Seq_Id"]))
    if missing_sequences:
        raise InputValidationError(f"{len(missing_sequences)} metadata IDs are missing from FASTA.")

    pdb_index: dict[str, Path] = {}
    json_index: dict[str, Path] = {}
    if pdb_dir:
        pdb_dir_path = Path(pdb_dir).expanduser().resolve()
        pdb_index = _build_file_index(pdb_dir_path, ".pdb")
        json_index = _build_file_index(pdb_dir_path, ".json")

    rows: list[dict[str, object]] = []
    manifest_rows: list[dict[str, str]] = []
    validation_rows: list[dict[str, object]] = []
    graph_base = repo_root / "runs" / "preprocess" / "wds"
    features_dir = graph_base / "features"
    dists_dir = graph_base / "dists"

    for _, row in meta.iterrows():
        seq_id = str(row["Seq_Id"])
        graph_id = sanitize_id(seq_id)
        sequence = str(fasta_records[seq_id]).strip().upper()
        fasta_out = fastas_dir / f"{graph_id}.fasta"
        _write_fasta(fasta_out, seq_id, sequence)

        non_standard = sorted({aa for aa in sequence if aa not in STANDARD_AMINO_ACIDS})
        pdb_path = _find_by_id(pdb_index, seq_id) if pdb_index else None
        plddt_json_path = _find_by_id(json_index, seq_id) if json_index else None

        features_path = features_dir / f"{graph_id}.npz"
        dists_path = dists_dir / f"{graph_id}_dists.npy"
        wildtype = str(row["Accession"]) if str(row["Accession"]).strip() else str(row["Species"])

        rows.append(
            {
                "Name": seq_id,
                "Graph_Id": graph_id,
                "Wildtype": wildtype,
                "Sequence": sequence,
                "lmax": float(row["Lambda_Max"]),
                "Method": "WDS",
                "Species": row.get("Species", ""),
                "Opsin_Family": row.get("Opsin_Family", ""),
                "Phylum": row.get("Phylum", ""),
                "Class": row.get("Class", ""),
                "Accession": row.get("Accession", ""),
                "Mutations": "" if pd.isna(row.get("Mutations", "")) else row.get("Mutations", ""),
                "RefId": "" if pd.isna(row.get("RefId", "")) else row.get("RefId", ""),
                "fasta_path": _relative_path(fasta_out, output_dir),
                "pdb_path": "" if pdb_path is None else pdb_path.as_posix(),
                "plddt_json_path": "" if plddt_json_path is None else plddt_json_path.as_posix(),
                "features_path": _relative_path(features_path, output_dir),
                "dists_path": _relative_path(dists_path, output_dir),
            }
        )
        manifest_rows.append(
            {
                "id": graph_id,
                "fasta_path": _relative_path(fasta_out, output_dir),
                "pdb_path": "" if pdb_path is None else pdb_path.as_posix(),
                "plddt_json_path": "" if plddt_json_path is None else plddt_json_path.as_posix(),
            }
        )
        validation_rows.append(
            {
                "id": seq_id,
                "sequence_length": len(sequence),
                "non_standard_residues": "".join(non_standard),
                "has_pdb": pdb_path is not None,
                "has_plddt_json": plddt_json_path is not None,
            }
        )

    dataset = pd.DataFrame(rows)
    validation = pd.DataFrame(validation_rows)

    training_csv = output_dir / "wds_training.csv"
    training_excel = output_dir / "wds_training.xlsx"
    validation_report = output_dir / "validation_report.csv"
    preprocess_manifest = output_dir / "preprocess_manifest.csv"
    predict_manifest_template = output_dir / "predict_manifest.template.csv"

    dataset.to_csv(training_csv, index=False)
    try:
        dataset.to_excel(training_excel, index=False)
    except ImportError:
        training_excel = training_csv
    validation.to_csv(validation_report, index=False)
    pd.DataFrame(manifest_rows).to_csv(preprocess_manifest, index=False)
    pd.DataFrame(
        [
            {
                "id": "new_opsin_id",
                "fasta_path": "path/to/new_opsin.fasta",
                "pdb_path": "path/to/new_opsin.pdb",
            }
        ]
    ).to_csv(predict_manifest_template, index=False)

    shutil.copy2(meta_path, copied_dir / meta_path.name)
    shutil.copy2(fasta_path, copied_dir / fasta_path.name)

    wildtypes = sorted(dataset["Wildtype"].astype(str).unique())
    random.Random(seed).shuffle(wildtypes)
    split_index = max(1, min(len(wildtypes) - 1, int(len(wildtypes) * train_fraction)))
    train_wildtypes = sorted(wildtypes[:split_index])
    test_wildtypes = sorted(wildtypes[split_index:])
    train_split = splits_dir / "train_all"
    test_split = splits_dir / "test_all"
    _write_lines(train_split, train_wildtypes)
    _write_lines(test_split, test_wildtypes)

    preprocess_config = configs_dir / "preprocess.wds.json"
    train_config = configs_dir / "train.wds.json"
    predict_config = configs_dir / "predict.wds.example.json"

    preprocess_config.write_text(
        json.dumps(
            {
                "input_manifest": _relative_from_file(preprocess_manifest, preprocess_config),
                "preprocessing": {
                    "reference_alignment": _relative_path(reference_alignment_path, configs_dir),
                    "reference_alignment_input": _relative_path(reference_alignment_input_path, configs_dir),
                    "reference_sequence_id": reference_sequence_id,
                    "reference_sequence_path": _relative_path(
                        (fastas_dir / f"{sanitize_id(reference_sequence_id)}.fasta").resolve(),
                        configs_dir,
                    ),
                    "reference_residue_sites": reference_residue_sites,
                    "site_gap_strategy": "next",
                    "output_dir": "../runs/preprocess/wds",
                    "mafft_executable": "mafft",
                    "feature_maker_binary": _relative_path((repo_root / feature_maker_binary).resolve(), configs_dir),
                    "chem_lib_path": _relative_path((repo_root / chem_lib_path).resolve(), configs_dir),
                    "amino_mapping_path": _relative_path((repo_root / amino_mapping_path).resolve(), configs_dir),
                    "original_feature_length": 18,
                },
            },
            indent=2,
        )
        + "\n"
    )

    train_config.write_text(
        json.dumps(
            {
                "run_name": "opsigen-wds-training",
                "data": {
                    "excel_path": _relative_from_file(training_excel, train_config),
                    "graph_features_path": "../runs/preprocess/wds/features",
                    "graph_dists_path": "../runs/preprocess/wds/dists",
                    "features_column": "features_path",
                    "dists_column": "dists_path",
                    "id_column": "Name",
                    "target_column": "lmax",
                    "wildtype_column": "Wildtype",
                    "train_wildtypes_list": _relative_from_file(train_split, train_config),
                    "test_wildtypes_list": _relative_from_file(test_split, train_config),
                    "indexes_to_keep": list(range(36)),
                    "dataset_normalize_last": True,
                    "drop_last_row": False,
                },
                "model": {
                    "name": "GAT21Model",
                    "number_features": 34,
                    "hidden_layer_size": 40,
                    "out_layer_size": 30,
                    "graph_threshold": 2,
                    "dropout": 0.1,
                },
                "training": {
                    "epochs": 100,
                    "learning_rate": 0.0001,
                    "weight_decay": 0.00001,
                    "l1_lambda": 0.0001,
                    "batch_size": 1,
                    "weighted_sampler": True,
                    "sampler_samples": int(len(dataset)),
                    "test_goal": 9,
                    "seed": seed,
                    "device": "auto",
                    "use_wandb": False,
                    "checkpoint_name": "opsigen_wds_model.pkl",
                },
                "outputs": {
                    "output_dir": "../runs/train/wds",
                },
            },
            indent=2,
        )
        + "\n"
    )

    predict_config.write_text(
        json.dumps(
            {
                "model": {
                    "model_path": "../runs/train/wds/checkpoints/opsigen_wds_model.pkl",
                },
                "normalization": {
                    "means_path": "../runs/train/wds/artifacts/means.npy",
                    "stds_path": "../runs/train/wds/artifacts/stds.npy",
                },
                "data": {
                    "indexes_to_keep": list(range(36)),
                    "graph_threshold": 2,
                    "device": "auto",
                },
                "preprocessing": {
                    "reference_alignment": _relative_path(reference_alignment_path, configs_dir),
                    "reference_alignment_input": _relative_path(reference_alignment_input_path, configs_dir),
                    "reference_sequence_id": reference_sequence_id,
                    "reference_sequence_path": _relative_path(
                        (fastas_dir / f"{sanitize_id(reference_sequence_id)}.fasta").resolve(),
                        configs_dir,
                    ),
                    "reference_residue_sites": reference_residue_sites,
                    "site_gap_strategy": "next",
                    "output_dir": "../runs/predict/wds/preprocessed",
                    "mafft_executable": "mafft",
                    "feature_maker_binary": _relative_path((repo_root / feature_maker_binary).resolve(), configs_dir),
                    "chem_lib_path": _relative_path((repo_root / chem_lib_path).resolve(), configs_dir),
                    "amino_mapping_path": _relative_path((repo_root / amino_mapping_path).resolve(), configs_dir),
                    "original_feature_length": 18,
                },
                "input_manifest": _relative_from_file(predict_manifest_template, predict_config),
                "outputs": {
                    "output_dir": "../runs/predict/wds",
                },
            },
            indent=2,
        )
        + "\n"
    )

    return PreparedWDSDataset(
        output_dir=output_dir,
        training_excel=training_excel,
        training_csv=training_csv,
        preprocess_manifest=preprocess_manifest,
        predict_manifest_template=predict_manifest_template,
        train_split=train_split,
        test_split=test_split,
        preprocess_config=preprocess_config,
        train_config=train_config,
        predict_config=predict_config,
        validation_report=validation_report,
    )
