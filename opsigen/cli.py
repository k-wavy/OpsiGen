"""Command-line entry points for OpsiGen."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import PredictionConfig, PreprocessRunConfig, TrainingConfig
from .dataset_prep import prepare_wds_dataset
from .exceptions import OpsiGenError
from .prediction import predict_from_config
from .preprocessing import preprocess_opsins
from .training import train_from_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="opsigen",
        description="Train OpsiGen models and run batch wavelength prediction.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Run the full training pipeline.")
    train_parser.add_argument("--config", required=True, type=Path, help="Training config JSON/YAML path.")

    predict_parser = subparsers.add_parser("predict", help="Run batch prediction.")
    predict_parser.add_argument("--config", required=True, type=Path, help="Prediction config JSON/YAML path.")

    preprocess_parser = subparsers.add_parser("preprocess", help="Run FASTA/PDB graph preprocessing only.")
    preprocess_parser.add_argument("--config", required=True, type=Path, help="Preprocessing config JSON/YAML path.")

    prepare_wds_parser = subparsers.add_parser("prepare-wds", help="Prepare WDS FASTA/meta files for OpsiGen.")
    prepare_wds_parser.add_argument("--meta", required=True, type=Path, help="WDS metadata TSV path.")
    prepare_wds_parser.add_argument("--fasta", required=True, type=Path, help="WDS FASTA path.")
    prepare_wds_parser.add_argument("--output-dir", default=Path("datasets/wds"), type=Path)
    prepare_wds_parser.add_argument("--configs-dir", default=Path("configs"), type=Path)
    prepare_wds_parser.add_argument("--pdb-dir", type=Path, default=None, help="Optional directory containing PDB files.")
    prepare_wds_parser.add_argument("--train-fraction", type=float, default=0.8)
    prepare_wds_parser.add_argument("--seed", type=int, default=7)
    prepare_wds_parser.add_argument("--reference-sequence-id", default="Bovine")
    prepare_wds_parser.add_argument(
        "--reference-alignment",
        type=Path,
        default=None,
        help=(
            "Optional precomputed animal-opsin MSA. If omitted, preprocessing builds "
            "datasets/wds/reference/wds_reference_alignment.fasta from the WDS FASTA."
        ),
    )
    prepare_wds_parser.add_argument(
        "--reference-residue-sites",
        default=None,
        help="Comma-separated 1-based reference residue sites, e.g. 83,122,292,299,300.",
    )

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "train":
            result = train_from_config(args.config, verbose=args.verbose)
            print(f"Training complete: {result.output_dir}")
            print(f"Final checkpoint: {result.final_checkpoint_path}")
            if result.best_checkpoint_path:
                print(f"Best checkpoint: {result.best_checkpoint_path}")
            return

        if args.command == "predict":
            result = predict_from_config(args.config, verbose=args.verbose)
            print(f"Prediction complete: {result.predictions_path}")
            return

        if args.command == "preprocess":
            config = PreprocessRunConfig.from_file(args.config)
            records = preprocess_opsins(config.records, config.preprocessing)
            print(f"Preprocessing complete: {config.preprocessing.output_dir}")
            print(f"Processed records: {len(records)}")
            return

        if args.command == "prepare-wds":
            result = prepare_wds_dataset(
                meta_path=args.meta,
                fasta_path=args.fasta,
                output_dir=args.output_dir,
                configs_dir=args.configs_dir,
                pdb_dir=args.pdb_dir,
                train_fraction=args.train_fraction,
                seed=args.seed,
                reference_sequence_id=args.reference_sequence_id,
                reference_alignment=args.reference_alignment,
                reference_residue_sites=(
                    [int(value) for value in args.reference_residue_sites.split(",") if value.strip()]
                    if args.reference_residue_sites
                    else None
                ),
            )
            print(f"WDS dataset prepared: {result.output_dir}")
            print(f"Training table: {result.training_excel}")
            print(f"Preprocess config: {result.preprocess_config}")
            print(f"Training config: {result.train_config}")
            print(f"Prediction config: {result.predict_config}")
            return

    except OpsiGenError as exc:
        print(f"opsigen: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    parser.error(f"Unknown command: {args.command}")


__all__ = ["build_parser", "main", "PredictionConfig", "PreprocessRunConfig", "TrainingConfig"]
