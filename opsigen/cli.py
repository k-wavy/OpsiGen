"""Command-line entry points for OpsiGen."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import PredictionConfig, PreprocessRunConfig, TrainingConfig
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

    except OpsiGenError as exc:
        print(f"opsigen: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    parser.error(f"Unknown command: {args.command}")


__all__ = ["build_parser", "main", "PredictionConfig", "PreprocessRunConfig", "TrainingConfig"]
