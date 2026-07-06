"""Deprecated compatibility entry point for OpsiGen training.

Prefer:

    python -m opsigen train --config configs/train.example.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from opsigen.training import train_from_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Deprecated wrapper around opsigen train.")
    parser.add_argument("config_path", type=Path)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    result = train_from_config(args.config_path, verbose=args.verbose)
    print(f"Training complete: {result.output_dir}")
    print(f"Final checkpoint: {result.final_checkpoint_path}")


if __name__ == "__main__":
    main()
