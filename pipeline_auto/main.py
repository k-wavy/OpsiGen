"""Deprecated compatibility entry point for graph preprocessing.

Prefer:

    python -m opsigen preprocess --config configs/predict.example.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from opsigen.config import PreprocessRunConfig
from opsigen.preprocessing import preprocess_opsins


def main() -> None:
    parser = argparse.ArgumentParser(description="Deprecated wrapper around opsigen preprocess.")
    parser.add_argument("config_file", type=Path)
    args = parser.parse_args()
    config = PreprocessRunConfig.from_file(args.config_file)
    records = preprocess_opsins(config.records, config.preprocessing)
    print(f"Preprocessing complete: {config.preprocessing.output_dir}")
    print(f"Processed records: {len(records)}")


if __name__ == "__main__":
    main()
