"""Deprecated compatibility entry point for single-graph prediction.

Prefer:

    python -m opsigen predict --config configs/predict.example.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from opsigen.prediction import legacy_predict_one


def main() -> None:
    parser = argparse.ArgumentParser(description="Deprecated wrapper around opsigen prediction.")
    parser.add_argument("config_path", type=Path)
    parser.add_argument("pickle_file", type=Path)
    parser.add_argument("output_file", type=Path)
    parser.add_argument("features_file", type=Path)
    parser.add_argument("dists_file", type=Path)
    args = parser.parse_args()
    prediction = legacy_predict_one(
        config_path=args.config_path,
        model_path=args.pickle_file,
        output_file=args.output_file,
        features_path=args.features_file,
        dists_path=args.dists_file,
    )
    print(f"absorption wavelength is {prediction}")


if __name__ == "__main__":
    main()
