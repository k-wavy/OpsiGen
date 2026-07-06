# OpsiGen Refactor Migration Notes

## What Changed

The repository now has one importable Python package, `opsigen`, and one command surface:

```bash
python -m opsigen preprocess --config pipeline_auto/config.json
python -m opsigen train --config configs/train.example.json
python -m opsigen predict --config configs/predict.example.json
```

The old workflow required manually running `pipeline_auto/run.sh` and then `predict/run.sh`. Those steps are now represented by reusable modules and config-driven orchestration.

## Legacy Script Audit

| Legacy path | Previous purpose | New location or status |
| --- | --- | --- |
| `pipeline_auto/main.py` | Ran alignment, PDB cutting, feature generation, and distance generation through shell commands. | Compatibility wrapper around `opsigen preprocess`. |
| `pipeline_auto/run.sh` | Deleted fixed output folders and ran `pipeline_auto/main.py`. | Compatibility wrapper; no destructive cleanup. |
| `pipeline_auto/aligner_plus.py` | Added one FASTA to a MAFFT alignment and mapped conserved positions. | Replaced by `opsigen.preprocessing.pipeline`. |
| `pipeline_auto/cutter.py` | Cut the selected residues out of a PDB. | Replaced by `opsigen.preprocessing.pipeline.cut_pdb_to_residues`. |
| `pipeline_auto/edge_maker.py` | Wrote inverse-distance matrices for cut PDBs. | Replaced by `opsigen.preprocessing.pipeline.write_distance_matrix`. |
| `pipeline_auto/sequences.py`, `pipeline_auto/aligner.py`, `pipeline_auto/cut_24_amino_acids.py` | Older alignment/position experiments with hard-coded paths. | Removed. |
| `pipeline_auto/*feature*.sh`, `pipeline_auto/run_mafft.sh` | Shell wrappers around native feature generation, amino-acid feature joining, and MAFFT. | Removed; subprocess calls now live in Python with validation. |
| `predict/train.py` | Infinite training loop with hard-coded W&B key and local paths. | Compatibility wrapper around `opsigen.training.train_from_config`. |
| `predict/calculate_one_rhodopsin.py` | Single precomputed-graph prediction utility. | Compatibility wrapper around `opsigen.prediction.legacy_predict_one`. |
| `predict/models.py` | Model architecture definitions. | Compatibility re-export; canonical definitions live in `opsigen.models`. |
| `predict/pdb_dataset.py` | Dataset and normalization logic. | Replaced by `opsigen.data`. |
| `predict/test.py`, `predict/test_prev.py`, `predict/new_pdb_dataset.py`, `predict/main.py` | Scratch/test scripts, some with breakpoints or incomplete code. | Removed. |
| `predict/excel_parser.py`, `predict/pdb_parser.py` | Older helper utilities not used by the central pipeline. | Removed. |
| `feature_maker/add_amino_acid_features/*.py` | Python scripts that joined amino-acid descriptors to native atom features using relative/hard-coded paths. | Replaced by `opsigen.preprocessing.pipeline.append_amino_acid_features`; descriptor data files remain. |

## Behavior Preserved

- The same 24 aligned opsin positions are used by default.
- MAFFT still uses `--add` and `--keeplength`.
- The native `feature_maker/interface2grid` binary is still used for atom features.
- Amino-acid descriptors from `feature_maker/add_amino_acid_features/amino_mapping` are still appended to atom features.
- Distance matrices still use inverse pairwise atom distances with self-distances set to zero.
- The bundled legacy model pickle, means, and stds remain available under `predict/`.

## Behavior Changed

- Training now runs for a configured number of epochs instead of an unbounded `while True`.
- W&B is opt-in through config; no API key is set in source code.
- Generated preprocessing and prediction outputs are written to `runs/` rather than source directories.
- Batch prediction is configured through `opsins` records or a manifest instead of fixed filenames.
- Missing files, malformed configs, missing MAFFT, and missing native feature generator binaries raise explicit errors.

## Recommended Migration

1. Move local path edits out of scripts and into `configs/train.example.json` or `configs/predict.example.json`.
2. Replace old training calls with `python -m opsigen train --config <config>`.
3. Replace old prediction calls with `python -m opsigen predict --config <config>`.
4. For notebooks, import `TrainingConfig`, `PredictionConfig`, `train_model`, and `run_predictions` instead of running shell commands.
