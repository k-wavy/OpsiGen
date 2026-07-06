# WDS Dataset Preparation

The WDS preparation command converts a sequence FASTA and lambda-max metadata TSV into OpsiGen-ready inputs.

```bash
python -m opsigen prepare-wds \
  --meta /path/to/wds_meta.tsv \
  --fasta /path/to/wds.fasta \
  --pdb-dir /path/to/pdb_folder \
  --output-dir datasets/wds \
  --configs-dir configs
```

Outputs:

- `datasets/wds/wds_training.xlsx`: training table with `Name`, `Wildtype`, `Sequence`, `lmax`, graph paths, and original metadata.
- `datasets/wds/wds_training.csv`: CSV copy of the same table.
- `datasets/wds/fastas/`: one FASTA per sequence ID for MAFFT preprocessing.
- `datasets/wds/preprocess_manifest.csv`: preprocessing manifest with `id`, `fasta_path`, `pdb_path`, and optional `plddt_json_path`.
- `datasets/wds/splits/train_all` and `datasets/wds/splits/test_all`: deterministic 80/20 split files.
- `datasets/wds/validation_report.csv`: sequence-length, non-standard residue, PDB, and pLDDT sidecar checks.
- `configs/preprocess.wds.json`: preprocessing config.
- `configs/train.wds.json`: training config.
- `configs/predict.wds.example.json`: prediction config template for a trained WDS model.

The command ignores `.json` pLDDT sidecars for preprocessing/training. If a JSON sidecar matches the sequence ID, its path is recorded only for traceability.

Run order:

```bash
python -m opsigen preprocess --config configs/preprocess.wds.json
python -m opsigen train --config configs/train.wds.json
python -m opsigen predict --config configs/predict.wds.example.json
```

If `preprocess_manifest.csv` has empty `pdb_path` values, rerun `prepare-wds` with `--pdb-dir` before preprocessing.
