# WDS Dataset Preparation

The WDS preparation command converts a sequence FASTA and lambda-max metadata TSV into OpsiGen-ready inputs.

Use an unaligned FASTA for `--fasta`. The WDS/bovine-reference workflow uses that FASTA to build a shared animal-opsin multiple sequence alignment, then maps bovine-numbered residue sites through alignment columns.

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
- `datasets/wds/reference/wds_reference_alignment.fasta`: shared animal-opsin MSA built by preprocessing if it does not already exist.
- `datasets/wds/preprocess_manifest.csv`: preprocessing manifest with `id`, `fasta_path`, `pdb_path`, and optional `plddt_json_path`.
- `datasets/wds/splits/train_all` and `datasets/wds/splits/test_all`: deterministic 80/20 split files.
- `datasets/wds/validation_report.csv`: sequence-length, non-standard residue, PDB, and pLDDT sidecar checks.
- `configs/preprocess.wds.json`: preprocessing config.
- `configs/train.wds.json`: training config.
- `configs/train_functional.wds.json`: functional/non-functional classifier training config.
- `configs/predict.wds.example.json`: prediction config template for a trained WDS model.

The command ignores `.json` pLDDT sidecars for preprocessing/training. If a JSON sidecar matches the sequence ID, its path is recorded only for traceability.

Run order:

```bash
python -m opsigen preprocess --config configs/preprocess.wds.json
python -m opsigen train --config configs/train.wds.json
python -m opsigen train-functional --config configs/train_functional.wds.json
python -m opsigen predict --config configs/predict.wds.example.json
```

If `preprocess_manifest.csv` has empty `pdb_path` values, rerun `prepare-wds` with `--pdb-dir` before preprocessing.

## Bovine Reference Numbering

The WDS configs use bovine rhodopsin numbering for residue-site selection:

```json
"reference_sequence_id": "Bovine",
"reference_sequence_path": "../datasets/wds/fastas/Bovine.fasta",
"reference_alignment": "../datasets/wds/reference/wds_reference_alignment.fasta",
"reference_alignment_input": "../datasets/wds/source/wds.fasta",
"reference_residue_sites": [46, 49, 52, 83, 86, 90, 93, 118, 122, 124, 132, 180, 197, 230, 233, 277, 285, 292, 298, 299, 300, 308],
"site_gap_strategy": "next"
```

The pipeline treats those sites as 1-based residue numbers in the ungapped bovine sequence. It finds the corresponding columns in `reference_alignment`, maps each query through the same MSA, then cuts corresponding residues from each query PDB. If `reference_alignment` does not exist yet, preprocessing builds it with MAFFT from `reference_alignment_input`.

`reference_residue_sites` controls the graph nodes. Do not copy these residue numbers into `data.indexes_to_keep` in the training config. `indexes_to_keep` selects feature columns from each node's feature matrix; the default `0..35` keeps all generated feature columns. `model.number_features` is also a feature width, not a residue count. With the default `GAT21Model`, changing the number of selected residue sites changes the number of graph nodes, not the model input feature count.

Useful site sets from Hagen et al. 2023:

- Figure 3 broad vertebrate sites: `46,49,52,83,86,90,93,118,122,124,132,180,197,230,233,277,285,292,298,299,300,308`
- Rh1 example sites: `83,122,292,299,300`
- SWS1 example sites: `86,90,93,118`
- SWS1 mammalian sites discussed in text: `46,49,50,52,86,90,93,114,118`
- LWS/MWS five-site rule: `180,197,277,285,308`

To use a different set, edit `reference_residue_sites` in `configs/preprocess.wds.json` and `configs/predict.wds.example.json`, or regenerate configs:

```bash
python -m opsigen prepare-wds \
  --meta /path/to/wds_meta.tsv \
  --fasta /path/to/wds.fasta \
  --pdb-dir /path/to/pdb_folder \
  --reference-residue-sites 83,122,292,299,300
```

## Functional Classifier

The wavelength regressor skips `lmax = 0` rows, because `0 nm` is a non-functional label rather than a physical absorption peak. To learn that signal, train the binary classifier:

```bash
python -m opsigen train-functional --config configs/train_functional.wds.json
```

By default, `classifier.positive_class` is `nonfunctional`, so the reported `precision`, `recall`, `f1`, `auroc`, and `average_precision` are for non-functional opsins. This is intentional when non-functional examples are rare. `training.weighted_sampler` oversamples the minority class, and `classifier.class_weighted_loss` upweights minority-class errors in the binary cross-entropy loss.
