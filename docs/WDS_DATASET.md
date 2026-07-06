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
- `configs/predict.wds.example.json`: prediction config template for a trained WDS model.

The command ignores `.json` pLDDT sidecars for preprocessing/training. If a JSON sidecar matches the sequence ID, its path is recorded only for traceability.

Run order:

```bash
python -m opsigen preprocess --config configs/preprocess.wds.json
python -m opsigen train --config configs/train.wds.json
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
