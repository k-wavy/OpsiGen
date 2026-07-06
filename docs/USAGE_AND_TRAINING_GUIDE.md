# OpsiGen Usage And Training Guide

This guide documents the current OpsiGen/RhoMax repository as a reusable
platform for predicting rhodopsin absorption maxima and for training new
structure-aware graph neural network models.

The short version: OpsiGen is not a sequence-only model. For each sequence, it
needs a matching protein structure, usually generated with AlphaFold2 or
ColabFold. The pipeline aligns the sequence to a reference microbial rhodopsin
alignment, cuts the 24 retinal-binding-pocket residues, converts the atoms in
those residues into a graph, normalizes node features with training-set
statistics, and runs a PyTorch Geometric model to predict lambda max in nm.

## Contents

1. [Repository Status](#repository-status)
2. [Conceptual Workflow](#conceptual-workflow)
3. [Repository Layout](#repository-layout)
4. [Environment Setup](#environment-setup)
5. [Using An Existing Model](#using-an-existing-model)
6. [Prediction Input Contracts](#prediction-input-contracts)
7. [Prediction Output Contracts](#prediction-output-contracts)
8. [Training New Models](#training-new-models)
9. [Data Preparation In Detail](#data-preparation-in-detail)
10. [Feature Generation In Detail](#feature-generation-in-detail)
11. [Training Configuration Reference](#training-configuration-reference)
12. [Evaluation And Reproducibility](#evaluation-and-reproducibility)
13. [Publishing A New Reusable Model](#publishing-a-new-reusable-model)
14. [Current Limitations And Modernization Notes](#current-limitations-and-modernization-notes)
15. [Troubleshooting](#troubleshooting)

## Repository Status

This repository contains the original RhoMax/OpsiGen research code plus a first
modernization pass.

What is usable now:

- Existing paper/model context is preserved in
  `rhomax-computational-prediction-of-rhodopsin-absorption-maxima-using-geometric-deep-learning.pdf`.
- The sample single-sequence pipeline is present in `pipeline_auto/`.
- A legacy trained model artifact is present at `predict/model_pickel`.
- Normalization arrays for the legacy model are present at `predict/means.npy`
  and `predict/stds.npy`.
- Model definitions are present in `predict/models.py`, including the published
  `GAT21Model` architecture used by `predict/conf_all`.
- Training now uses finite, config-driven epochs and no longer contains a
  committed Weights & Biases API key.

What still needs care:

- Full training feature arrays are not fully committed in this checkout. The
  repo contains `pipeline_auto/features/cutted_parts0.npz` and
  `pipeline_auto/dists/cutted_parts0_dists.npy` as sample artifacts, but a full
  retraining run needs one feature file and one distance file for every
  phenotype-table row.
- `feature_maker/interface2grid` is a Linux ELF executable. On macOS, the
  precomputed sample arrays can be inspected, but feature generation needs
  Linux, WSL, Docker, or a rebuilt/replaced feature-maker binary.
- Several scripts are legacy/research scratch scripts with hard-coded lab paths.
  The recommended prediction and training paths are documented below.
- The project is not yet packaged as an installable Python module. Run commands
  from the directories shown in this guide.

## Conceptual Workflow

OpsiGen follows the method described in the RhoMax paper:

1. Start with an opsin amino-acid sequence.
2. Generate or provide a matching protein structure.
3. Align the sequence to a reference microbial rhodopsin multiple sequence
   alignment.
4. Map the 24 retinal-binding-pocket alignment positions to sequence positions.
5. Cut the matching residues from the structure.
6. Represent all atoms in those residues as graph nodes.
7. Build graph edges from inverse interatomic distance.
8. Attach a 36-feature node vector:
   - 18 amino-acid physicochemical features from the Inoue/Karasuyama feature
     set.
   - 15 Tripos 5.2 atom-type one-hot features.
   - 3 atom-level features: partial atomic charge, solvent accessible area, and
     atomic radius.
9. Normalize node features using means and standard deviations computed on the
   training split.
10. Run a graph neural network to predict lambda max in nm.

The paper reports training on 884 microbial rhodopsin records derived from 75
wildtypes. Wildtype-aware train/test splits are critical because most mutants
differ by only one or two residues from their wildtype. Keeping a wildtype and
its mutants in the same split avoids leakage.

## Repository Layout

```text
OpsiGen/
|-- README.md
|-- docs/
|   `-- USAGE_AND_TRAINING_GUIDE.md
|-- rhomax-computational-prediction-of-rhodopsin-absorption-maxima-using-geometric-deep-learning.pdf
|-- excel/
|   |-- data.xlsx
|   |-- sequences.fas
|   |-- wavelength.dat
|   `-- splits/
|       |-- train0 ... train4
|       |-- test0 ... test4
|       |-- train_all
|       `-- test_all
|-- pipeline_auto/
|   |-- config.json
|   |-- sample_fasta.fasta
|   |-- sample_pdb.pdb
|   |-- main.py
|   |-- aligner_plus.py
|   |-- cutter.py
|   |-- edge_maker.py
|   |-- feature_maker.sh
|   |-- add_amino_acid_features.sh
|   |-- features/
|   |-- dists/
|   `-- cutted_parts/
|-- feature_maker/
|   |-- interface2grid
|   |-- chem.lib
|   `-- add_amino_acid_features/
|       |-- amino_acid_feature.csv
|       |-- amino_mapping
|       |-- main.py
|       `-- protein.py
`-- predict/
    |-- conf_all
    |-- model_pickel
    |-- means.npy
    |-- stds.npy
    |-- calculate_one_rhodopsin.py
    |-- train.py
    |-- pdb_dataset.py
    `-- models.py
```

Important path conventions:

- `pipeline_auto/` is the single-query feature-generation workspace.
- `predict/` is the model training and inference workspace.
- `excel/data.xlsx` is the main phenotype table used by `PDBDataset`.
- `excel/sequences.fas` and `excel/wavelength.dat` are used by alignment and
  historical preprocessing code.
- `excel/splits/train0` through `train4` and `excel/splits/test0` through
  `test4` are the wildtype-held-out split definitions.
- `excel/splits/train_all` and `excel/splits/test_all` contain the same 75
  wildtype names. Use them only when intentionally fitting/evaluating on all
  wildtypes, for example to train a final production model after evaluation.

## Environment Setup

### Recommended Platform

Use Linux for full feature generation and model training. macOS can run many
Python-only inspection tasks, but the committed `feature_maker/interface2grid`
binary is Linux-only.

Recommended options:

- Native Linux workstation or server.
- Linux GPU node.
- Docker/Podman Linux container.
- WSL2 on Windows.

### Python Environment

The legacy requirements pin PyTorch 1.13 and PyTorch Geometric CUDA 11.7 wheels.
Those versions are old but match the committed model era. For faithful
reproduction, start with Python 3.9 or 3.10 on Linux.

```bash
conda create -n opsigen python=3.9
conda activate opsigen
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PyTorch Geometric packages fail to install from `requirements.txt`, install
the matching wheel set manually for your PyTorch and CUDA/CPU target. For CPU
only, follow the official PyTorch Geometric installation selector and pin
versions close to the repository:

- `torch==1.13.0`
- `torch-geometric==2.2.0`
- `torch-scatter==2.1.0`
- `torch-sparse==0.6.16`
- `torch-cluster==1.6.0`
- `torch-spline-conv==1.2.1`

### External Tools

Install MAFFT:

```bash
conda install -c bioconda mafft
```

Confirm it is available:

```bash
mafft --version
```

For full structure generation, install ColabFold/AlphaFold2 separately. OpsiGen
does not currently run AlphaFold2 itself. It assumes you provide PDB files.

### Feature Maker

`pipeline_auto/feature_maker.sh` calls:

```bash
../feature_maker/interface2grid -i INPUT_PDB_DIR -o OUTPUT_FEATURE_DIR
```

The committed executable is:

```bash
file feature_maker/interface2grid
```

Expected current result:

```text
ELF 64-bit LSB executable, x86-64, statically linked
```

On non-Linux platforms, rebuild the feature maker or run inside Linux.

## Using An Existing Model

This section describes the current local single-query path using the committed
legacy model.

### Step 1 - Prepare An Input FASTA

Create a FASTA file with one amino-acid sequence:

```fasta
>my_opsin
MNGTEGPNFYVPFSNKTGVVRSPFEAPQYYLAEPWQFSMLAAYMFLLIMLGFPINFLT...
```

Rules:

- Use amino-acid sequence, not nucleotide sequence.
- Use standard one-letter amino-acid codes.
- Avoid gaps in the query FASTA.
- Keep exactly one record for the current `pipeline_auto` script path.
- The FASTA header becomes the sequence name used during alignment.

Place or copy it to:

```text
pipeline_auto/sample_fasta.fasta
```

### Step 2 - Prepare A Matching PDB

Generate a structure for the same sequence. The paper used AlphaFold2 through
ColabFold and selected the highest-ranked model.

Rules for the current cutter:

- PDB should contain the same protein sequence as the FASTA.
- The desired chain should be the first chain in the first model.
- Residues should be present in sequence order.
- Avoid missing residues in the 24 binding-pocket positions.
- Avoid PDBs where residue numbering does not match the sequence order unless
  the cutter has been updated to handle insertion codes and offsets.

Place or copy it to:

```text
pipeline_auto/sample_pdb.pdb
```

### Step 3 - Review `pipeline_auto/config.json`

Default single-query config:

```json
{
  "fasta_path": "./sample_fasta.fasta",
  "wavelength_file": "../excel/wavelength.dat",
  "pdb_path": "./sample_pdb.pdb",
  "aligning_index": -1,
  "sequences": "../excel/sequences.fas",
  "cutted_parts_dir": "./cutted_parts/",
  "features": "./features/",
  "cutted_result_pdb_path": "./cutted_parts/cutted_parts0.pdb",
  "feature_maker_script": "./feature_maker.sh",
  "amino_acid_feature_script": "./add_amino_acid_features.sh",
  "graph_maker_script": "./edge_maker.py",
  "edge_dists_path": "./dists/",
  "mafft_id": "mafft_id_0"
}
```

Most single-query users only need to change `fasta_path` and `pdb_path`.

Important fields:

- `aligning_index`: set to `-1` to search the reference alignment for the best
  match. Set a positive integer only when you know the exact reference index.
- `sequences`: reference aligned FASTA used for pocket-position mapping.
- `wavelength_file`: historical wavelength labels used to filter reference
  sequence indices.
- `cutted_result_pdb_path`: output PDB containing only the 24 pocket residues.
- `features`: output directory for atom and amino-acid feature arrays.
- `edge_dists_path`: output directory for inverse-distance matrices.
- `mafft_id`: output basename under `pipeline_auto/mafft_alignments/`.

### Step 4 - Generate Graph Features

From the repository root:

```bash
cd pipeline_auto
rm -rf dists features cutted_parts
mkdir -p dists features cutted_parts mafft_alignments
python ./main.py ./config.json
```

The legacy shortcut is:

```bash
cd pipeline_auto
./run.sh
```

The explicit command sequence is easier to debug and is recommended.

Expected outputs:

```text
pipeline_auto/cutted_parts/cutted_parts0.pdb
pipeline_auto/features/cutted_parts0.npz
pipeline_auto/dists/cutted_parts0_dists.npy
pipeline_auto/mafft_alignments/mafft_id_0.fasta
```

Check shapes:

```bash
python - <<'PY'
import numpy as np
features = np.load("features/cutted_parts0.npz")
dists = np.load("dists/cutted_parts0_dists.npy")
print("features", features.shape, features.dtype)
print("dists", dists.shape, dists.dtype)
PY
```

Expected pattern:

```text
features (N, 36) float64
dists (N, N) float64
```

`N` is the number of atoms in the 24 selected residues. The sample artifact has
209 atoms.

### Step 5 - Run Prediction

Use the direct Python command instead of the legacy SLURM wrapper:

```bash
cd ../predict
python calculate_one_rhodopsin.py \
  ./conf_all \
  ./model_pickel \
  ../pipeline_auto/result.txt \
  ../pipeline_auto/features/cutted_parts0.npz \
  ../pipeline_auto/dists/cutted_parts0_dists.npy
```

Expected output file:

```text
pipeline_auto/result.txt
```

Example content:

```text
absorption wavelength is 532.4
```

The exact value depends on the input structure, feature generation, installed
PyTorch/PyG versions, and model artifact.

### Step 6 - Interpret The Result

The model predicts lambda max in nm. Treat it as a model estimate, not a
replacement for spectroscopy.

Use additional caution when:

- The sequence is far outside the microbial rhodopsin families represented in
  the training set.
- The AlphaFold/ColabFold structure has low confidence in the binding pocket.
- The 24 pocket residues cannot be mapped cleanly.
- The predicted wavelength is outside the training distribution.
- The query is an animal/type II opsin. The current RhoMax data are microbial
  rhodopsin centered.

## Prediction Input Contracts

### FASTA Contract

The current single-query pipeline expects:

- One FASTA record.
- Header line starting with `>`.
- Amino-acid sequence on the following line or lines.
- No spaces inside the sequence.
- No alignment gaps in the query sequence.

The parser in `pipeline_auto/aligner_plus.py` currently reads:

```python
return lines[1], lines[0][1:]
```

That means multi-line FASTA input is not robust in the current implementation.
For now, keep the sequence on one line for single-query prediction.

### PDB Contract

The cutter in `pipeline_auto/cutter.py` currently assumes:

- The first model is the protein model.
- The first chain is the chain to cut.
- Residues are indexed by simple integer IDs.
- The sequence positions from alignment can be used directly against chain
  residue order.

This is acceptable for many simple AlphaFold/ColabFold PDBs, but it is not a
general PDB parser for all experimental structures.

### Reference Alignment Contract

The 24 pocket positions are hard-coded in `pipeline_auto/sequences.py`:

```python
POSITIONS = [
    429, 465, 469, 597, 599, 600, 603, 604,
    607, 651, 652, 655, 687, 690, 691, 694,
    774, 777, 778, 781, 828, 832, 835, 836,
]
```

These are positions in the reference alignment, not raw sequence positions.
MAFFT adds the query to `excel/sequences.fas`, then the pipeline translates
alignment positions back to ungapped query positions.

## Prediction Output Contracts

Feature generation output:

- `cutted_parts0.pdb`: PDB containing only the selected pocket residues.
- `cutted_parts0.npz`: NumPy array with node features, shape `(N, 36)`.
- `cutted_parts0_dists.npy`: NumPy array with inverse-distance edge weights,
  shape `(N, N)`.

Prediction output:

- Plain-text file containing one line:

```text
absorption wavelength is <float>
```

The current prediction function loads:

- A pickled PyTorch model.
- `means.npy`.
- `stds.npy`.
- One feature array.
- One distance array.

Normalization behavior:

- The feature matrix is first subset by `indexes_to_keep`.
- Columns with zero training-set standard deviation are dropped.
- Remaining columns are z-score normalized.
- With the current `predict/conf_all`, 36 raw columns become 34 model input
  columns because two columns have zero standard deviation.

## Training New Models

This section gives the reproducible training workflow for new data.

### Step 1 - Define The Model Version

Every training run should have a version name. Examples:

- `rhomax_2024_reproduction`
- `vpod_1.4_structure`
- `opsigen_lab_2026_07`

Use this version consistently in:

- Training config filename.
- Output checkpoint directory.
- Normalization artifact directory.
- Model registry or release notes.
- Evaluation reports.

### Step 2 - Prepare The Phenotype Table

Create or update an Excel file matching `excel/data.xlsx`.

Required columns:

- `Name`: unique sequence/variant name.
- `Wildtype`: parent wildtype label. All mutants of the same parent must share
  exactly the same string.
- `Sequence`: ungapped amino-acid sequence.
- `lmax`: experimentally measured lambda max in nm.
- `Method`: optional experimental method label.

Recommended audit columns:

- The 24 binding-pocket residue columns used by the original data table:
  `20`, `49`, `53`, `83`, `85`, `86`, `89`, `90`, `93`, `118`, `119`, `122`,
  `138`, `141`, `142`, `145`, `182`, `185`, `186`, `189`, `208`, `212`, `215`,
  `216`.

The current `PDBDataset` uses `Wildtype` and `lmax` directly, and it relies on
feature/distance filenames by row index. The other columns are still important
for curation, audit, compatibility with historical analyses, and later
explainability work.

### Step 3 - Validate Phenotype Data

Before feature generation, validate:

1. `Name` values are unique.
2. `Wildtype` is non-empty for every row.
3. `Sequence` contains only standard amino-acid letters unless you have a
   deliberate policy for non-standard residues.
4. `lmax` is numeric and in nm.
5. Mutants are labeled with the same `Wildtype` as their parent.
6. Replicate measurements have an explicit policy:
   - Keep each replicate as a separate row only if structures/features are also
     separate.
   - Otherwise average or select records before training.
7. Sequence and structure identifiers are traceable.

### Step 4 - Generate Structures

For each phenotype-table row:

1. Run AlphaFold2 or ColabFold on the exact `Sequence`.
2. Use the highest-ranked model unless your protocol says otherwise.
3. Save the PDB with a stable name tied to the row index and `Name`.
4. Record structure-generation metadata:
   - Tool name and version.
   - Database/version date if applicable.
   - Model rank used.
   - Mean pLDDT.
   - Pocket-region confidence summary if available.
   - Whether templates were used.

The RhoMax paper generated five structures for each sequence using AlphaFold2
through ColabFold with default parameters and without templates, then selected
the highest-ranked structure.

### Step 5 - Generate Graph Features For Every Row

For row index `i`, create:

```text
db_features/cutted_parts{i}.npz
db_dists/cutted_parts{i}_dists.npy
```

The training dataset constructs paths exactly this way:

```python
features = graph_features_path + "cutted_parts{}.npz".format(idx)
dists = graph_dists_path + "cutted_parts{}_dists.npy".format(idx)
```

A full training run needs arrays for every row that should be trainable. If a
row has missing arrays, the dataset skips it by returning target `0`.

### Step 6 - Create Wildtype-Aware Splits

Do not randomly split individual rows. Split by `Wildtype`.

Reason: most mutants are very similar to their wildtype. If mutants from the
same wildtype appear in both training and testing, evaluation is overly
optimistic.

For a paper-style split:

1. List all unique wildtypes.
2. Randomly assign 65 wildtypes to train and 10 wildtypes to test.
3. Write one wildtype name per line.
4. Repeat to create multiple folds.

Example split file:

```text
AR3
Chrimson
GtACR2
...
```

The existing repository has:

```text
excel/splits/train0
excel/splits/test0
...
excel/splits/train4
excel/splits/test4
```

Use these for evaluation. Use `train_all` only when intentionally fitting a
final model after evaluation.

### Step 7 - Configure Training

Use `predict/conf_all` as a template. For a new model, create a new config, for
example:

```bash
cp predict/conf_all predict/conf_my_dataset
```

Edit paths and metadata. A minimal config is:

```json
{
  "graph_features_path": "../db_features/",
  "graph_dists_path": "../db_dists/",
  "excel_path": "../excel/data.xlsx",
  "pickle_file": "./model_pickle",
  "train_wildtypes_list": "../excel/splits/train0",
  "test_wildtypes_list": "../excel/splits/test0",
  "lr": 0.0001,
  "hidden_layer_size": 40,
  "out_layer_size": 30,
  "graph_th": 2,
  "model_name": "GAT21Model",
  "number_features": 34,
  "dataset_normalize_last": true,
  "test_goal": 9,
  "epochs": 1000,
  "seed": 123,
  "num_samples": 884,
  "wandb": false,
  "checkpoint_dir": "./checkpoints",
  "means_path": "./means.npy",
  "stds_path": "./stds.npy",
  "indexes_to_keep": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                      10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                      20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
                      30, 31, 32, 33, 34, 35]
}
```

### Step 8 - Train

From the `predict/` directory:

```bash
cd predict
python train.py ./conf_my_dataset
```

Expected artifacts:

- `means.npy`: training-set feature means.
- `stds.npy`: training-set feature standard deviations.
- `checkpoints/<model>_best_epoch-....pkl`: best checkpoint so far by test MAE.

Expected console metrics:

```text
epoch=1 train_mae=... test_mae=... energy_loss=...
saved best checkpoint: ./checkpoints/model_pickle_best_epoch-0001_mae-...
```

### Step 9 - Evaluate Across Splits

For paper-style evaluation, repeat training for each split:

```bash
for split in 0 1 2 3 4; do
  cp conf_all "conf_split${split}"
  # edit train_wildtypes_list, test_wildtypes_list, checkpoint_dir,
  # means_path, and stds_path for this split before running
  python train.py "conf_split${split}"
done
```

Recommended artifact layout:

```text
predict/runs/rhomax_2026_07/
|-- split0/
|   |-- conf.json
|   |-- means.npy
|   |-- stds.npy
|   `-- checkpoints/
|-- split1/
...
`-- final_all_wildtypes/
```

Do not choose a final production checkpoint based only on a single lucky split.
Report mean, median, and standard deviation across splits.

## Data Preparation In Detail

### Phenotype Table Schema

`excel/data.xlsx` in the current repository has:

- 884 rows.
- 75 unique wildtypes.
- 29 columns:
  - `Name`
  - `Wildtype`
  - `Sequence`
  - `lmax`
  - `Method`
  - 24 pocket-position columns.

Use this file as the schema reference for new datasets.

### Sequence Cleaning

Recommended rules:

1. Uppercase all sequences.
2. Remove whitespace.
3. Reject empty sequences.
4. Reject sequences with unknown residues unless manually curated.
5. Keep a record of any sequence edits.
6. Confirm sequence length is plausible for microbial rhodopsins.
7. Confirm the conserved retinal-binding lysine region is present when
   applicable.

### Lambda Max Curation

Recommended rules:

1. Store all lambda max values in nm.
2. Keep the original publication/source measurement in a separate metadata
   table.
3. If multiple experimental values exist:
   - Prefer matching experimental conditions.
   - Prefer direct absorption measurements over indirect estimates.
   - Document averaging or selection rules.
4. Do not mix nm and eV in the `lmax` column.

The model loss is trained on nm differences. The paper also reports eV for
comparison, but the repository training target is `lmax` in nm.

### Structure Curation

Recommended rules:

1. Generate structures from the exact sequence in the table.
2. Keep one structure per row unless rows are exact duplicate sequences and you
   intentionally share structure artifacts.
3. Remove non-protein chains unless your parser is updated.
4. Confirm first chain is the intended chain.
5. Confirm binding-pocket residues exist after cutting.
6. Store per-structure QC metadata.

### Split Curation

Recommended rules:

1. Split by `Wildtype`, not by row.
2. Keep all variants of a wildtype in one split.
3. Maintain multiple independent splits.
4. Store split files in version control.
5. Store random seed and split-generation script.
6. Plot lambda max distributions for each train/test split.

## Feature Generation In Detail

Feature generation has three stages.

### Stage 1 - Pocket Residue Mapping And Cutting

Entrypoint:

```bash
python pipeline_auto/main.py pipeline_auto/config.json
```

Relevant files:

- `pipeline_auto/aligner_plus.py`
- `pipeline_auto/sequences.py`
- `pipeline_auto/cutter.py`

Process:

1. Parse query FASTA.
2. Add query to the reference alignment with MAFFT.
3. Extract the aligned query.
4. Map the 24 reference alignment positions to ungapped query residue positions.
5. Parse the PDB.
6. Keep only matching residues.
7. Save `cutted_parts0.pdb`.

Validation:

- The cut PDB should contain exactly 24 residues.
- The residues should correspond to expected binding-pocket positions.
- Failures here usually mean FASTA/PDB mismatch, bad chain selection, or
  alignment-position mapping problems.

### Stage 2 - Atom-Level Features

Entrypoint:

```bash
pipeline_auto/feature_maker.sh INPUT_PDB_DIR OUTPUT_FEATURE_DIR
```

This calls `feature_maker/interface2grid`, which creates atom-level feature
arrays from PDB files.

Current caveat:

- The binary is Linux-only in this checkout.
- The CMake files still contain historical absolute lab paths and should be
  modernized before rebuilding from source.

### Stage 3 - Amino-Acid Feature Augmentation

Entrypoint:

```bash
pipeline_auto/add_amino_acid_features.sh INPUT_PDB_DIR ATOM_FEATURE_DIR OUTPUT_FEATURE_DIR
```

This calls `feature_maker/add_amino_acid_features/main.py`.

Process:

1. Load atom-level features from stage 2.
2. Parse PDB residue names.
3. Map each residue to amino-acid physicochemical features using
   `feature_maker/add_amino_acid_features/amino_mapping`.
4. Horizontally concatenate amino-acid features and atom-level features.
5. Save a final `(N, 36)` NumPy array.

### Stage 4 - Edge Matrix Generation

Entrypoint:

```bash
python pipeline_auto/edge_maker.py INPUT_PDB_DIR OUTPUT_DIST_DIR
```

Process:

1. Parse all atoms in the cut PDB.
2. Compute Euclidean atom-atom distances.
3. Convert distances to inverse distances with `1 / distance`.
4. Set infinities to `0`.
5. Save `(N, N)` NumPy arrays.

The model uses `graph_th=2`. Since the matrix stores inverse distances, the
forward pass keeps entries where:

```python
edge_weight > 1 / graph_th
```

With `graph_th=2`, this corresponds to atom pairs closer than 2 A.

## Training Configuration Reference

### Data Paths

- `excel_path`: Excel phenotype table.
- `graph_features_path`: directory containing `cutted_parts{i}.npz`.
- `graph_dists_path`: directory containing `cutted_parts{i}_dists.npy`.
- `train_wildtypes_list`: one wildtype per line.
- `test_wildtypes_list`: one wildtype per line.

### Model Hyperparameters

- `model_name`: class name in `predict/models.py`.
- `number_features`: input feature count after zero-std columns are dropped.
- `hidden_layer_size`: hidden graph layer width.
- `out_layer_size`: graph layer output width.
- `graph_th`: distance threshold in angstroms, expressed through inverse
  distance filtering.
- `lr`: AdamW learning rate.
- `weight_decay`: L2 regularization coefficient. Default `0.00001`.
- `l1_alpha`: L1 regularization coefficient. Default `0.0001`.
- `dropout`: dropout passed to model constructors that use it. Default `0.1`.

### Training Controls

- `epochs`: finite number of epochs to train.
- `seed`: Python, NumPy, PyTorch, and CUDA seed.
- `num_samples`: number of samples drawn by `WeightedRandomSampler` per epoch.
- `test_goal`: console milestone threshold in nm MAE.
- `checkpoint_dir`: directory for best checkpoints.
- `pickle_file`: base name used when naming checkpoints.
- `load_checkpoint`: optional path to resume from a pickled model.

### Normalization Controls

- `means_path`: where training means are saved.
- `stds_path`: where training standard deviations are saved.
- `dataset_normalize_last`: if `false`, the last three features are not
  normalized; if `true`, all non-constant features are normalized.
- `indexes_to_keep`: raw feature columns to keep before zero-std filtering.

### W&B Controls

- `wandb`: default `false`.
- `wandb_project`: optional project name.
- `wandb_run_name`: optional run name.

If `wandb` is `true`, set credentials in the environment before training:

```bash
export WANDB_API_KEY=...
```

Do not commit API keys to the repository.

## Evaluation And Reproducibility

A reproducible run should preserve:

1. Git commit SHA.
2. Training config JSON.
3. Python version.
4. Full `pip freeze`.
5. PyTorch version.
6. PyTorch Geometric package versions.
7. CUDA version or CPU-only note.
8. Structure-generation metadata.
9. Phenotype-table version.
10. Split files.
11. Feature-generation code version.
12. Model checkpoint.
13. `means.npy`.
14. `stds.npy`.
15. Evaluation metrics.

Recommended evaluation metrics:

- Mean absolute error in nm.
- Median absolute error in nm.
- Standard deviation of absolute error in nm.
- Same metrics in eV, using `E = 1239.8 / wavelength_nm`.
- Per-wildtype error summaries.
- Error stratified by lambda max range, especially red-shifted sequences.
- Train/test distribution plots.

Recommended leakage checks:

- No overlap between train and test wildtype labels.
- No exact duplicate sequence shared between train and test.
- No mutant of a wildtype placed in the opposite split.
- No normalization statistics computed on test data.

## Publishing A New Reusable Model

When a trained model is ready for reuse, publish it as a bundle:

```text
models/<model_version>/
|-- model.pkl
|-- means.npy
|-- stds.npy
|-- train_config.json
|-- feature_contract.md
|-- metrics.json
|-- splits/
|   |-- train0
|   `-- test0
`-- environment/
    |-- pip_freeze.txt
    `-- git_commit.txt
```

Minimum metadata:

- Model name.
- Model class.
- Training data version.
- Number of rows.
- Number of wildtypes.
- Whether the final model was trained on all wildtypes.
- Feature dimensions before and after zero-std filtering.
- Graph threshold.
- Validation metrics.
- Known applicability domain.

To use a new model in current inference:

1. Copy the checkpoint to a stable path, for example
   `predict/models/my_model.pkl`.
2. Copy matching `means.npy` and `stds.npy`.
3. Create a config based on `predict/conf_all` with those normalization paths.
4. Run `calculate_one_rhodopsin.py` with the new config and model path.

Example:

```bash
cd predict
python calculate_one_rhodopsin.py \
  ./conf_my_model \
  ./models/my_model.pkl \
  ../pipeline_auto/result_my_model.txt \
  ../pipeline_auto/features/cutted_parts0.npz \
  ../pipeline_auto/dists/cutted_parts0_dists.npy
```

## Current Limitations And Modernization Notes

The following are known issues and recommended next steps.

### Packaging

Current state:

- Scripts are run from specific directories.
- Imports rely on current working directory.

Recommended modernization:

- Convert to an installable package, for example `opsigen`.
- Add console scripts:
  - `opsigen predict`
  - `opsigen make-features`
  - `opsigen train`
  - `opsigen evaluate`

### Feature Maker

Current state:

- Linux binary committed.
- CMake files include historical absolute paths.

Recommended modernization:

- Rebuild feature maker from clean source.
- Add Dockerfile for feature generation.
- Add a pure Python fallback if scientifically equivalent.
- Add tests that verify sample feature arrays.

### PDB Parsing

Current state:

- Assumes first model and first chain.
- Assumes simple residue numbering.

Recommended modernization:

- Add CLI/config options for chain ID and residue numbering mode.
- Validate sequence-to-structure mapping before cutting.
- Fail with clear diagnostics when residues are missing.

### FASTA Parsing

Current state:

- Single-query parser assumes sequence is on line 2.

Recommended modernization:

- Use Biopython `SeqIO` everywhere.
- Support multi-line FASTA and multiple records.

### Model Serialization

Current state:

- Uses whole-model pickle files.

Recommended modernization:

- Save `state_dict` plus model config.
- Keep backward compatibility loader for legacy pickles.
- Add artifact manifest with hash checksums.

### Evaluation

Current state:

- Training prints metrics but does not yet write a structured metrics file.

Recommended modernization:

- Write `metrics.csv` per epoch.
- Write `predictions.tsv` for test rows.
- Add an evaluation CLI.
- Add split aggregation reports.

## Troubleshooting

### `mafft: command not found`

Install MAFFT:

```bash
conda install -c bioconda mafft
```

### `cannot execute binary file: feature_maker/interface2grid`

You are probably on macOS or the wrong CPU architecture. Use Linux, Docker,
WSL2, or rebuild the feature maker for your platform.

### `AssertionError` In `cutter.py`

The cutter expected exactly 24 residues after cutting. Common causes:

- FASTA and PDB sequences do not match.
- The desired protein is not the first chain.
- The PDB has missing residues.
- Residue numbering is offset.
- MAFFT alignment did not map the query correctly.

### `FileNotFoundError` For Feature Or Distance Arrays

Training and prediction require both files:

```text
cutted_parts{i}.npz
cutted_parts{i}_dists.npy
```

Regenerate features or fix the paths in your config.

### Model Input Feature Mismatch

If PyTorch reports a feature dimension mismatch:

1. Check raw feature shape is `(N, 36)`.
2. Check `indexes_to_keep`.
3. Check `means.npy` and `stds.npy` came from the same training run.
4. Check how many `stds != 0` columns remain.
5. Set `number_features` to the post-filtered feature count for new training.

For the current legacy config, raw features are 36 columns and the model input
count is 34 after constant columns are removed.

### Training Skips Too Many Rows

`PDBDataset` skips rows when:

- Feature file is missing.
- Distance file is missing.
- Wildtype is not present in the selected split file.

Audit with a small script that checks every expected row index has both graph
files before training.

### Results Differ From The Paper

Expected causes:

- Different AlphaFold/ColabFold version.
- Different structures or structure ranks.
- Different split.
- Different PyTorch/PyG version.
- Different feature-maker build.
- Different random seed or GPU determinism behavior.
- Training on `train_all`/`test_all` rather than the paper splits.

For strict reproduction, archive the full environment and all generated graph
features.
