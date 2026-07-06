# OpsiGen

OpsiGen is the research codebase behind **RhoMax**, a geometric deep learning
tool for predicting microbial rhodopsin absorption maxima from sequence-derived
structure. It uses an opsin amino-acid sequence plus a matching protein
structure, extracts the 24 retinal-binding-pocket residues, represents their
atoms as a graph, and predicts lambda max in nm.

![Main Figure](Images/main_figure.png)

The repository is being modernized into a reusable and reproducible platform.
This README is the practical entry point; the detailed operating manual is in
[docs/USAGE_AND_TRAINING_GUIDE.md](docs/USAGE_AND_TRAINING_GUIDE.md).
For an executable notebook version modeled after the VPOD workflow style, use
[docs/OpsiGen_training_prediction_workflow.ipynb](docs/OpsiGen_training_prediction_workflow.ipynb).

## What This Repository Contains

- A copy of the RhoMax publication:
  `rhomax-computational-prediction-of-rhodopsin-absorption-maxima-using-geometric-deep-learning.pdf`.
- The original phenotype table and wildtype split files under `excel/`.
- A single-query feature-generation workflow under `pipeline_auto/`.
- Feature-maker utilities under `feature_maker/`.
- PyTorch/PyTorch Geometric model definitions, training code, normalization
  arrays, and a legacy pickled model under `predict/`.
- Detailed documentation for existing-model use, new-model training, data prep,
  preprocessing, and artifact publishing under `docs/`.

## Important Platform Notes

- OpsiGen/RhoMax is **not sequence-only**. It needs a matching PDB structure for
  each query sequence.
- The recommended structure source is AlphaFold2 or ColabFold. The repository
  does not currently run AlphaFold2 itself.
- The committed `feature_maker/interface2grid` binary is Linux-only. Full
  feature generation on macOS requires Linux, Docker/WSL, or rebuilding the
  feature maker.
- Legacy SLURM/lab-path scripts remain in the repo for provenance. Prefer the
  direct Python commands below for local use.
- `predict/model_pickel` keeps the historical misspelling. Keep that exact path
  when using the committed legacy model.

## Quick Setup

Use Linux for the full pipeline.

```bash
conda create -n opsigen python=3.9
conda activate opsigen
python -m pip install --upgrade pip
pip install -r requirements.txt
conda install -c bioconda mafft
```

If PyTorch Geometric packages fail to install, use the official PyTorch
Geometric wheel selector for a PyTorch 1.13-compatible install.

## Predict With The Existing Model

1. Put one amino-acid FASTA record at:

   ```text
   pipeline_auto/sample_fasta.fasta
   ```

2. Put the matching PDB at:

   ```text
   pipeline_auto/sample_pdb.pdb
   ```

3. Generate graph features:

   ```bash
   cd pipeline_auto
   rm -rf dists features cutted_parts
   mkdir -p dists features cutted_parts mafft_alignments
   python ./main.py ./config.json
   ```

4. Predict lambda max:

   ```bash
   cd ../predict
   python calculate_one_rhodopsin.py \
     ./conf_all \
     ./model_pickel \
     ../pipeline_auto/result.txt \
     ../pipeline_auto/features/cutted_parts0.npz \
     ../pipeline_auto/dists/cutted_parts0_dists.npy
   ```

5. Read:

   ```text
   pipeline_auto/result.txt
   ```

The output is a single predicted absorption maximum in nm.

## Train A New Model

At a high level:

1. Prepare `excel/data.xlsx` with one row per measured opsin or mutant.
2. Generate one matching structure per row.
3. Generate one feature array and one distance matrix per row:

   ```text
   db_features/cutted_parts{i}.npz
   db_dists/cutted_parts{i}_dists.npy
   ```

4. Split by `Wildtype`, not by individual row.
5. Create a training config from `predict/conf_all`.
6. Run:

   ```bash
   cd predict
   python train.py ./conf_all
   ```

Training now uses finite config-driven epochs, explicit seeds, explicit
normalization paths, and local checkpoint output. Weights & Biases is disabled
unless `"wandb": true` is set in the config.

Read the complete training protocol in
[docs/USAGE_AND_TRAINING_GUIDE.md](docs/USAGE_AND_TRAINING_GUIDE.md#training-new-models).

## Key Data Contracts

Training table columns:

- `Name`: unique sequence or variant name.
- `Wildtype`: parent wildtype label used for leakage-safe splitting.
- `Sequence`: ungapped amino-acid sequence.
- `lmax`: experimental lambda max in nm.
- `Method`: optional measurement method.
- 24 binding-pocket audit columns matching the historical table.

Feature tensors:

- Features: `cutted_parts{i}.npz`, shape `(N, 36)`.
- Distances: `cutted_parts{i}_dists.npy`, shape `(N, N)`.
- `N` is the number of atoms in the 24 selected pocket residues.

Normalization:

- Training writes `means.npy` and `stds.npy`.
- Prediction must use the matching normalization arrays from the same training
  run.
- The current legacy model uses 36 raw feature columns, then drops constant
  columns to feed 34 model input features.

## Citation

If you use this repository, cite the RhoMax paper:

Sela, M.; Church, J. R.; Schapiro, I.; Schneidman-Duhovny, D. **RhoMax:
Computational Prediction of Rhodopsin Absorption Maxima Using Geometric Deep
Learning.** *Journal of Chemical Information and Modeling* 2024, 64,
4630-4639. https://doi.org/10.1021/acs.jcim.4c00467

## Development Priorities

The next platform improvements should be:

- Package the project as an installable Python module.
- Replace directory-sensitive scripts with stable CLI commands.
- Add a Dockerfile for Linux feature generation.
- Replace whole-model pickle artifacts with `state_dict` plus model manifests.
- Add structured training metrics and evaluation reports.
- Add tests for FASTA parsing, pocket residue mapping, feature shapes, and
  prediction artifact compatibility.
