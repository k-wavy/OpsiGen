# OpsiGen

OpsiGen predicts rhodopsin absorption wavelength from sequence/PDB-derived graph features.

![Main Figure](Images/main_figure.png)

## Repository Layout

```text
opsigen/                  Python package with reusable pipeline code
  preprocessing/          FASTA/PDB alignment, residue cutting, feature generation, distances
  data.py                 Graph dataset and normalization utilities
  training.py             Central training workflow
  prediction.py           Central batch prediction workflow
  cli.py                  `python -m opsigen ...` command line entry point
configs/                  Example training and prediction configs
examples/                 Example batch input manifest
feature_maker/            Native `interface2grid` feature generator and source assets
pipeline_auto/            Legacy compatibility wrapper plus sample FASTA/PDB inputs
predict/                  Legacy compatibility wrappers plus bundled sample model artifacts
my_notebook.ipynb         Import-based example notebook
docs/MIGRATION.md         Audit of replaced/deprecated legacy scripts
```

## Install

```bash
pip install -r requirements.txt
pip install -e .
```

Prediction from FASTA/PDB inputs also requires `mafft` and the native `feature_maker/interface2grid` binary. If you rebuild the native helper, configure its dependency roots with CMake variables such as `-DGAMB_ROOT=/path/to/gamb` and `-DDOCKINGLIB_ROOT=/path/to/DockingLib`.
The native helper also needs `feature_maker/chem.lib`; set `preprocessing.chem_lib_path` if it is not next to `interface2grid`.

## Train A New Model

Edit `configs/train.example.json` so the graph feature and distance directories point to your training dataset. Then run:

```bash
python -m opsigen train --config configs/train.example.json
```

For the WDS sequence/lambda-max files, prepare the metadata and configs first:

```bash
python -m opsigen prepare-wds \
  --meta /path/to/wds_meta.tsv \
  --fasta /path/to/wds.fasta \
  --pdb-dir /path/to/pdb_folder \
  --output-dir datasets/wds \
  --configs-dir configs
```

If the PDB directory is not available yet, omit `--pdb-dir`; this still creates the training table, FASTA records, split files, and config templates. Re-run the same command with `--pdb-dir` before preprocessing.
Use the unaligned sequence FASTA for `--fasta`; the WDS preprocessing config builds a shared animal-opsin MSA with MAFFT and maps bovine-numbered reference sites through that alignment.

Then run:

```bash
python -m opsigen preprocess --config configs/preprocess.wds.json
python -m opsigen train --config configs/train.wds.json
```

The WDS configs use Bovine as `reference_sequence_id` and bovine-numbered spectral tuning sites from Hagen et al. 2023. Edit `preprocessing.reference_residue_sites` to change which residues become GNN nodes.

The training entry point performs configuration loading, input validation, dataset normalization, weighted sampling, model construction, training/evaluation, checkpointing, metrics logging, and metadata writing.

Training outputs are written under `outputs.output_dir`, for example:

```text
runs/train/example/
  artifacts/means.npy
  artifacts/stds.npy
  checkpoints/best_opsigen_model.pkl
  checkpoints/opsigen_model.pkl
  logs/metrics.jsonl
  logs/opsigen.log
  metadata.json
```

## Run Batch Prediction

For FASTA/PDB inputs, edit `configs/predict.example.json` or provide an input manifest with columns `id,fasta_path,pdb_path`.

```bash
python -m opsigen predict --config configs/predict.example.json
```

The prediction entry point runs preprocessing when records do not already provide `features_path` and `dists_path`, then loads the model, applies normalization, predicts each graph, and writes:

```text
runs/predict/example/
  predictions.csv
  metadata.json
  logs/opsigen.log
  preprocessed/
    alignments/
    atom_features/
    cut_pdbs/
    dists/
    features/
```

To predict from already-preprocessed graph files, use records with `features_path` and `dists_path` instead of `fasta_path` and `pdb_path`.

## Programmatic Usage

```python
from opsigen.config import TrainingConfig, PredictionConfig
from opsigen.training import train_model
from opsigen.prediction import run_predictions

training_config = TrainingConfig.from_file("configs/train.example.json")
training_result = train_model(training_config)

prediction_config = PredictionConfig.from_file("configs/predict.example.json")
prediction_result = run_predictions(prediction_config)
```

The notebook `my_notebook.ipynb` demonstrates the same import-based workflow with toggles for expensive preprocessing, training, and prediction steps.

## Legacy Commands

These wrappers remain for compatibility but should not be used for new work:

```bash
python pipeline_auto/main.py pipeline_auto/config.json
python predict/train.py configs/train.example.json
python predict/calculate_one_rhodopsin.py predict/conf_all predict/model_pickel output.txt features.npz dists.npy
```

See `docs/MIGRATION.md` for the script audit and replacement map.
