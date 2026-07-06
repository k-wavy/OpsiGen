# Native Feature Maker

`interface2grid` is the native helper used by `opsigen.preprocessing` to create atom-level feature arrays from cut PDB files.

The Python pipeline calls it as:

```bash
feature_maker/interface2grid -i <cut_pdb_dir> -o <atom_features_dir>
```

If you need to rebuild it, configure local dependency roots with CMake cache variables instead of editing `CMakeLists.txt`:

```bash
cmake -S feature_maker -B feature_maker/build \
  -DGAMB_ROOT=/path/to/gamb \
  -DDOCKINGLIB_ROOT=/path/to/DockingLib
cmake --build feature_maker/build
```

The generated atom features are not the final model inputs. OpsiGen appends amino-acid descriptors from `feature_maker/add_amino_acid_features/amino_mapping` and writes final graph features under the configured preprocessing output directory.
