"""FASTA/PDB preprocessing pipeline for OpsiGen graph inputs."""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.spatial import distance

from opsigen.config import OpsinRecord, PreprocessConfig
from opsigen.exceptions import ExternalToolError, InputValidationError

LOGGER = logging.getLogger("opsigen.preprocessing")


@dataclass(frozen=True)
class FastaRecord:
    """A parsed FASTA record."""

    name: str
    sequence: str


@dataclass(frozen=True)
class PreprocessedRecord:
    """Paths produced for one preprocessed opsin."""

    id: str
    fasta_path: Path
    pdb_path: Path
    alignment_path: Path
    cut_pdb_path: Path
    atom_features_path: Path
    features_path: Path
    dists_path: Path


def sanitize_id(value: str) -> str:
    """Convert a record identifier into a stable filesystem-safe stem."""

    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    cleaned = cleaned.strip("._-")
    if not cleaned:
        raise InputValidationError(f"Invalid empty opsin identifier derived from {value!r}")
    return cleaned


def parse_fasta(path: Path) -> list[FastaRecord]:
    """Parse FASTA records without requiring Biopython."""

    if not path.exists():
        raise InputValidationError(f"FASTA file does not exist: {path}")

    records: list[FastaRecord] = []
    current_name: str | None = None
    current_sequence: list[str] = []
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if current_name is not None:
                records.append(FastaRecord(current_name, "".join(current_sequence)))
            current_name = line[1:].strip()
            current_sequence = []
        else:
            current_sequence.append(line)

    if current_name is not None:
        records.append(FastaRecord(current_name, "".join(current_sequence)))

    if not records:
        raise InputValidationError(f"No FASTA records found in {path}")
    return records


def read_single_fasta(path: Path) -> FastaRecord:
    """Read exactly the first FASTA record from a file."""

    records = parse_fasta(path)
    if len(records) > 1:
        LOGGER.warning("FASTA %s contains %s records; using the first one.", path, len(records))
    return records[0]


def read_aligned_sequence(path: Path, name: str) -> str:
    """Extract a named sequence from a FASTA alignment."""

    records = parse_fasta(path)
    for record in records:
        if record.name == name or record.name.split()[0] == name.split()[0]:
            return record.sequence
    available = ", ".join(record.name for record in records[-5:])
    raise InputValidationError(
        f"Could not find sequence {name!r} in MAFFT output {path}. Last records: {available}"
    )


def find_fasta_record(path: Path, name: str) -> FastaRecord | None:
    """Find a FASTA record by exact name or first whitespace-delimited token."""

    name_token = name.split()[0]
    for record in parse_fasta(path):
        if record.name == name or record.name.split()[0] == name_token:
            return record
    return None


def run_mafft_add(
    *,
    query_fasta: Path,
    reference_alignment: Path,
    output_alignment: Path,
    mafft_executable: str,
) -> None:
    """Run MAFFT with ``--add`` and ``--keeplength``."""

    if shutil.which(mafft_executable) is None and not Path(mafft_executable).exists():
        raise ExternalToolError(
            f"MAFFT executable was not found: {mafft_executable}. "
            "Install MAFFT or set preprocessing.mafft_executable."
        )
    if not reference_alignment.exists():
        raise InputValidationError(f"Reference alignment does not exist: {reference_alignment}")

    output_alignment.parent.mkdir(parents=True, exist_ok=True)
    command = [
        mafft_executable,
        "--add",
        str(query_fasta),
        "--keeplength",
        str(reference_alignment),
    ]
    LOGGER.info("Running MAFFT for %s", query_fasta)
    with output_alignment.open("w") as stdout:
        completed = subprocess.run(
            command,
            stdout=stdout,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    if completed.returncode != 0:
        raise ExternalToolError(
            f"MAFFT failed for {query_fasta} with exit code {completed.returncode}:\n"
            f"{completed.stderr.strip()}"
        )


def residue_positions_from_alignment(
    aligned_sequence: str,
    aligned_positions: Iterable[int],
) -> list[int]:
    """Convert reference alignment columns to residue positions in the added sequence."""

    positions: list[int] = []
    for position in aligned_positions:
        if position <= 0 or position >= len(aligned_sequence):
            raise InputValidationError(
                f"Alignment position {position} is outside the aligned sequence length "
                f"({len(aligned_sequence)})."
            )
        ungapped_before = len(aligned_sequence[:position].replace("-", ""))
        positions.append(ungapped_before + int(aligned_sequence[position - 1] == "-"))
    return positions


def alignment_columns_for_reference_sites(
    reference_aligned_sequence: str,
    reference_residue_sites: Iterable[int],
) -> dict[int, int]:
    """Map 1-based reference residue numbers to 0-based alignment columns."""

    requested = [int(site) for site in reference_residue_sites]
    requested_set = set(requested)
    columns: dict[int, int] = {}
    ungapped_position = 0
    for column, residue in enumerate(reference_aligned_sequence):
        if residue == "-":
            continue
        ungapped_position += 1
        if ungapped_position in requested_set:
            columns[ungapped_position] = column
    missing = [site for site in requested if site not in columns]
    if missing:
        raise InputValidationError(
            "Reference residue site(s) were not found in the aligned reference sequence: "
            f"{missing}. Check reference_sequence_id/reference_sequence_path and numbering."
        )
    return columns


def residue_positions_from_reference_sites(
    query_aligned_sequence: str,
    reference_aligned_sequence: str,
    reference_residue_sites: Iterable[int],
    *,
    gap_strategy: str = "next",
) -> list[int]:
    """Convert reference residue numbers to query residue positions.

    ``reference_residue_sites`` are 1-based positions in the ungapped reference
    sequence, for example bovine rhodopsin numbering from Hagen et al. The
    returned positions are 1-based residue numbers in the query structure.
    """

    if gap_strategy not in {"next", "skip", "error"}:
        raise InputValidationError(
            f"Invalid site_gap_strategy={gap_strategy!r}; expected one of: next, skip, error."
        )

    site_to_column = alignment_columns_for_reference_sites(
        reference_aligned_sequence,
        reference_residue_sites,
    )
    positions: list[int] = []
    for site in reference_residue_sites:
        column = site_to_column[int(site)]
        if column >= len(query_aligned_sequence):
            raise InputValidationError(
                f"Alignment column {column + 1} for reference site {site} is outside query alignment."
            )
        if query_aligned_sequence[column] != "-":
            positions.append(len(query_aligned_sequence[: column + 1].replace("-", "")))
            continue
        if gap_strategy == "skip":
            LOGGER.warning("Skipping reference site %s because query has a gap at that alignment column.", site)
            continue
        if gap_strategy == "error":
            raise InputValidationError(
                f"Query has a gap at reference site {site}; set site_gap_strategy='skip' or 'next' to continue."
            )
        next_position = len(query_aligned_sequence[:column].replace("-", "")) + 1
        ungapped_length = len(query_aligned_sequence.replace("-", ""))
        if next_position > ungapped_length:
            LOGGER.warning("Skipping reference site %s because query gap is after the last residue.", site)
            continue
        LOGGER.warning(
            "Reference site %s maps to a query gap; using next query residue position %s.",
            site,
            next_position,
        )
        positions.append(next_position)
    return positions


def resolve_reference_aligned_sequence(config: PreprocessConfig) -> str:
    """Resolve the aligned reference sequence used for reference-residue numbering."""

    if not config.reference_residue_sites:
        raise InputValidationError("No reference_residue_sites were configured.")
    if not config.reference_sequence_id:
        raise InputValidationError(
            "reference_residue_sites requires preprocessing.reference_sequence_id, "
            "for example 'Bovine'."
        )

    existing_record = find_fasta_record(config.reference_alignment, config.reference_sequence_id)
    if existing_record is not None:
        return existing_record.sequence

    if config.reference_sequence_path is None:
        raise InputValidationError(
            f"Reference sequence {config.reference_sequence_id!r} was not found in "
            f"{config.reference_alignment}. Set preprocessing.reference_sequence_path."
        )

    output_alignment = config.output_dir / "reference_alignments" / f"{sanitize_id(config.reference_sequence_id)}.fasta"
    run_mafft_add(
        query_fasta=config.reference_sequence_path,
        reference_alignment=config.reference_alignment,
        output_alignment=output_alignment,
        mafft_executable=config.mafft_executable,
    )
    return read_aligned_sequence(output_alignment, config.reference_sequence_id)


def cut_pdb_to_residues(input_pdb: Path, output_pdb: Path, residue_numbers: Iterable[int]) -> None:
    """Write a PDB containing only selected residues from the first model/chain."""

    if not input_pdb.exists():
        raise InputValidationError(f"PDB file does not exist: {input_pdb}")

    try:
        from Bio import PDB
        from Bio.PDB import PDBIO
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ExternalToolError("Biopython is required for PDB preprocessing.") from exc

    selected = set(int(position) for position in residue_numbers)
    if not selected:
        raise InputValidationError("No residue positions were selected for PDB cutting.")

    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure(input_pdb.stem, str(input_pdb))
    model = next(structure.get_models(), None)
    if model is None:
        raise InputValidationError(f"PDB has no models: {input_pdb}")
    chain = next(model.get_chains(), None)
    if chain is None:
        raise InputValidationError(f"PDB first model has no chains: {input_pdb}")

    for residue in list(chain.get_residues()):
        if residue.id[1] not in selected:
            chain.detach_child(residue.id)

    remaining = list(chain.get_residues())
    if len(remaining) != len(selected):
        LOGGER.warning(
            "Cut PDB %s retained %s residues for %s requested positions.",
            input_pdb,
            len(remaining),
            len(selected),
        )
    if not remaining:
        raise InputValidationError(f"PDB cutting retained zero residues for {input_pdb}")

    output_pdb.parent.mkdir(parents=True, exist_ok=True)
    io = PDBIO()
    io.set_structure(structure)
    io.save(str(output_pdb))


def run_native_feature_maker(
    cut_pdb_dir: Path,
    atom_features_dir: Path,
    binary: Path,
    chem_lib_path: Path | None = None,
) -> None:
    """Run the native ``interface2grid`` feature generator."""

    if not binary.exists():
        raise ExternalToolError(
            f"Native feature generator does not exist: {binary}. "
            "Build feature_maker/interface2grid or set preprocessing.feature_maker_binary."
        )
    if chem_lib_path is None:
        chem_lib_path = binary.parent / "chem.lib"
    if not chem_lib_path.exists():
        raise ExternalToolError(
            f"Native feature generator chemistry library does not exist: {chem_lib_path}. "
            "Set preprocessing.chem_lib_path to the correct chem.lib file."
        )
    atom_features_dir.mkdir(parents=True, exist_ok=True)
    output_dir_arg = str(atom_features_dir.resolve()) + "/"
    command = [
        str(binary.resolve()),
        "-i",
        str(cut_pdb_dir.resolve()),
        "-o",
        output_dir_arg,
        "-l",
        str(chem_lib_path.resolve()),
    ]
    LOGGER.info("Running native feature maker for %s", cut_pdb_dir)
    completed = subprocess.run(
        command,
        cwd=str(binary.resolve().parent),
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise ExternalToolError(
            f"Feature maker failed with exit code {completed.returncode}:\n"
            f"{completed.stderr.strip()}"
        )


def load_amino_mapping(path: Path) -> dict[str, np.ndarray]:
    """Load amino-acid descriptor mapping."""

    if not path.exists():
        raise InputValidationError(f"Amino-acid mapping file does not exist: {path}")
    mapping = json.loads(path.read_text())
    return {key: np.asarray(value, dtype=float) for key, value in mapping.items()}


def _amino_acid_from_pdb_line(line: str) -> str | None:
    tokens = line.split()
    if len(tokens) < 7:
        return None
    amino_acid_index = 3 if len(tokens) in {11, 12} or tokens[0] == "HETATM" else 2
    return tokens[amino_acid_index]


def amino_features_for_pdb(pdb_path: Path, amino_mapping: dict[str, np.ndarray]) -> np.ndarray:
    """Generate per-atom amino-acid descriptors from PDB ATOM/HETATM lines."""

    rows: list[np.ndarray] = []
    for line in pdb_path.read_text().splitlines():
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue
        amino_acid = _amino_acid_from_pdb_line(line)
        if amino_acid is None:
            continue
        if amino_acid not in amino_mapping:
            raise InputValidationError(
                f"Amino acid {amino_acid!r} from {pdb_path} is not present in {len(amino_mapping)}-entry mapping."
            )
        rows.append(np.copy(amino_mapping[amino_acid]))
    if not rows:
        raise InputValidationError(f"No ATOM/HETATM features could be parsed from {pdb_path}")
    return np.vstack(rows)


def append_amino_acid_features(
    *,
    cut_pdb_dir: Path,
    atom_features_dir: Path,
    output_features_dir: Path,
    amino_mapping_path: Path,
    original_feature_length: int,
) -> None:
    """Combine native atom features with amino-acid descriptors."""

    amino_mapping = load_amino_mapping(amino_mapping_path)
    output_features_dir.mkdir(parents=True, exist_ok=True)
    for pdb_path in sorted(cut_pdb_dir.glob("*.pdb")):
        stem = pdb_path.stem
        atom_feature_path = atom_features_dir / f"{stem}.npz"
        if not atom_feature_path.exists():
            raise InputValidationError(f"Missing native atom feature file for {stem}: {atom_feature_path}")

        atom_features_raw = np.load(atom_feature_path)
        if atom_features_raw.size % original_feature_length != 0:
            raise InputValidationError(
                f"Native feature array {atom_feature_path} with shape {atom_features_raw.shape} "
                f"is not divisible by original_feature_length={original_feature_length}."
            )
        atom_features = atom_features_raw.reshape(-1, original_feature_length)
        amino_features = amino_features_for_pdb(pdb_path, amino_mapping)
        if amino_features.shape[0] != atom_features.shape[0]:
            raise InputValidationError(
                f"Feature row mismatch for {stem}: {amino_features.shape[0]} amino rows vs "
                f"{atom_features.shape[0]} native atom rows."
            )
        with (output_features_dir / f"{stem}.npz").open("wb") as handle:
            np.save(handle, np.hstack([amino_features, atom_features]))


def write_distance_matrix(input_pdb: Path, output_path: Path) -> None:
    """Create an inverse-distance adjacency matrix from a PDB."""

    try:
        from Bio import PDB
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ExternalToolError("Biopython is required for PDB preprocessing.") from exc

    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure(input_pdb.stem, str(input_pdb))
    atoms = list(structure.get_atoms())
    if not atoms:
        raise InputValidationError(f"No atoms found in PDB: {input_pdb}")
    coords = np.asarray([atom.coord for atom in atoms], dtype=np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        dists = 1 / distance.cdist(coords, coords)
    dists[~np.isfinite(dists)] = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, dists)


def _planned_paths(record: OpsinRecord, config: PreprocessConfig) -> PreprocessedRecord:
    if record.fasta_path is None:
        raise InputValidationError(f"Record {record.id} is missing fasta_path.")
    if record.pdb_path is None:
        raise InputValidationError(f"Record {record.id} is missing pdb_path.")
    record_id = sanitize_id(record.id)
    return PreprocessedRecord(
        id=record_id,
        fasta_path=record.fasta_path,
        pdb_path=record.pdb_path,
        alignment_path=config.output_dir / "alignments" / f"{record_id}.fasta",
        cut_pdb_path=config.output_dir / "cut_pdbs" / f"{record_id}.pdb",
        atom_features_path=config.output_dir / "atom_features" / f"{record_id}.npz",
        features_path=config.output_dir / "features" / f"{record_id}.npz",
        dists_path=config.output_dir / "dists" / f"{record_id}_dists.npy",
    )


def preprocess_opsins(
    records: Iterable[OpsinRecord],
    config: PreprocessConfig,
) -> list[PreprocessedRecord]:
    """Run the full graph preprocessing workflow for many opsins."""

    planned = [_planned_paths(record, config) for record in records]
    if not planned:
        raise InputValidationError("No records were provided for preprocessing.")

    config.output_dir.mkdir(parents=True, exist_ok=True)
    cut_dir = config.output_dir / "cut_pdbs"
    cut_dir.mkdir(parents=True, exist_ok=True)
    reference_aligned_sequence = (
        resolve_reference_aligned_sequence(config)
        if config.reference_residue_sites
        else None
    )

    for item in planned:
        fasta_record = read_single_fasta(item.fasta_path)
        run_mafft_add(
            query_fasta=item.fasta_path,
            reference_alignment=config.reference_alignment,
            output_alignment=item.alignment_path,
            mafft_executable=config.mafft_executable,
        )
        aligned_sequence = read_aligned_sequence(item.alignment_path, fasta_record.name)
        if reference_aligned_sequence is not None and config.reference_residue_sites is not None:
            residue_numbers = residue_positions_from_reference_sites(
                aligned_sequence,
                reference_aligned_sequence,
                config.reference_residue_sites,
                gap_strategy=config.site_gap_strategy,
            )
        else:
            residue_numbers = residue_positions_from_alignment(
                aligned_sequence,
                config.aligned_positions,
            )
        LOGGER.info("Cutting %s to %s selected residue positions.", item.pdb_path, len(residue_numbers))
        cut_pdb_to_residues(item.pdb_path, item.cut_pdb_path, residue_numbers)

    run_native_feature_maker(
        cut_dir,
        config.output_dir / "atom_features",
        config.feature_maker_binary,
        config.chem_lib_path,
    )
    append_amino_acid_features(
        cut_pdb_dir=cut_dir,
        atom_features_dir=config.output_dir / "atom_features",
        output_features_dir=config.output_dir / "features",
        amino_mapping_path=config.amino_mapping_path,
        original_feature_length=config.original_feature_length,
    )

    for item in planned:
        write_distance_matrix(item.cut_pdb_path, item.dists_path)

    LOGGER.info("Preprocessed %s opsin record(s) into %s", len(planned), config.output_dir)
    return planned
