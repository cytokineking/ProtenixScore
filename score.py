import csv
import hashlib
import json
import logging
import os
import shutil
import tempfile
import time
import traceback
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import torch
from biotite.structure import AtomArray, get_chain_starts, get_residue_starts

from protenix.data.constants import mmcif_restype_3to1
from protenix.data.filter import Filter
from protenix.data.json_maker import atom_array_to_input_json
from protenix.data.parser import MMCIFParser
from protenix.data.utils import pdb_to_cif
from protenix.data.infer_data_pipeline import InferenceDataset
from protenix.utils.file_io import save_json
from protenix.utils.torch_utils import round_values
from protenix.utils.torch_utils import to_device
from runner.batch_inference import get_default_runner

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@dataclass
class ScoreResult:
    sample_name: str
    summary: dict
    full_data: Optional[dict]
    output_dir: Path
    msa_resolution: Optional[List[dict]] = None
    prep_seconds: Optional[float] = None
    model_seconds: Optional[float] = None
    total_seconds: Optional[float] = None


@dataclass(frozen=True)
class MSAMapEntry:
    row_number: int
    sample_id: Optional[str]
    chain_id: Optional[str]
    role: Optional[str]
    sequence_norm: Optional[str]
    msa_dir: Optional[Path]
    non_pairing_path: Optional[Path]
    pairing_path: Optional[Path]


@dataclass
class MSAMapIndex:
    sample_chain: Dict[Tuple[str, str], MSAMapEntry]
    role_sequence: Dict[Tuple[str, str], MSAMapEntry]
    sequence_only: Dict[str, MSAMapEntry]
    entries: List[MSAMapEntry]


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def collect_input_files(input_path: str, recursive: bool, globs: Iterable[str]) -> List[Path]:
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    patterns = list(globs)
    files: List[Path] = []
    if path.is_file():
        files = [path]
    else:
        if recursive:
            for pattern in patterns:
                files.extend(path.rglob(pattern))
        else:
            for pattern in patterns:
                files.extend(path.glob(pattern))
    files = [p for p in files if p.suffix.lower() in {".pdb", ".cif"}]
    files = sorted(set(files))
    return files


def _sanitize_name(name: str) -> str:
    name = name.strip().replace(" ", "_")
    return "".join(ch for ch in name if ch.isalnum() or ch in {"_", "-", "."})


def _prepare_intermediate_dirs(output_dir: Path, keep: bool, intermediate_dir: Optional[str]) -> Tuple[Optional[Path], Optional[tempfile.TemporaryDirectory]]:
    if keep:
        if intermediate_dir is None:
            inter_dir = output_dir / "intermediate"
        else:
            inter_dir = Path(intermediate_dir)
        _ensure_dir(inter_dir)
        return inter_dir, None
    tmp_dir = tempfile.TemporaryDirectory()
    return Path(tmp_dir.name), tmp_dir


def _parse_structure_to_json(
    cif_path: Path,
    sample_name: str,
    assembly_id: Optional[str],
    altloc: str,
    output_json: Optional[Path],
) -> Tuple[dict, AtomArray]:
    parser = MMCIFParser(cif_path)
    atom_array = parser.get_structure(altloc=altloc, model=1, bond_lenth_threshold=None)

    atom_array = Filter.remove_water(atom_array)
    atom_array = Filter.remove_hydrogens(atom_array)
    atom_array = parser.mse_to_met(atom_array)
    atom_array = Filter.remove_element_X(atom_array)

    if any(["DIFFRACTION" in m for m in parser.methods]):
        atom_array = Filter.remove_crystallization_aids(atom_array, parser.entity_poly_type)

    if assembly_id is not None:
        atom_array = parser.expand_assembly(atom_array, assembly_id)

    json_dict = atom_array_to_input_json(
        atom_array,
        parser,
        assembly_id=assembly_id,
        output_json=str(output_json) if output_json is not None else None,
        sample_name=sample_name,
        save_entity_and_asym_id=True,
        include_discont_poly_poly_bonds=True,
    )
    if isinstance(json_dict, dict):
        json_dict = [json_dict]
    if output_json is not None:
        with open(output_json, "w") as f:
            json.dump(json_dict, f, indent=2)
    return json_dict, atom_array


def _build_chain_order_by_entity(atom_array: AtomArray) -> List[str]:
    chain_starts = get_chain_starts(atom_array, add_exclusive_stop=False)
    chain_starts_atom_array = atom_array[chain_starts]
    ordered_chain_ids: List[str] = []
    unique_label_entity_id = np.unique(atom_array.label_entity_id)
    for label_entity_id in unique_label_entity_id:
        chain_ids = chain_starts_atom_array.chain_id[
            chain_starts_atom_array.label_entity_id == label_entity_id
        ]
        ordered_chain_ids.extend(chain_ids.tolist())
    return ordered_chain_ids


def _replace_unknown_residues(json_dict: dict, sample_name: str) -> None:
    """Replace unknown protein residues ('X') with glycine to avoid CCD lookup failures."""
    if isinstance(json_dict, list):
        samples = json_dict
    else:
        samples = [json_dict]
    total_replaced = 0
    for sample in samples:
        sequences = sample.get("sequences", [])
        for entry in sequences:
            protein = entry.get("proteinChain")
            if not protein:
                continue
            seq = protein.get("sequence", "")
            if "X" in seq:
                count = seq.count("X")
                protein["sequence"] = seq.replace("X", "G")
                total_replaced += count
    if total_replaced > 0:
        logger.warning(
            "%s: replaced %d unknown residues (X) with GLY for scoring",
            sample_name,
            total_replaced,
        )


def _extract_chain_sequences(atom_array: AtomArray) -> Dict[str, str]:
    """Extract per-chain sequences from the atom array."""
    sequences: Dict[str, str] = {}
    if hasattr(atom_array, "label_asym_id"):
        chain_ids = np.unique(atom_array.label_asym_id)
        chain_field = "label_asym_id"
    else:
        chain_ids = np.unique(atom_array.chain_id)
        chain_field = "chain_id"
    for chain_id in chain_ids:
        if chain_field == "label_asym_id":
            chain_atoms = atom_array[atom_array.label_asym_id == chain_id]
        else:
            chain_atoms = atom_array[atom_array.chain_id == chain_id]
        starts = get_residue_starts(chain_atoms, add_exclusive_stop=True)
        res_names = chain_atoms.res_name[starts[:-1]]
        seq = "".join(mmcif_restype_3to1.get(res, "X") for res in res_names)
        sequences[str(chain_id)] = seq
    return sequences


def _apply_chain_sequence_overrides(
    json_dict: dict,
    chain_sequences: Dict[str, str],
    overrides: Dict[str, str],
    sample_name: str,
) -> None:
    """Override protein sequences in JSON using chain-derived or user-provided sequences."""
    if isinstance(json_dict, list):
        samples = json_dict
    else:
        samples = [json_dict]

    used_overrides = set()
    for sample in samples:
        sequences = sample.get("sequences", [])
        for entry in sequences:
            protein = entry.get("proteinChain")
            if not protein:
                continue
            label_asym_ids = protein.get("label_asym_id") or []
            chosen_seq = None
            chosen_chain = None
            for chain_id in label_asym_ids:
                if chain_id in overrides:
                    chosen_seq = overrides[chain_id]
                    chosen_chain = chain_id
                    used_overrides.add(chain_id)
                    break
            if chosen_seq is None:
                for chain_id in label_asym_ids:
                    if chain_id in chain_sequences:
                        chosen_seq = chain_sequences[chain_id]
                        chosen_chain = chain_id
                        break
            if chosen_seq:
                protein["sequence"] = chosen_seq
                logger.info(
                    "%s: set sequence for chain %s (len=%d)",
                    sample_name,
                    chosen_chain,
                    len(chosen_seq),
                )
                logger.info(
                    "%s: chain %s sequence: %s",
                    sample_name,
                    chosen_chain,
                    chosen_seq,
                )

    unused = [k for k in overrides.keys() if k not in used_overrides]
    if unused:
        logger.warning("%s: chain_sequence overrides not used: %s", sample_name, ",".join(unused))


def _parse_chain_sequence_overrides(values: Iterable[str]) -> Dict[str, str]:
    overrides: Dict[str, str] = {}
    for raw in values:
        if not raw:
            continue
        if "=" not in raw:
            raise ValueError(f"Invalid --chain_sequence value: {raw} (expected CHAIN=SEQUENCE)")
        chain_id, seq = raw.split("=", 1)
        chain_id = chain_id.strip()
        seq = seq.strip()
        if not chain_id or not seq:
            raise ValueError(f"Invalid --chain_sequence value: {raw} (empty chain or sequence)")
        overrides[chain_id] = seq
    return overrides


def _parse_chain_list(value: Optional[str]) -> List[str]:
    if not value:
        return []
    items = []
    for part in value.split(","):
        part = part.strip()
        if part:
            items.append(part)
    return items


def _load_fasta_sequences(path: Optional[str]) -> List[str]:
    if not path:
        return []
    sequences: List[str] = []
    current: List[str] = []
    with open(path, "r") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current:
                    sequences.append("".join(current))
                    current = []
                continue
            current.append(line)
    if current:
        sequences.append("".join(current))
    return sequences


def _hash_sequence(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("utf-8")).hexdigest()


def _normalize_sequence(sequence: str) -> str:
    normalized = []
    for ch in sequence.upper():
        if ch.isspace() or ch in {"-", "."}:
            continue
        normalized.append(ch)
    return "".join(normalized)


def _is_role_enabled(use_msas: str, role: str) -> bool:
    if use_msas == "both":
        return True
    if use_msas == "target":
        return role == "target"
    if use_msas == "binder":
        return role == "binder"
    return False


def _read_csv_row_value(row: dict, key: str) -> str:
    value = row.get(key, "")
    if value is None:
        return ""
    return str(value).strip()


def _resolve_csv_path(path_value: str, csv_path: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (csv_path.parent / path).resolve()


def _validate_msa_dir_exists(
    msa_dir: Path,
    *,
    sample_name: str,
    chain_id: str,
    source_label: str,
) -> List[str]:
    non_pairing = msa_dir / "non_pairing.a3m"
    if not non_pairing.exists():
        raise FileNotFoundError(
            f"{sample_name}: missing MSA file {non_pairing} for chain {chain_id} ({source_label})"
        )
    warnings: List[str] = []
    pairing = msa_dir / "pairing.a3m"
    if not pairing.exists():
        warnings.append(f"missing pairing.a3m in {msa_dir}")
    return warnings


def _load_msa_map_index(msa_map_csv: Optional[str]) -> MSAMapIndex:
    if not msa_map_csv:
        return MSAMapIndex(sample_chain={}, role_sequence={}, sequence_only={}, entries=[])

    path = Path(msa_map_csv)
    if not path.exists():
        raise FileNotFoundError(f"--msa_map_csv file not found: {path}")

    sample_chain: Dict[Tuple[str, str], MSAMapEntry] = {}
    role_sequence: Dict[Tuple[str, str], MSAMapEntry] = {}
    sequence_only: Dict[str, MSAMapEntry] = {}
    entries: List[MSAMapEntry] = []

    with open(path, "r", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"--msa_map_csv has no header row: {path}")
        for row_number, row in enumerate(reader, start=2):
            sample_raw = _read_csv_row_value(row, "sample_id") or _read_csv_row_value(row, "sample")
            sample_id = _sanitize_name(sample_raw) if sample_raw else None
            chain_id = _read_csv_row_value(row, "chain_id") or None

            role_raw = _read_csv_row_value(row, "role").lower()
            role: Optional[str] = role_raw or None
            if role is not None and role not in {"target", "binder"}:
                raise ValueError(
                    f"--msa_map_csv row {row_number}: role must be target or binder, got {role_raw!r}"
                )

            seq_raw = _read_csv_row_value(row, "sequence")
            sequence_norm = _normalize_sequence(seq_raw) if seq_raw else None

            has_sample_chain = bool(sample_id and chain_id)
            has_sequence = bool(sequence_norm)
            if not (has_sample_chain or has_sequence):
                raise ValueError(
                    f"--msa_map_csv row {row_number}: provide sample_id+chain_id and/or sequence"
                )

            msa_dir_raw = _read_csv_row_value(row, "msa_dir")
            non_pairing_raw = _read_csv_row_value(row, "non_pairing_path")
            pairing_raw = _read_csv_row_value(row, "pairing_path")

            if msa_dir_raw and (non_pairing_raw or pairing_raw):
                raise ValueError(
                    f"--msa_map_csv row {row_number}: use either msa_dir OR pairing_path+non_pairing_path"
                )
            if not msa_dir_raw and not (non_pairing_raw and pairing_raw):
                raise ValueError(
                    f"--msa_map_csv row {row_number}: missing MSA location; provide msa_dir or pairing_path+non_pairing_path"
                )

            entry = MSAMapEntry(
                row_number=row_number,
                sample_id=sample_id,
                chain_id=chain_id,
                role=role,
                sequence_norm=sequence_norm,
                msa_dir=_resolve_csv_path(msa_dir_raw, path) if msa_dir_raw else None,
                non_pairing_path=_resolve_csv_path(non_pairing_raw, path) if non_pairing_raw else None,
                pairing_path=_resolve_csv_path(pairing_raw, path) if pairing_raw else None,
            )
            entries.append(entry)

            if has_sample_chain:
                sample_key = (sample_id, chain_id)
                if sample_key in sample_chain:
                    prev = sample_chain[sample_key]
                    raise ValueError(
                        f"--msa_map_csv duplicate sample_id+chain_id key {sample_key} "
                        f"(rows {prev.row_number} and {row_number})"
                    )
                sample_chain[sample_key] = entry

            if has_sequence and role is not None:
                role_key = (role, sequence_norm)
                if role_key in role_sequence:
                    prev = role_sequence[role_key]
                    raise ValueError(
                        f"--msa_map_csv duplicate role+sequence key {role_key} "
                        f"(rows {prev.row_number} and {row_number})"
                    )
                role_sequence[role_key] = entry

            if has_sequence and role is None:
                if sequence_norm in sequence_only:
                    prev = sequence_only[sequence_norm]
                    raise ValueError(
                        f"--msa_map_csv duplicate sequence key {sequence_norm[:16]}... "
                        f"(rows {prev.row_number} and {row_number})"
                    )
                sequence_only[sequence_norm] = entry

    return MSAMapIndex(
        sample_chain=sample_chain,
        role_sequence=role_sequence,
        sequence_only=sequence_only,
        entries=entries,
    )


def _resolve_map_entry(
    map_index: MSAMapIndex,
    *,
    sample_id: str,
    chain_id: str,
    role: str,
    sequence_norm: str,
) -> Tuple[Optional[MSAMapEntry], Optional[str]]:
    if not map_index.entries:
        return None, None

    sample_key = (sample_id, chain_id)
    if sample_key in map_index.sample_chain:
        return map_index.sample_chain[sample_key], "sample_chain"

    if sequence_norm:
        role_key = (role, sequence_norm)
        if role_key in map_index.role_sequence:
            return map_index.role_sequence[role_key], "role_sequence"
        if sequence_norm in map_index.sequence_only:
            return map_index.sequence_only[sequence_norm], "sequence"

    return None, None


def _remove_tree(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def _pid_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _read_lock_pid(lock_path: Path) -> Optional[int]:
    try:
        text = lock_path.read_text()
    except OSError:
        return None
    for token in text.split():
        if token.startswith("pid="):
            value = token.split("=", 1)[1].strip()
            if value.isdigit():
                return int(value)
    return None


@contextmanager
def _cache_lock(lock_path: Path, timeout_sec: float = 300.0, stale_sec: float = 1800.0) -> Iterable[None]:
    _ensure_dir(lock_path.parent)
    start = time.time()
    while True:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w") as handle:
                handle.write(f"pid={os.getpid()} ts={time.time()}\n")
            break
        except FileExistsError:
            try:
                stat = lock_path.stat()
                age = time.time() - stat.st_mtime
            except OSError:
                age = 0.0

            lock_pid = _read_lock_pid(lock_path)
            if lock_pid is not None and not _pid_is_running(lock_pid):
                logger.warning("Removing stale cache lock (dead pid=%s): %s", lock_pid, lock_path)
                try:
                    lock_path.unlink()
                except FileNotFoundError:
                    pass
                continue
            if lock_pid is None and age > stale_sec:
                logger.warning("Removing stale cache lock (old/unowned): %s", lock_path)
                try:
                    lock_path.unlink()
                except FileNotFoundError:
                    pass
                continue

            if time.time() - start > timeout_sec:
                raise TimeoutError(f"Timed out waiting for cache lock: {lock_path}")
            time.sleep(0.2)
    try:
        yield
    finally:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def _build_cache_key_context(sequence_norm: str, role: str, args) -> Tuple[str, dict]:
    context = {
        "version": "msa_cache_v1",
        "normalized_sequence": sequence_norm,
        "role": role,
        "provider": args.msa_provider,
        "host_url": args.msa_host_url,
        "pairing_strategy": "colabfold_single_query",
    }
    payload = json.dumps(context, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest(), context


def _cache_entry_ready(msa_dir: Path) -> bool:
    return (msa_dir / "non_pairing.a3m").exists() and (msa_dir / "manifest.json").exists()


def _write_cache_manifest(msa_dir: Path, context: dict, sequence_norm: str) -> None:
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sequence_sha256": _hash_sequence(sequence_norm),
        "provider": context.get("provider"),
        "host_url": context.get("host_url"),
        "role": context.get("role"),
        "cache_context": context,
    }
    with open(msa_dir / "manifest.json", "w") as handle:
        json.dump(manifest, handle, indent=2)


def _materialize_map_entry_dir(
    entry: MSAMapEntry,
    sample_msa_root: Path,
) -> Path:
    if entry.msa_dir is not None:
        return entry.msa_dir

    if entry.non_pairing_path is None or entry.pairing_path is None:
        raise ValueError(f"MSA map row {entry.row_number} is missing explicit MSA file paths")

    key_payload = f"{entry.non_pairing_path}|{entry.pairing_path}"
    key = hashlib.sha256(key_payload.encode("utf-8")).hexdigest()[:16]
    materialized = sample_msa_root / "map_materialized" / key
    if materialized.exists():
        return materialized

    _ensure_dir(materialized)
    shutil.copy2(entry.non_pairing_path, materialized / "non_pairing.a3m")
    shutil.copy2(entry.pairing_path, materialized / "pairing.a3m")
    return materialized


def _set_msa_dir_on_protein(protein: dict, msa_dir: Path) -> None:
    protein["msa"] = {
        "precomputed_msa_dir": str(msa_dir),
        "pairing_db": "uniref100",
    }


def _write_single_chain_msa(
    *,
    sample_msa_root: Path,
    sample_name: str,
    chain_id: str,
    sequence_norm: str,
    tag: str,
) -> Path:
    safe_chain = _sanitize_name(chain_id) or "chain"
    msa_dir = sample_msa_root / f"single_{safe_chain}_{tag}"
    desc = f"{sample_name}_{safe_chain}_{tag}"
    _write_single_sequence_msa(msa_dir, sequence_norm, desc)
    return msa_dir


def _resolve_enabled_chain_msa(
    *,
    sample_name: str,
    chain_id: str,
    role: str,
    sequence_norm: str,
    sample_msa_root: Path,
    args,
    map_index: MSAMapIndex,
) -> Tuple[Optional[Path], Optional[str], Optional[str], Optional[str], List[str]]:
    warnings: List[str] = []

    map_entry, map_strategy = _resolve_map_entry(
        map_index,
        sample_id=sample_name,
        chain_id=chain_id,
        role=role,
        sequence_norm=sequence_norm,
    )
    if map_entry is not None:
        msa_dir = _materialize_map_entry_dir(map_entry, sample_msa_root)
        warnings.extend(
            _validate_msa_dir_exists(
                msa_dir, sample_name=sample_name, chain_id=chain_id, source_label="map"
            )
        )
        return msa_dir, "map", map_strategy, None, warnings

    shared_dir_raw = args.target_msa_shared_dir if role == "target" else args.binder_msa_shared_dir
    if shared_dir_raw:
        msa_dir = Path(shared_dir_raw)
        warnings.extend(
            _validate_msa_dir_exists(
                msa_dir, sample_name=sample_name, chain_id=chain_id, source_label="shared"
            )
        )
        return msa_dir, "shared", "shared", None, warnings

    cache_key: Optional[str] = None
    cache_context: Optional[dict] = None
    cache_dir: Optional[Path] = None
    cache_mode = args.msa_cache_mode
    if cache_mode != "none" and args.msa_cache_dir:
        cache_key, cache_context = _build_cache_key_context(sequence_norm, role, args)
        cache_dir = Path(args.msa_cache_dir) / cache_key

        if cache_mode in {"readwrite", "read"} and _cache_entry_ready(cache_dir):
            warnings.extend(
                _validate_msa_dir_exists(
                    cache_dir, sample_name=sample_name, chain_id=chain_id, source_label="cache"
                )
            )
            return cache_dir, "cache", "cache", cache_key, warnings

    if args.msa_provider != "mmseqs2":
        return None, None, None, cache_key, warnings

    if not sequence_norm:
        warnings.append("empty sequence; cannot fetch from provider")
        return None, None, None, cache_key, warnings

    if cache_mode in {"readwrite", "write"} and cache_dir is not None and cache_context is not None:
        lock_path = Path(args.msa_cache_dir) / ".locks" / f"{cache_key}.lock"
        with _cache_lock(lock_path):
            if cache_mode == "readwrite" and _cache_entry_ready(cache_dir):
                warnings.extend(
                    _validate_msa_dir_exists(
                        cache_dir, sample_name=sample_name, chain_id=chain_id, source_label="cache"
                    )
                )
                return cache_dir, "cache", "cache", cache_key, warnings

            tmp_dir = Path(args.msa_cache_dir) / ".tmp" / f"{cache_key}_{os.getpid()}_{int(time.time()*1000)}"
            _remove_tree(tmp_dir)
            _ensure_dir(tmp_dir.parent)
            _maybe_fetch_colabfold_msa(sequence_norm, tmp_dir, args)
            _write_cache_manifest(tmp_dir, cache_context, sequence_norm)

            if cache_dir.exists():
                _remove_tree(cache_dir)
            _ensure_dir(cache_dir.parent)
            os.replace(tmp_dir, cache_dir)

        warnings.extend(
            _validate_msa_dir_exists(
                cache_dir, sample_name=sample_name, chain_id=chain_id, source_label="fetched"
            )
        )
        return cache_dir, "fetched", "fetch", cache_key, warnings

    fetched_dir = sample_msa_root / "fetched" / f"{role}_{_hash_sequence(sequence_norm)[:16]}"
    _maybe_fetch_colabfold_msa(sequence_norm, fetched_dir, args)
    warnings.extend(
        _validate_msa_dir_exists(
            fetched_dir, sample_name=sample_name, chain_id=chain_id, source_label="fetched"
        )
    )
    return fetched_dir, "fetched", "fetch", cache_key, warnings


def _iter_protein_entries(json_dict: dict) -> Iterable[dict]:
    samples = json_dict if isinstance(json_dict, list) else [json_dict]
    for sample in samples:
        for entry in sample.get("sequences", []):
            protein = entry.get("proteinChain")
            if protein:
                yield protein


def _role_for_chain(
    *,
    chain_id: str,
    sequence_norm: str,
    target_chain_ids: Set[str],
    target_sequences: Set[str],
) -> str:
    if chain_id in target_chain_ids:
        return "target"
    if sequence_norm and sequence_norm in target_sequences:
        return "target"
    return "binder"


def _apply_msa_resolution(
    *,
    json_dict: dict,
    sample_name: str,
    inter_dir: Path,
    args,
    map_index: MSAMapIndex,
) -> List[dict]:
    target_chain_ids = set(_parse_chain_list(args.target_chains))
    target_sequences = {_normalize_sequence(seq) for seq in _load_fasta_sequences(args.target_chain_sequences)}

    sample_msa_root = inter_dir / f"{sample_name}_msa"
    _ensure_dir(sample_msa_root)

    resolution_records: List[dict] = []
    chain_counter = 0
    for protein in _iter_protein_entries(json_dict):
        chain_counter += 1
        label_asym_ids = protein.get("label_asym_id") or []
        chain_id = str(label_asym_ids[0]) if label_asym_ids else f"chain_{chain_counter}"
        sequence_raw = str(protein.get("sequence", "") or "")
        sequence_norm = _normalize_sequence(sequence_raw)
        role = _role_for_chain(
            chain_id=chain_id,
            sequence_norm=sequence_norm,
            target_chain_ids=target_chain_ids,
            target_sequences=target_sequences,
        )

        enabled = _is_role_enabled(args.use_msas, role)
        cache_key = None
        warnings: List[str] = []
        if not enabled:
            msa_dir = _write_single_chain_msa(
                sample_msa_root=sample_msa_root,
                sample_name=sample_name,
                chain_id=chain_id,
                sequence_norm=sequence_norm,
                tag=f"disabled_{chain_counter}",
            )
            source = "single"
            match_strategy = "single"
        else:
            msa_dir, source, match_strategy, cache_key, warnings = _resolve_enabled_chain_msa(
                sample_name=sample_name,
                chain_id=chain_id,
                role=role,
                sequence_norm=sequence_norm,
                sample_msa_root=sample_msa_root,
                args=args,
                map_index=map_index,
            )
            if msa_dir is None:
                if args.msa_missing_policy == "single":
                    msa_dir = _write_single_chain_msa(
                        sample_msa_root=sample_msa_root,
                        sample_name=sample_name,
                        chain_id=chain_id,
                        sequence_norm=sequence_norm,
                        tag=f"fallback_{chain_counter}",
                    )
                    source = "single"
                    match_strategy = "single"
                    warnings.append("fallback to single-sequence due to unresolved MSA")
                else:
                    raise FileNotFoundError(
                        f"{sample_name}: unable to resolve MSA for chain {chain_id} "
                        f"(role={role}); checked map/shared/cache/provider with "
                        f"msa_missing_policy={args.msa_missing_policy}"
                    )

        _set_msa_dir_on_protein(protein, msa_dir)
        resolution_records.append(
            {
                "chain_id": chain_id,
                "role": role,
                "sequence_sha": _hash_sequence(sequence_norm),
                "source": source,
                "cache_key": cache_key,
                "msa_dir": str(msa_dir),
                "warnings": warnings,
                "match_strategy": match_strategy,
            }
        )

    return resolution_records


def _map_entry_msa_dir_ready(entry: MSAMapEntry) -> bool:
    if entry.msa_dir is not None:
        return (entry.msa_dir / "non_pairing.a3m").exists()
    if entry.non_pairing_path is None or entry.pairing_path is None:
        return False
    return entry.non_pairing_path.exists() and entry.pairing_path.exists()


def _validate_msa_args(args) -> None:
    if args.use_msas not in {"both", "target", "binder", "false"}:
        raise ValueError(f"Invalid --use_msas value: {args.use_msas}")
    if args.msa_provider not in {"mmseqs2", "none"}:
        raise ValueError(f"Invalid --msa_provider value: {args.msa_provider}")
    if args.msa_cache_mode not in {"readwrite", "read", "write", "none"}:
        raise ValueError(f"Invalid --msa_cache_mode value: {args.msa_cache_mode}")
    if args.msa_missing_policy not in {"error", "single"}:
        raise ValueError(f"Invalid --msa_missing_policy value: {args.msa_missing_policy}")

    if args.msa_provider == "none" and args.msa_cache_mode == "write":
        raise ValueError("--msa_provider none cannot be combined with --msa_cache_mode write")

    if args.msa_cache_mode != "none" and not args.msa_cache_dir:
        args.msa_cache_dir = os.path.join(args.output, "msa_cache")

    args.use_msa = True


def _collect_chain_role_records_for_preflight(
    *,
    file_path: Path,
    args,
    inter_dir: Path,
    chain_sequence_overrides: Dict[str, str],
) -> List[Tuple[str, str, str]]:
    sample_name = _sanitize_name(file_path.stem)
    cif_path = file_path
    cleanup_cif = False
    if file_path.suffix.lower() == ".pdb":
        cif_path = inter_dir / f"preflight_{sample_name}.cif"
        pdb_to_cif(str(file_path), str(cif_path), entry_id=sample_name)
        cleanup_cif = True

    try:
        json_dict, atom_array_src = _parse_structure_to_json(
            cif_path=cif_path,
            sample_name=sample_name,
            assembly_id=args.assembly_id,
            altloc=args.altloc,
            output_json=None,
        )
        chain_sequences = _extract_chain_sequences(atom_array_src)
        _apply_chain_sequence_overrides(
            json_dict,
            chain_sequences,
            chain_sequence_overrides,
            sample_name,
        )
        _replace_unknown_residues(json_dict, sample_name)

        target_chain_ids = set(_parse_chain_list(args.target_chains))
        target_sequences = {_normalize_sequence(seq) for seq in _load_fasta_sequences(args.target_chain_sequences)}

        records: List[Tuple[str, str, str]] = []
        for protein in _iter_protein_entries(json_dict):
            label_asym_ids = protein.get("label_asym_id") or []
            chain_id = str(label_asym_ids[0]) if label_asym_ids else "?"
            sequence_norm = _normalize_sequence(str(protein.get("sequence", "") or ""))
            role = _role_for_chain(
                chain_id=chain_id,
                sequence_norm=sequence_norm,
                target_chain_ids=target_chain_ids,
                target_sequences=target_sequences,
            )
            records.append((chain_id, role, sequence_norm))
        return records
    finally:
        if cleanup_cif:
            try:
                os.remove(cif_path)
            except OSError:
                pass


def _validate_msa_preflight(
    *,
    args,
    input_files: List[Path],
    map_index: MSAMapIndex,
    inter_dir: Path,
    chain_sequence_overrides: Dict[str, str],
) -> None:
    for entry in map_index.entries:
        if not _map_entry_msa_dir_ready(entry):
            raise FileNotFoundError(
                f"--msa_map_csv row {entry.row_number}: MSA location is missing required files"
            )

    for shared_dir_raw, role in (
        (args.target_msa_shared_dir, "target"),
        (args.binder_msa_shared_dir, "binder"),
    ):
        if not shared_dir_raw:
            continue
        shared_dir = Path(shared_dir_raw)
        if not (shared_dir / "non_pairing.a3m").exists():
            raise FileNotFoundError(
                f"--{role}_msa_shared_dir missing non_pairing.a3m: {shared_dir}"
            )

    if args.msa_provider != "none":
        return

    unresolved: List[str] = []
    coverage_counts = {"map": 0, "shared": 0, "cache": 0}
    for file_path in input_files:
        sample_name = _sanitize_name(file_path.stem)
        records = _collect_chain_role_records_for_preflight(
            file_path=file_path,
            args=args,
            inter_dir=inter_dir,
            chain_sequence_overrides=chain_sequence_overrides,
        )
        for chain_id, role, sequence_norm in records:
            if not _is_role_enabled(args.use_msas, role):
                continue

            entry, _ = _resolve_map_entry(
                map_index,
                sample_id=sample_name,
                chain_id=chain_id,
                role=role,
                sequence_norm=sequence_norm,
            )
            if entry is not None:
                coverage_counts["map"] += 1
                continue

            shared_dir_raw = args.target_msa_shared_dir if role == "target" else args.binder_msa_shared_dir
            if shared_dir_raw:
                coverage_counts["shared"] += 1
                continue

            if args.msa_cache_mode in {"read", "readwrite"} and args.msa_cache_dir and sequence_norm:
                cache_key, _ = _build_cache_key_context(sequence_norm, role, args)
                cache_dir = Path(args.msa_cache_dir) / cache_key
                if _cache_entry_ready(cache_dir):
                    coverage_counts["cache"] += 1
                    continue

            unresolved.append(f"{sample_name}:{chain_id}:{role}")

    if unresolved and args.msa_missing_policy == "error":
        preview = ", ".join(unresolved[:10])
        suffix = "" if len(unresolved) <= 10 else f" ... (+{len(unresolved) - 10} more)"
        raise ValueError(
            "--msa_provider none requires full pre-resolved coverage for enabled roles when "
            "--msa_missing_policy=error. "
            f"Unresolved chains: {preview}{suffix}"
        )
    if unresolved and args.msa_missing_policy == "single":
        logger.warning(
            "MSA preflight: %d enabled-role chains unresolved; they will fall back to single-sequence "
            "because --msa_missing_policy=single and --msa_provider=none",
            len(unresolved),
        )

    logger.info(
        "MSA preflight coverage (provider=none): map=%d shared=%d cache=%d",
        coverage_counts["map"],
        coverage_counts["shared"],
        coverage_counts["cache"],
    )


def _write_msa_resolution(sample_out_dir: Path, records: List[dict]) -> None:
    with open(sample_out_dir / "msa_resolution.json", "w") as handle:
        json.dump(records, handle, indent=2)


def _write_msa_resolution_summary(output_dir: Path, results: List[ScoreResult]) -> None:
    summary = {
        "total_chains": 0,
        "chains_by_role": {"target": 0, "binder": 0},
        "source_counts": {"single": 0, "map": 0, "shared": 0, "cache": 0, "fetched": 0},
        "fetch_count": 0,
        "cache_hits": 0,
        "cache_misses": 0,
        "single_policy_fallbacks": 0,
    }
    for result in results:
        for rec in result.msa_resolution or []:
            summary["total_chains"] += 1
            role = rec.get("role")
            if role in summary["chains_by_role"]:
                summary["chains_by_role"][role] += 1
            source = rec.get("source")
            if source in summary["source_counts"]:
                summary["source_counts"][source] += 1
            if source == "fetched":
                summary["fetch_count"] += 1
                summary["cache_misses"] += 1
            if source == "cache":
                summary["cache_hits"] += 1
            warnings = rec.get("warnings", [])
            if any("fallback to single-sequence" in str(item) for item in warnings):
                summary["single_policy_fallbacks"] += 1

    with open(output_dir / "msa_resolution_summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)


def _maybe_fetch_colabfold_msa(
    sequence: str,
    msa_dir: Path,
    args,
) -> None:
    from protenixscore.msa_colabfold import ColabFoldMSAConfig, ensure_msa_dir

    cfg = ColabFoldMSAConfig(
        host_url=args.msa_host_url,
        use_env=args.msa_use_env,
        use_filter=args.msa_use_filter,
    )
    ensure_msa_dir(sequence=sequence, out_dir=msa_dir, cfg=cfg, force=args.msa_cache_refresh)


def _write_single_sequence_msa(msa_dir: Path, sequence: str, description: str) -> None:
    _ensure_dir(msa_dir)
    header = f">{description}"
    body = sequence.strip()
    content = f"{header}\n{body}\n"
    for fname in ("non_pairing.a3m", "pairing.a3m"):
        with open(msa_dir / fname, "w") as f:
            f.write(content)


def _build_chain_id_map(
    atom_array_src: AtomArray, atom_array_internal: AtomArray
) -> Dict[str, Dict[str, str]]:
    source_order = _build_chain_order_by_entity(atom_array_src)

    internal_chain_starts = get_chain_starts(atom_array_internal, add_exclusive_stop=False)
    internal_chain_ids = atom_array_internal.chain_id[internal_chain_starts].tolist()

    if len(source_order) != len(internal_chain_ids):
        raise ValueError(
            "Chain count mismatch between source and internal atom arrays: "
            f"{len(source_order)} vs {len(internal_chain_ids)}"
        )

    internal_to_source = dict(zip(internal_chain_ids, source_order))
    source_to_internal = {v: k for k, v in internal_to_source.items()}

    return {
        "internal_order": internal_chain_ids,
        "source_order": source_order,
        "internal_to_source": internal_to_source,
        "source_to_internal": source_to_internal,
    }


def _build_source_coord_map(atom_array_src: AtomArray) -> Dict[Tuple[str, int, str], np.ndarray]:
    coord_map: Dict[Tuple[str, int, str], np.ndarray] = {}
    for atom in atom_array_src:
        key = (atom.chain_id, int(atom.res_id), atom.atom_name)
        coord_map[key] = atom.coord
    return coord_map


def _map_coords_to_internal(
    atom_array_internal: AtomArray,
    chain_id_map: Dict[str, Dict[str, str]],
    source_coord_map: Dict[Tuple[str, int, str], np.ndarray],
    missing_policy: str,
) -> Tuple[np.ndarray, List[Tuple[str, int, str]]]:
    coords = np.zeros((len(atom_array_internal), 3), dtype=np.float32)
    missing: List[Tuple[str, int, str]] = []

    internal_to_source = chain_id_map["internal_to_source"]

    for idx, atom in enumerate(atom_array_internal):
        source_chain = internal_to_source.get(atom.chain_id)
        if source_chain is None:
            missing.append((atom.chain_id, int(atom.res_id), atom.atom_name))
            continue
        key = (source_chain, int(atom.res_id), atom.atom_name)
        if key in source_coord_map:
            coords[idx] = source_coord_map[key]
        else:
            missing.append(key)
            if missing_policy == "reference":
                coords[idx] = atom.coord
            elif missing_policy == "zero":
                coords[idx] = np.zeros(3, dtype=np.float32)
            elif missing_policy == "error":
                raise ValueError(f"Missing atom coordinate for {key}")
            else:
                coords[idx] = atom.coord

    return coords, missing


def _configure_device(device: str) -> None:
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    elif device.startswith("cuda:"):
        gpu_id = device.split(":", 1)[1]
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id


def _load_runner(args) -> object:
    runner = get_default_runner(
        seeds=[101],
        n_cycle=10,
        n_step=200,
        n_sample=1,
        dtype=args.dtype,
        model_name=args.model_name,
        use_msa=args.use_msa,
        trimul_kernel="torch",
        triatt_kernel="torch",
    )
    if args.checkpoint_dir:
        runner.configs.load_checkpoint_dir = args.checkpoint_dir
        runner.load_checkpoint()
    runner.configs.use_msa = args.use_msa
    runner.configs.num_workers = args.num_workers
    if hasattr(runner.configs, "esm"):
        runner.configs.esm.enable = bool(args.use_esm)
    return runner


def _write_chain_id_map(path: Path, chain_id_map: Dict[str, Dict[str, str]]) -> None:
    with open(path, "w") as f:
        json.dump(chain_id_map, f, indent=2)


def _write_summary(path: Path, summary: dict) -> None:
    summary_copy = _safe_round_values(summary.copy())
    with open(path, "w") as f:
        json.dump(_json_safe(summary_copy), f, indent=4)


def _write_full(path: Path, full_data: dict) -> None:
    from runner.dumper import get_clean_full_confidence

    full_copy = get_clean_full_confidence(full_data.copy())
    save_json(full_copy, path, indent=4)


def _json_safe(value):
    if isinstance(value, torch.Tensor):
        arr = value
        if arr.dtype == torch.bfloat16:
            arr = arr.float()
        return arr.cpu().numpy().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def _safe_round_values(data: dict, recursive: bool = True):
    for k, v in data.items():
        if isinstance(v, torch.Tensor):
            if v.dtype == torch.bool:
                data[k] = v.cpu().numpy().tolist()
            else:
                if v.dtype == torch.bfloat16:
                    v = v.float()
                data[k] = np.round(v.cpu().numpy(), 2)
        elif isinstance(v, (np.floating, np.integer)):
            data[k] = float(v)
        elif isinstance(v, np.ndarray):
            if v.dtype == np.bool_:
                data[k] = v.tolist()
            else:
                data[k] = np.round(v, 2)
        elif isinstance(v, list):
            try:
                arr = np.array(v)
                if arr.dtype == np.bool_:
                    data[k] = v
                else:
                    data[k] = list(np.round(arr, 2))
            except Exception:
                data[k] = v
        elif isinstance(v, dict) and recursive:
            data[k] = _safe_round_values(v, recursive)
    return data


def _calc_ipsae_d0_array(n0res_per_residue: torch.Tensor, pair_type: str = "protein") -> torch.Tensor:
    """Vectorized d0 for ipSAE, following AF3Score's TM-score-like normalization.

    n0res_per_residue is the count of interface partners (per residue in chain1)
    passing the PAE cutoff.
    """
    if pair_type not in {"protein", "nucleic_acid"}:
        pair_type = "protein"

    # L is clamped at a minimum of 27.0 (AF3Score convention).
    L = torch.clamp(n0res_per_residue.to(dtype=torch.float32), min=27.0)
    min_value = 2.0 if pair_type == "nucleic_acid" else 1.0

    # d0 = 1.24 * (L-15)^(1/3) - 1.8
    d0 = 1.24 * torch.pow(L - 15.0, 1.0 / 3.0) - 1.8
    return torch.clamp(d0, min=min_value)


def _calculate_ipsae_for_chain_pair(
    token_pair_pae: torch.Tensor,
    token_asym_id: torch.Tensor,
    chain_i: int,
    chain_j: int,
    pae_cutoff: float,
    pair_type: str = "protein",
) -> float:
    """Compute directional ipSAE for chain_i -> chain_j.

    This matches AF3Score's definition:
    - valid pairs are those with PAE < pae_cutoff
    - per-residue PTM-like score averaged over valid pairs
    - final score is max over residues in chain_i
    """
    idx_i = torch.nonzero(token_asym_id == int(chain_i), as_tuple=False).squeeze(-1)
    idx_j = torch.nonzero(token_asym_id == int(chain_j), as_tuple=False).squeeze(-1)
    if idx_i.numel() == 0 or idx_j.numel() == 0:
        return 0.0

    sub_pae = token_pair_pae.index_select(0, idx_i).index_select(1, idx_j).to(dtype=torch.float32)
    if sub_pae.numel() == 0:
        return 0.0

    valid_mask = sub_pae < float(pae_cutoff)
    n0res = valid_mask.sum(dim=1)  # (N_i,)

    d0 = _calc_ipsae_d0_array(n0res, pair_type=pair_type)  # (N_i,)
    ptm_matrix = 1.0 / (1.0 + torch.square(sub_pae / d0[:, None]))

    masked_sum = (ptm_matrix * valid_mask.to(dtype=ptm_matrix.dtype)).sum(dim=1)
    denom = torch.clamp(n0res.to(dtype=torch.float32), min=1.0)
    ipsae_per_residue = masked_sum / denom
    score = float(ipsae_per_residue.max().item()) if ipsae_per_residue.numel() else 0.0
    return score


def _calculate_ipsae_metrics(
    full_data: dict,
    chain_id_map: Dict[str, Dict[str, str]],
    pae_cutoff: float,
    target_source_chains: Optional[List[str]] = None,
) -> Dict[str, object]:
    """Calculate ipSAE metrics from Protenix full_data and chain mapping.

    Returns a dict containing:
    - ipsae_by_chain_pair: {"A_B": 0.83, "B_A": 0.81, ...} (source chain IDs)
    - ipsae_max: max over all directional chain pairs
    - ipsae_interface_max: if target_source_chains given, max over target<->binder directions
    - ipsae_target_to_binder: max over target->binder
    - ipsae_binder_to_target: max over binder->target
    """
    if not full_data:
        return {}

    token_asym_id = full_data.get("token_asym_id")
    token_pair_pae = full_data.get("token_pair_pae")
    if token_asym_id is None or token_pair_pae is None:
        return {}

    if not isinstance(token_asym_id, torch.Tensor):
        token_asym_id = torch.as_tensor(token_asym_id)
    if not isinstance(token_pair_pae, torch.Tensor):
        token_pair_pae = torch.as_tensor(token_pair_pae)

    # Optionally drop tokens that aren't associated with residue frames.
    token_mask = full_data.get("token_has_frame")
    if token_mask is not None:
        if not isinstance(token_mask, torch.Tensor):
            token_mask = torch.as_tensor(token_mask)
        token_mask = token_mask.to(dtype=torch.bool)
        token_asym_id = token_asym_id[token_mask]
        token_pair_pae = token_pair_pae[token_mask][:, token_mask]

    token_asym_id = token_asym_id.to(dtype=torch.long)
    n_chain = len(chain_id_map.get("internal_order", []))
    if n_chain <= 1:
        return {}

    internal_order = chain_id_map["internal_order"]
    internal_to_source = chain_id_map["internal_to_source"]
    idx_to_source = []
    for idx in range(n_chain):
        internal_id = internal_order[idx]
        idx_to_source.append(internal_to_source.get(internal_id, str(internal_id)))

    ipsae_by_pair: Dict[str, float] = {}
    for i in range(n_chain):
        for j in range(n_chain):
            if i == j:
                continue
            src_i = idx_to_source[i]
            src_j = idx_to_source[j]
            key = f"{src_i}_{src_j}"
            ipsae_by_pair[key] = _calculate_ipsae_for_chain_pair(
                token_pair_pae=token_pair_pae,
                token_asym_id=token_asym_id,
                chain_i=i,
                chain_j=j,
                pae_cutoff=pae_cutoff,
                pair_type="protein",
            )

    values = list(ipsae_by_pair.values())
    ipsae_max = float(max(values)) if values else 0.0

    result: Dict[str, object] = {
        "ipsae_pae_cutoff": float(pae_cutoff),
        "ipsae_by_chain_pair": ipsae_by_pair,
        "ipsae_max": ipsae_max,
    }

    targets = set(target_source_chains or [])
    if targets:
        all_chains = set(idx_to_source)
        binders = all_chains - targets
        t2b_vals = [v for k, v in ipsae_by_pair.items() if k.split("_", 1)[0] in targets and k.split("_", 1)[1] in binders]
        b2t_vals = [v for k, v in ipsae_by_pair.items() if k.split("_", 1)[0] in binders and k.split("_", 1)[1] in targets]
        t2b = float(max(t2b_vals)) if t2b_vals else 0.0
        b2t = float(max(b2t_vals)) if b2t_vals else 0.0
        result["ipsae_target_to_binder"] = t2b
        result["ipsae_binder_to_target"] = b2t
        result["ipsae_interface_max"] = float(max(t2b, b2t))
    else:
        # If no target/binder split is provided, "interface" reduces to the global max.
        result["ipsae_interface_max"] = ipsae_max

    return result


def _score_single(
    file_path: Path,
    runner,
    args,
    output_dir: Path,
    inter_dir: Path,
    chain_sequence_overrides: Dict[str, str],
) -> ScoreResult:
    start_total = time.perf_counter()
    sample_name = _sanitize_name(file_path.stem)
    sample_out_dir = output_dir / sample_name
    if sample_out_dir.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {sample_out_dir}")
    _ensure_dir(sample_out_dir)

    cif_path = file_path
    cleanup_cif = False
    if file_path.suffix.lower() == ".pdb":
        if not args.convert_pdb_to_cif:
            raise ValueError("PDB input requires --convert_pdb_to_cif")
        cif_path = inter_dir / f"{sample_name}.cif"
        pdb_to_cif(str(file_path), str(cif_path), entry_id=sample_name)
        cleanup_cif = not args.keep_intermediate

    json_path = inter_dir / f"{sample_name}.json"
    json_dict, atom_array_src = _parse_structure_to_json(
        cif_path=cif_path,
        sample_name=sample_name,
        assembly_id=args.assembly_id,
        altloc=args.altloc,
        output_json=json_path,
    )
    chain_sequences = _extract_chain_sequences(atom_array_src)
    _apply_chain_sequence_overrides(
        json_dict,
        chain_sequences,
        chain_sequence_overrides,
        sample_name,
    )
    _replace_unknown_residues(json_dict, sample_name)
    map_index = getattr(
        args,
        "_msa_map_index",
        MSAMapIndex(sample_chain={}, role_sequence={}, sequence_only={}, entries=[]),
    )
    msa_resolution = _apply_msa_resolution(
        json_dict=json_dict,
        sample_name=sample_name,
        inter_dir=inter_dir,
        args=args,
        map_index=map_index,
    )
    _write_msa_resolution(sample_out_dir, msa_resolution)
    with open(json_path, "w") as f:
        json.dump(json_dict, f, indent=2)

    # Protenix v1+ expects InferenceDataset(configs) where configs carries
    # input_json_path/dump_dir/use_msa. Older Protenix versions accepted these
    # as constructor kwargs; keep things compatible by populating configs here.
    runner.configs.input_json_path = str(json_path)
    runner.configs.dump_dir = str(output_dir)
    runner.configs.use_msa = args.use_msa
    dataset = InferenceDataset(runner.configs)

    if len(dataset.inputs) != 1:
        raise ValueError("Expected a single sample in generated JSON")

    start_prep = time.perf_counter()
    data, atom_array_internal, _ = dataset.process_one(dataset.inputs[0])

    if args.max_tokens is not None and data["N_token"].item() > args.max_tokens:
        raise ValueError(
            f"{sample_name}: N_token {data['N_token'].item()} exceeds max_tokens {args.max_tokens}"
        )
    if args.max_atoms is not None and data["N_atom"].item() > args.max_atoms:
        raise ValueError(
            f"{sample_name}: N_atom {data['N_atom'].item()} exceeds max_atoms {args.max_atoms}"
        )

    chain_id_map = _build_chain_id_map(atom_array_src, atom_array_internal)
    source_coord_map = _build_source_coord_map(atom_array_src)

    coords_np, missing_atoms = _map_coords_to_internal(
        atom_array_internal,
        chain_id_map,
        source_coord_map,
        args.missing_atom_policy,
    )

    if missing_atoms:
        logger.warning(
            "%s: %d atoms missing in input structure; policy=%s",
            sample_name,
            len(missing_atoms),
            args.missing_atom_policy,
        )
        missing_path = sample_out_dir / "missing_atoms.json"
        with open(missing_path, "w") as f:
            json.dump(
                [
                    {"chain_id": k[0], "res_id": k[1], "atom_name": k[2]}
                    for k in missing_atoms
                ],
                f,
                indent=2,
            )

    coords = torch.tensor(coords_np, dtype=torch.float32).unsqueeze(0)

    data = to_device(data, runner.device)
    coords = coords.to(device=runner.device, dtype=data["input_feature_dict"]["ref_pos"].dtype)

    eval_precision = {
        "fp32": torch.float32,
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }[runner.configs.dtype]

    enable_amp = (
        torch.autocast(device_type="cuda", dtype=eval_precision)
        if torch.cuda.is_available()
        else nullcontext()
    )

    start_model = time.perf_counter()
    with torch.no_grad():
        with enable_amp:
            pred_dict, _, _ = runner.model(
                input_feature_dict=data["input_feature_dict"],
                label_full_dict=None,
                label_dict=None,
                mode="inference",
                score_only=True,
                x_pred_coords=coords,
            )
    end_model = time.perf_counter()

    summary = pred_dict["summary_confidence"][0]
    full_data_raw = pred_dict["full_data"][0] if "full_data" in pred_dict else None

    if getattr(args, "write_ipsae", False) and full_data_raw is not None:
        target_chains = _parse_chain_list(args.target_chains) if getattr(args, "target_chains", None) else []
        ipsae_metrics = _calculate_ipsae_metrics(
            full_data=full_data_raw,
            chain_id_map=chain_id_map,
            pae_cutoff=float(getattr(args, "ipsae_pae_cutoff", 10.0)),
            target_source_chains=target_chains,
        )
        summary.update(ipsae_metrics)

    full_data = full_data_raw if args.write_full_confidence else None

    _write_chain_id_map(sample_out_dir / "chain_id_map.json", chain_id_map)

    if args.write_summary_confidence:
        _write_summary(sample_out_dir / "summary_confidence.json", summary)

    if args.write_full_confidence and full_data is not None:
        _write_full(sample_out_dir / "full_confidence.json", full_data)

    if cleanup_cif:
        try:
            os.remove(cif_path)
        except OSError:
            pass

    end_total = time.perf_counter()
    prep_seconds = start_model - start_total
    model_seconds = end_model - start_model
    total_seconds = end_total - start_total
    logger.info(
        "%s: timing prep=%.2fs model=%.2fs total=%.2fs",
        sample_name,
        prep_seconds,
        model_seconds,
        total_seconds,
    )

    return ScoreResult(
        sample_name=sample_name,
        summary=summary,
        full_data=full_data,
        output_dir=sample_out_dir,
        msa_resolution=msa_resolution,
        prep_seconds=prep_seconds,
        model_seconds=model_seconds,
        total_seconds=total_seconds,
    )


def _write_aggregate_csv(results: List[ScoreResult], csv_path: Path) -> None:
    if not results:
        return
    rows = []
    for result in results:
        summary = result.summary
        row = {
            "sample": result.sample_name,
            "plddt": float(summary.get("plddt", 0.0)),
            "ptm": float(summary.get("ptm", 0.0)),
            "iptm": float(summary.get("iptm", 0.0)),
            "ranking_score": float(summary.get("ranking_score", 0.0)),
            "ipsae_interface_max": float(summary.get("ipsae_interface_max", 0.0)),
            "ipsae_target_to_binder": float(summary.get("ipsae_target_to_binder", 0.0)),
            "ipsae_binder_to_target": float(summary.get("ipsae_binder_to_target", 0.0)),
        }
        rows.append(row)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_score(args) -> None:
    if not args.score_only:
        raise ValueError("Only score-only mode is supported.")

    _validate_msa_args(args)

    _configure_device(args.device)

    output_dir = Path(args.output)
    _ensure_dir(output_dir)

    input_files = collect_input_files(args.input, args.recursive, args.glob)
    if not input_files:
        raise FileNotFoundError("No input structures found")

    if args.batch_size != 1:
        logger.warning("batch_size > 1 is not supported yet; using 1")

    inter_dir, temp_dir = _prepare_intermediate_dirs(output_dir, args.keep_intermediate, args.intermediate_dir)
    try:
        chain_sequence_overrides = _parse_chain_sequence_overrides(
            getattr(args, "chain_sequence", [])
        )
        map_index = _load_msa_map_index(getattr(args, "msa_map_csv", None))
        args._msa_map_index = map_index

        if args.msa_cache_mode != "none" and args.msa_cache_dir:
            _ensure_dir(Path(args.msa_cache_dir))

        if getattr(args, "validate_msa_inputs", True):
            _validate_msa_preflight(
                args=args,
                input_files=input_files,
                map_index=map_index,
                inter_dir=inter_dir,
                chain_sequence_overrides=chain_sequence_overrides,
            )

        logger.info(
            "MSA config: use_msas=%s provider=%s cache_mode=%s cache_dir=%s map_csv=%s "
            "target_shared=%s binder_shared=%s missing_policy=%s validate=%s",
            args.use_msas,
            args.msa_provider,
            args.msa_cache_mode,
            args.msa_cache_dir,
            args.msa_map_csv,
            args.target_msa_shared_dir,
            args.binder_msa_shared_dir,
            args.msa_missing_policy,
            args.validate_msa_inputs,
        )

        runner = _load_runner(args)

        failed_records: List[str] = []
        results: List[ScoreResult] = []

        for file_path in input_files:
            try:
                result = _score_single(
                    file_path=file_path,
                    runner=runner,
                    args=args,
                    output_dir=output_dir,
                    inter_dir=inter_dir,
                    chain_sequence_overrides=chain_sequence_overrides,
                )
                results.append(result)
            except Exception:
                logger.exception("Failed to score %s", file_path)
                failed_records.append(
                    f"{file_path}:\n{traceback.format_exc()}"
                )

        if failed_records:
            failed_path = Path(args.failed_log)
            _ensure_dir(failed_path.parent)
            with open(failed_path, "w") as f:
                f.write("\n".join(failed_records))

        _write_aggregate_csv(results, Path(args.aggregate_csv))
        _write_msa_resolution_summary(output_dir, results)
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()
