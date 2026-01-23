import csv
import hashlib
import json
import logging
import os
import tempfile
import time
import traceback
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

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
    prep_seconds: Optional[float] = None
    model_seconds: Optional[float] = None
    total_seconds: Optional[float] = None


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


def _set_precomputed_msa(protein: dict, msa_dir: Path, sample_name: str, chain_id: str) -> None:
    non_pairing = msa_dir / "non_pairing.a3m"
    if not non_pairing.exists():
        raise FileNotFoundError(
            f"{sample_name}: missing MSA file {non_pairing} for chain {chain_id}"
        )
    protein["msa"] = {
        "precomputed_msa_dir": str(msa_dir),
        "pairing_db": "uniref100",
    }


def _maybe_fetch_colabfold_msa(
    sequence: str,
    msa_dir: Path,
    args,
) -> None:
    from protenixscore.msa_colabfold import ColabFoldMSAConfig, ensure_msa_dir

    cfg = ColabFoldMSAConfig(
        host_url=args.msa_host,
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


def _inject_precomputed_msa(
    json_dict: dict,
    msa_base_dir: Path,
    sample_name: str,
) -> int:
    if isinstance(json_dict, list):
        samples = json_dict
    else:
        samples = [json_dict]
    msa_count = 0
    for sample in samples:
        sequences = sample.get("sequences", [])
        protein_idx = 0
        for entry in sequences:
            protein = entry.get("proteinChain")
            if not protein:
                continue
            protein_idx += 1
            if "msa" in protein:
                continue
            msa_dir = msa_base_dir / f"entity_{protein_idx}"
            non_pairing = msa_dir / "non_pairing.a3m"
            if not non_pairing.exists():
                raise FileNotFoundError(
                    f"{sample_name}: missing MSA file {non_pairing}"
                )
            protein["msa"] = {
                "precomputed_msa_dir": str(msa_dir),
                "pairing_db": "uniref100",
            }
            msa_count += 1
    if msa_count > 0:
        logger.info(
            "%s: injected precomputed MSA for %d protein entities",
            sample_name,
            msa_count,
        )
    return msa_count


def _inject_single_sequence_msa(json_dict: dict, msa_base_dir: Path, sample_name: str) -> int:
    if isinstance(json_dict, list):
        samples = json_dict
    else:
        samples = [json_dict]
    msa_count = 0
    for sample in samples:
        sequences = sample.get("sequences", [])
        protein_idx = 0
        for entry in sequences:
            protein = entry.get("proteinChain")
            if not protein:
                continue
            protein_idx += 1
            if "msa" in protein:
                continue
            sequence = protein.get("sequence", "")
            if not sequence:
                continue
            msa_dir = msa_base_dir / f"entity_{protein_idx}"
            desc = f"{sample_name}_entity_{protein_idx}"
            _write_single_sequence_msa(msa_dir, sequence, desc)
            protein["msa"] = {
                "precomputed_msa_dir": str(msa_dir),
                "pairing_db": "uniref100",
            }
            msa_count += 1
    if msa_count > 0:
        logger.info("%s: injected single-sequence MSA for %d protein entities", sample_name, msa_count)
    return msa_count


def _resolve_msa_base(msa_path: Path, sample_name: str) -> Path:
    per_sample = msa_path / sample_name
    if per_sample.exists():
        return per_sample
    return msa_path


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
    if args.use_msa:
        target_chain_ids = set(_parse_chain_list(args.target_chains))
        target_sequences = set(_load_fasta_sequences(args.target_chain_sequences))
        use_target_strategy = bool(
            target_chain_ids
            or target_sequences
            or args.target_msa_path
            or args.msa_cache_dir
            or args.msa_source != "none"
        )

        if not use_target_strategy:
            if args.msa_path:
                msa_root = _resolve_msa_base(Path(args.msa_path), sample_name)
                _inject_precomputed_msa(json_dict, msa_root, sample_name)
            else:
                msa_base_dir = inter_dir / f"{sample_name}_msa"
                _ensure_dir(msa_base_dir)
                _inject_single_sequence_msa(json_dict, msa_base_dir, sample_name)
        else:
            if args.msa_path:
                logger.warning(
                    "%s: --msa_path ignored because target MSA strategy is enabled",
                    sample_name,
                )

            target_msa_base = Path(args.target_msa_path) if args.target_msa_path else None
            msa_cache_dir = Path(args.msa_cache_dir) if args.msa_cache_dir else None
            if args.msa_source == "colabfold" and msa_cache_dir is None and target_msa_base is None:
                raise ValueError(
                    "--msa_source colabfold requires --msa_cache_dir or --target_msa_path"
                )

            msa_base_dir = inter_dir / f"{sample_name}_msa"
            _ensure_dir(msa_base_dir)

            target_idx = 0
            protein_idx = 0
            samples = json_dict if isinstance(json_dict, list) else [json_dict]
            for sample in samples:
                sequences = sample.get("sequences", [])
                for entry in sequences:
                    protein = entry.get("proteinChain")
                    if not protein:
                        continue
                    protein_idx += 1
                    label_asym_ids = protein.get("label_asym_id") or []
                    chain_id = label_asym_ids[0] if label_asym_ids else "?"
                    sequence = protein.get("sequence", "")
                    is_target = False
                    if chain_id in target_chain_ids:
                        is_target = True
                    elif sequence and sequence in target_sequences:
                        is_target = True

                    if is_target:
                        target_idx += 1
                        if target_msa_base is not None:
                            msa_dir = target_msa_base / f"entity_{target_idx}"
                            _set_precomputed_msa(protein, msa_dir, sample_name, chain_id)
                        elif msa_cache_dir is not None:
                            msa_dir = msa_cache_dir / _hash_sequence(sequence)
                            if not (msa_dir / "non_pairing.a3m").exists():
                                if args.msa_source == "colabfold":
                                    _maybe_fetch_colabfold_msa(sequence, msa_dir, args)
                                else:
                                    raise FileNotFoundError(
                                        f"{sample_name}: target MSA cache miss for chain {chain_id} "
                                        f"and msa_source={args.msa_source}"
                                    )
                            _set_precomputed_msa(protein, msa_dir, sample_name, chain_id)
                        else:
                            raise ValueError(
                                f"{sample_name}: target chain {chain_id} requires "
                                "--target_msa_path or --msa_cache_dir"
                            )
                    else:
                        if args.binder_msa_mode == "single":
                            msa_dir = msa_base_dir / f"entity_{protein_idx}"
                            _write_single_sequence_msa(msa_dir, sequence, f"{sample_name}_entity_{protein_idx}")
                            protein["msa"] = {
                                "precomputed_msa_dir": str(msa_dir),
                                "pairing_db": "uniref100",
                            }
                        elif args.binder_msa_mode == "none":
                            continue
                        else:
                            raise ValueError(f"Unknown binder_msa_mode: {args.binder_msa_mode}")

        with open(json_path, "w") as f:
            json.dump(json_dict, f, indent=2)

    dataset = InferenceDataset(
        input_json_path=str(json_path),
        dump_dir=str(output_dir),
        use_msa=args.use_msa,
        configs=runner.configs,
    )

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
    full_data = pred_dict["full_data"][0] if args.write_full_confidence else None

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
        }
        rows.append(row)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_score(args) -> None:
    if not args.score_only:
        raise ValueError("Only score-only mode is supported.")

    if args.use_msa and (args.target_chains or args.target_chain_sequences):
        if args.msa_source == "none" and args.msa_cache_dir is None and args.target_msa_path is None:
            args.msa_source = "colabfold"
        if args.msa_source == "colabfold" and args.msa_cache_dir is None and args.target_msa_path is None:
            args.msa_cache_dir = os.path.join(args.output, "msa_cache")

    _configure_device(args.device)

    output_dir = Path(args.output)
    _ensure_dir(output_dir)

    input_files = collect_input_files(args.input, args.recursive, args.glob)
    if not input_files:
        raise FileNotFoundError("No input structures found")

    if args.batch_size != 1:
        logger.warning("batch_size > 1 is not supported yet; using 1")

    inter_dir, temp_dir = _prepare_intermediate_dirs(output_dir, args.keep_intermediate, args.intermediate_dir)

    runner = _load_runner(args)

    chain_sequence_overrides = _parse_chain_sequence_overrides(
        getattr(args, "chain_sequence", [])
    )

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
        except Exception as exc:
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

    if temp_dir is not None:
        temp_dir.cleanup()
