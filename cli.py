import argparse
import os
from typing import List, Optional

from protenixscore.score import run_score


DEFAULT_GLOBS = "*.pdb,*.cif"
DEFAULT_CHECKPOINT_DIR = os.environ.get("PROTENIX_CHECKPOINT_DIR")


def _parse_globs(value: str) -> List[str]:
    patterns = []
    for item in value.split(","):
        item = item.strip()
        if item:
            patterns.append(item)
    return patterns or ["*.pdb", "*.cif"]


def _prompt(text: str, default: Optional[str] = None) -> str:
    if default:
        prompt = f"{text} [{default}]: "
    else:
        prompt = f"{text}: "
    value = input(prompt).strip()
    return value or (default or "")


def _prompt_bool(text: str, default: bool) -> bool:
    default_str = "y" if default else "n"
    value = _prompt(f"{text} (y/n)", default_str).lower()
    if value in {"y", "yes", "true", "1"}:
        return True
    if value in {"n", "no", "false", "0"}:
        return False
    return default


def _str2bool(value: str) -> bool:
    if isinstance(value, bool):
        return value
    val = value.strip().lower()
    if val in {"y", "yes", "true", "1"}:
        return True
    if val in {"n", "no", "false", "0"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="protenixscore",
        description="Score existing structures with Protenix confidence head.",
    )
    subparsers = parser.add_subparsers(dest="command")

    score = subparsers.add_parser("score", help="Score structures")
    score.add_argument("--input", required=True, help="Input PDB/CIF file or directory")
    score.add_argument("--output", required=True, help="Output directory")
    score.add_argument("--recursive", action="store_true", help="Recurse into subdirectories")
    score.add_argument("--glob", default=DEFAULT_GLOBS, help="Comma-separated glob patterns")

    score.add_argument("--score_only", action="store_true", default=True, help="Score-only mode (always on)")
    score.add_argument(
        "--use_msas",
        default="both",
        choices=["both", "target", "binder", "false"],
        help="Real-MSA usage mode by role (default: both).",
    )
    score.add_argument("--use_esm", action="store_true", default=False, help="Enable ESM embeddings")
    score.add_argument("--msa_map_csv", default=None, help="CSV map for provided MSAs.")
    score.add_argument("--target_msa_shared_dir", default=None, help="Shared MSA dir for all target chains.")
    score.add_argument("--binder_msa_shared_dir", default=None, help="Shared MSA dir for all binder chains.")
    score.add_argument(
        "--msa_provider",
        default="mmseqs2",
        choices=["mmseqs2", "none"],
        help="Provider for unresolved enabled-role MSAs (default: mmseqs2).",
    )
    score.add_argument(
        "--msa_host_url",
        default="https://api.colabfold.com",
        help="MMseqs2/ColabFold-compatible host URL.",
    )
    score.add_argument(
        "--msa_cache_mode",
        default="readwrite",
        choices=["readwrite", "read", "write", "none"],
        help="MSA cache read/write policy (default: readwrite).",
    )
    score.add_argument(
        "--msa_cache_dir",
        default=None,
        help="MSA cache dir (default: <output>/msa_cache when cache mode is not none).",
    )
    score.add_argument(
        "--msa_missing_policy",
        default="error",
        choices=["error", "single"],
        help="Behavior when enabled-role MSA is unresolved (default: error).",
    )
    score.add_argument(
        "--validate_msa_inputs",
        type=_str2bool,
        default=True,
        help="Run fail-fast MSA preflight validation (true/false). Default: true",
    )
    score.add_argument(
        "--chain_sequence",
        action="append",
        default=[],
        help="Override chain sequences (format: CHAIN=SEQUENCE). Repeatable.",
    )
    score.add_argument(
        "--target_chains",
        default=None,
        help="Comma-separated chain IDs treated as target (e.g. A,B).",
    )
    score.add_argument(
        "--target_chain_sequences",
        default=None,
        help="FASTA file with target sequences (match by sequence).",
    )
    score.add_argument(
        "--msa_use_env",
        type=_str2bool,
        default=True,
        help="Include environmental databases in ColabFold MSA (true/false).",
    )
    score.add_argument(
        "--msa_use_filter",
        type=_str2bool,
        default=True,
        help="Enable ColabFold filtering (true/false).",
    )
    score.add_argument(
        "--msa_cache_refresh",
        type=_str2bool,
        default=False,
        help="Force re-fetch MSAs even if cached.",
    )

    score.add_argument("--convert_pdb_to_cif", action="store_true", default=True, help="Convert PDB to CIF")
    score.add_argument("--keep_intermediate", action="store_true", default=False, help="Keep intermediate CIF/JSON")
    score.add_argument(
        "--intermediate_dir",
        default=None,
        help="Directory for intermediate files (default: <output>/intermediate)",
    )
    score.add_argument("--assembly_id", default=None, help="Assembly ID to expand (mmCIF) ")
    score.add_argument("--altloc", default="first", help="Altloc selection (first/A/B)")

    score.add_argument(
        "--checkpoint_dir",
        default=DEFAULT_CHECKPOINT_DIR,
        help="Protenix checkpoint directory (defaults to PROTENIX_CHECKPOINT_DIR)",
    )
    score.add_argument(
        "--model_name",
        default="protenix_base_default_v1.0.0",
        help="Protenix model name",
    )
    score.add_argument("--device", default="auto", help="cpu|cuda:N|auto")
    score.add_argument("--dtype", default="bf16", help="fp32|bf16|fp16")
    score.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")

    score.add_argument("--batch_size", type=int, default=1, help="Batch size (currently 1)")
    score.add_argument("--max_tokens", type=int, default=None, help="Optional max tokens")
    score.add_argument("--max_atoms", type=int, default=None, help="Optional max atoms")

    score.add_argument("--write_full_confidence", action="store_true", default=True)
    score.add_argument("--write_summary_confidence", action="store_true", default=True)
    score.add_argument(
        "--write_ipsae",
        type=_str2bool,
        default=True,
        help="Compute and write ipSAE metrics into summary outputs (true/false). Default: true",
    )
    score.add_argument(
        "--ipsae_pae_cutoff",
        type=float,
        default=10.0,
        help="PAE cutoff in Angstroms for ipSAE (default: 10.0).",
    )
    score.add_argument("--summary_format", default="json", choices=["json", "csv"])
    score.add_argument(
        "--aggregate_csv",
        default=None,
        help="Write global CSV summary (default: <output>/summary.csv)",
    )
    score.add_argument("--overwrite", action="store_true", default=False)
    score.add_argument(
        "--failed_log",
        default=None,
        help="Failed records log (default: <output>/failed_records.txt)",
    )
    score.add_argument(
        "--missing_atom_policy",
        default="reference",
        choices=["reference", "zero", "error"],
        help="How to fill missing atoms when mapping coordinates",
    )

    interactive = subparsers.add_parser("interactive", help="Guided scoring")

    return parser


def _interactive_args() -> argparse.Namespace:
    input_path = _prompt("Input PDB/CIF path")
    output_path = _prompt("Output directory", "./protenixscore_out")
    model_name = _prompt("Model name", "protenix_base_default_v1.0.0")
    checkpoint_dir = _prompt(
        "Checkpoint dir (optional)",
        DEFAULT_CHECKPOINT_DIR or "",
    )
    use_msas = _prompt("use_msas (both|target|binder|false)", "both")
    use_esm = _prompt_bool("Use ESM", False)
    dtype = _prompt("dtype (fp32|bf16|fp16)", "bf16")
    device = _prompt("device (cpu|cuda:N|auto)", "auto")
    msa_map_csv = _prompt("MSA map CSV (optional)", "")
    target_msa_shared_dir = _prompt("Shared target MSA dir (optional)", "")
    binder_msa_shared_dir = _prompt("Shared binder MSA dir (optional)", "")

    args = argparse.Namespace(
        command="score",
        input=input_path,
        output=output_path,
        recursive=False,
        glob=DEFAULT_GLOBS,
        score_only=True,
        use_msas=use_msas,
        use_esm=use_esm,
        msa_map_csv=msa_map_csv or None,
        target_msa_shared_dir=target_msa_shared_dir or None,
        binder_msa_shared_dir=binder_msa_shared_dir or None,
        msa_provider="mmseqs2",
        msa_host_url="https://api.colabfold.com",
        msa_cache_mode="readwrite",
        msa_cache_dir=None,
        msa_missing_policy="error",
        validate_msa_inputs=True,
        chain_sequence=[],
        target_chains=None,
        target_chain_sequences=None,
        msa_use_env=True,
        msa_use_filter=True,
        msa_cache_refresh=False,
        convert_pdb_to_cif=True,
        keep_intermediate=False,
        intermediate_dir=None,
        assembly_id=None,
        altloc="first",
        checkpoint_dir=checkpoint_dir or None,
        model_name=model_name,
        device=device,
        dtype=dtype,
        num_workers=4,
        batch_size=1,
        max_tokens=None,
        max_atoms=None,
        write_full_confidence=True,
        write_summary_confidence=True,
        write_ipsae=True,
        ipsae_pae_cutoff=10.0,
        summary_format="json",
        aggregate_csv=None,
        overwrite=False,
        failed_log=None,
        missing_atom_policy="reference",
    )
    return args


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "interactive":
        args = _interactive_args()
    elif args.command != "score":
        parser.print_help()
        return

    args.glob = _parse_globs(args.glob)
    if args.failed_log is None:
        args.failed_log = os.path.join(args.output, "failed_records.txt")
    if args.aggregate_csv is None:
        args.aggregate_csv = os.path.join(args.output, "summary.csv")

    run_score(args)


if __name__ == "__main__":
    main()
