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
        "--use_msa",
        type=_str2bool,
        default=True,
        help="Enable MSA features (true/false). Default: true",
    )
    score.add_argument("--use_esm", action="store_true", default=False, help="Enable ESM embeddings")
    score.add_argument(
        "--msa_path",
        default=None,
        help="Path to precomputed MSA dirs (per-sample or shared).",
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
        "--target_msa_path",
        default=None,
        help="Path to target MSA dirs (entity_1, entity_2, ...).",
    )
    score.add_argument(
        "--binder_msa_mode",
        default="single",
        choices=["single", "none"],
        help="MSA mode for binder chains (default: single).",
    )
    score.add_argument(
        "--msa_cache_dir",
        default=None,
        help="Cache dir for target MSAs (hashed by sequence).",
    )
    score.add_argument(
        "--msa_source",
        default="none",
        choices=["none", "colabfold"],
        help="How to generate target MSAs when not cached.",
    )
    score.add_argument(
        "--msa_host",
        default="https://api.colabfold.com",
        help="ColabFold server host URL.",
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
        default="protenix_base_default_v0.5.0",
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
    model_name = _prompt("Model name", "protenix_base_default_v0.5.0")
    checkpoint_dir = _prompt(
        "Checkpoint dir (optional)",
        DEFAULT_CHECKPOINT_DIR or "",
    )
    use_msa = _prompt_bool("Use MSA", True)
    use_esm = _prompt_bool("Use ESM", False)
    dtype = _prompt("dtype (fp32|bf16|fp16)", "bf16")
    device = _prompt("device (cpu|cuda:N|auto)", "auto")
    msa_path = _prompt("MSA path (optional)", "")

    args = argparse.Namespace(
        command="score",
        input=input_path,
        output=output_path,
        recursive=False,
        glob=DEFAULT_GLOBS,
        score_only=True,
        use_msa=use_msa,
        use_esm=use_esm,
        msa_path=msa_path or None,
        chain_sequence=[],
        target_chains=None,
        target_chain_sequences=None,
        target_msa_path=None,
        binder_msa_mode="single",
        msa_cache_dir=None,
        msa_source="none",
        msa_host="https://api.colabfold.com",
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
