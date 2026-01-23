import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import List

from protenixscore.cli import _str2bool
from protenixscore.score import (
    _configure_device,
    _load_runner,
    _parse_chain_sequence_overrides,
    _prepare_intermediate_dirs,
    _sanitize_name,
    collect_input_files,
    _score_single,
)


def _parse_globs(value: str) -> List[str]:
    patterns = []
    for item in value.split(","):
        item = item.strip()
        if item:
            patterns.append(item)
    return patterns or ["*.pdb", "*.cif"]


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2))


def _parse_forward_times(log_path: Path) -> List[float]:
    times: List[float] = []
    pattern = re.compile(r"Model forward time: ([0-9.]+)s")
    if not log_path.exists():
        return times
    for line in log_path.read_text().splitlines():
        match = pattern.search(line)
        if match:
            try:
                times.append(float(match.group(1)))
            except ValueError:
                pass
    return times


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="protenixscore-benchmark",
        description="Benchmark protenixscore vs full Protenix inference.",
    )
    parser.add_argument("--input", required=True, help="Input PDB/CIF file or directory")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--recursive", action="store_true", help="Recurse into subdirectories")
    parser.add_argument("--glob", default="*.pdb,*.cif", help="Comma-separated glob patterns")

    parser.add_argument("--use_msa", type=_str2bool, default=True)
    parser.add_argument("--use_esm", type=_str2bool, default=False)
    parser.add_argument("--msa_path", default=None)
    parser.add_argument("--chain_sequence", action="append", default=[])
    parser.add_argument("--target_chains", default=None)
    parser.add_argument("--target_chain_sequences", default=None)
    parser.add_argument("--target_msa_path", default=None)
    parser.add_argument("--binder_msa_mode", default="single", choices=["single", "none"])
    parser.add_argument("--msa_cache_dir", default=None)
    parser.add_argument("--msa_source", default="none", choices=["none", "colabfold"])
    parser.add_argument("--msa_host", default="https://api.colabfold.com")
    parser.add_argument("--msa_use_env", type=_str2bool, default=True)
    parser.add_argument("--msa_use_filter", type=_str2bool, default=True)
    parser.add_argument("--msa_cache_refresh", type=_str2bool, default=False)

    parser.add_argument("--checkpoint_dir", default=os.environ.get("PROTENIX_CHECKPOINT_DIR"))
    parser.add_argument("--model_name", default="protenix_base_default_v0.5.0")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--inference_repo", default=None, help="Path to Protenix repo")
    parser.add_argument("--inference_seed", type=int, default=101)
    parser.add_argument("--inference_n_cycle", type=int, default=10)
    parser.add_argument("--inference_n_step", type=int, default=200)
    parser.add_argument("--inference_n_sample", type=int, default=1)
    parser.add_argument("--triangle_attention", default="torch")
    parser.add_argument("--triangle_multiplicative", default="torch")
    parser.add_argument("--skip_inference", action="store_true", default=False)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.glob = _parse_globs(args.glob)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    score_dir = output_dir / "score_outputs"
    infer_dir = output_dir / "infer_outputs"
    inter_dir = output_dir / "intermediate"
    for path in (score_dir, infer_dir, inter_dir):
        path.mkdir(parents=True, exist_ok=True)

    input_files = collect_input_files(args.input, args.recursive, args.glob)
    if not input_files:
        raise FileNotFoundError("No input structures found")

    args.score_only = True
    args.keep_intermediate = True
    args.intermediate_dir = str(inter_dir)
    args.output = str(score_dir)
    args.overwrite = True
    args.convert_pdb_to_cif = True
    args.assembly_id = None
    args.altloc = "first"
    args.batch_size = 1
    args.max_tokens = None
    args.max_atoms = None
    args.write_full_confidence = True
    args.write_summary_confidence = True
    args.summary_format = "json"
    args.aggregate_csv = str(score_dir / "summary.csv")
    args.failed_log = str(score_dir / "failed_records.txt")
    args.missing_atom_policy = "reference"

    _configure_device(args.device)
    runner = _load_runner(args)

    chain_sequence_overrides = _parse_chain_sequence_overrides(args.chain_sequence)
    score_rows = []
    json_paths = []

    for file_path in input_files:
        result = _score_single(
            file_path=file_path,
            runner=runner,
            args=args,
            output_dir=score_dir,
            inter_dir=inter_dir,
            chain_sequence_overrides=chain_sequence_overrides,
        )
        score_rows.append(
            {
                "sample": result.sample_name,
                "prep_seconds": result.prep_seconds,
                "model_seconds": result.model_seconds,
                "total_seconds": result.total_seconds,
                "plddt": float(result.summary.get("plddt", 0.0)),
                "ptm": float(result.summary.get("ptm", 0.0)),
                "iptm": float(result.summary.get("iptm", 0.0)),
                "ranking_score": float(result.summary.get("ranking_score", 0.0)),
            }
        )
        json_paths.append(inter_dir / f"{_sanitize_name(file_path.stem)}.json")

    timing_csv = output_dir / "score_timing.csv"
    if score_rows:
        with open(timing_csv, "w", newline="") as f:
            header = list(score_rows[0].keys())
            f.write(",".join(header) + "\n")
            for row in score_rows:
                f.write(",".join(str(row.get(col, "")) for col in header) + "\n")

    score_total = sum(r.get("total_seconds") or 0.0 for r in score_rows)
    score_model_total = sum(r.get("model_seconds") or 0.0 for r in score_rows)

    summary = {
        "num_samples": len(score_rows),
        "score_total_seconds": score_total,
        "score_avg_seconds": (score_total / len(score_rows)) if score_rows else 0.0,
        "score_model_avg_seconds": (score_model_total / len(score_rows)) if score_rows else 0.0,
        "score_timing_csv": str(timing_csv),
    }

    if not args.skip_inference and score_rows:
        combined_json = output_dir / "infer_inputs.json"
        combined_payload = []
        for path in json_paths:
            if not path.exists():
                continue
            payload = json.loads(path.read_text())
            if isinstance(payload, list):
                combined_payload.extend(payload)
            else:
                combined_payload.append(payload)
        combined_json.write_text(json.dumps(combined_payload, indent=2))

        infer_repo = Path(args.inference_repo) if args.inference_repo else Path(__file__).resolve().parents[1] / "Protenix"
        infer_script = infer_repo / "runner" / "inference.py"
        if not infer_script.exists():
            raise FileNotFoundError(f"inference.py not found at {infer_script}")

        infer_log = output_dir / "infer.log"
        cmd = [
            "python",
            str(infer_script),
            "--model_name",
            args.model_name,
            "--seeds",
            str(args.inference_seed),
            "--dump_dir",
            str(infer_dir),
            "--input_json_path",
            str(combined_json),
            "--model.N_cycle",
            str(args.inference_n_cycle),
            "--sample_diffusion.N_sample",
            str(args.inference_n_sample),
            "--sample_diffusion.N_step",
            str(args.inference_n_step),
            "--triangle_attention",
            args.triangle_attention,
            "--triangle_multiplicative",
            args.triangle_multiplicative,
        ]

        env = os.environ.copy()
        if "PROTENIX_DATA_ROOT_DIR" in os.environ:
            env["PROTENIX_DATA_ROOT_DIR"] = os.environ["PROTENIX_DATA_ROOT_DIR"]

        start = time.perf_counter()
        with open(infer_log, "w") as log_handle:
            subprocess.run(cmd, check=True, stdout=log_handle, stderr=log_handle, env=env)
        end = time.perf_counter()

        forward_times = _parse_forward_times(infer_log)
        summary.update(
            {
                "inference_total_seconds": end - start,
                "inference_avg_seconds": (end - start) / len(score_rows),
                "inference_model_avg_seconds": (sum(forward_times) / len(forward_times))
                if forward_times
                else None,
                "inference_log": str(infer_log),
            }
        )
        if score_total > 0:
            summary["speedup_total"] = (end - start) / score_total
        if summary.get("inference_model_avg_seconds") and summary["score_model_avg_seconds"]:
            summary["speedup_model"] = summary["inference_model_avg_seconds"] / summary["score_model_avg_seconds"]

    _write_json(output_dir / "benchmark_summary.json", summary)


if __name__ == "__main__":
    main()
