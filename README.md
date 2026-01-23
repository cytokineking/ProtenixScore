# ProtenixScore

This repo was inspired by AF3Score (https://github.com/Mingchenchen/AF3Score).

Score existing protein structures (PDB or CIF) with the Protenix confidence
head, without running diffusion. ProtenixScore is designed for fast, reproducible
"score-only" evaluation of fixed coordinates and is suitable for batch pipelines.
In practice it is typically ~2.5-3x faster than running the full Protenix
inference pipeline when you only need confidence scoring. 

Key features
- Score-only mode: uses provided coordinates, no diffusion sampling.
- PDB or CIF input, with automatic PDB -> CIF conversion.
- Per-structure outputs plus an aggregate CSV summary.
- MSA features enabled by default (single-sequence MSA is injected if not provided).

## Requirements

- This repo checked out locally.
- Pinned Protenix fork installed (use `install_protenixscore.sh`).
- Python 3.11 environment with Protenix dependencies installed (conda or existing).
- Protenix checkpoint + CCD/data cache (downloaded by the install script unless skipped).

## Install (recommended)

Clone this repo, then run the install script from the repo root:

```bash
git clone https://github.com/cytokineking/ProtenixScore
cd protenixscore
./install_protenixscore.sh
```

This clones the pinned Protenix fork (modified to support score-only mode), installs dependencies, and downloads
weights/CCD data unless skipped. It also wires up `PROTENIX_CHECKPOINT_DIR` and
`PROTENIX_DATA_ROOT_DIR` (conda activation or printed for manual export).
See `./install_protenixscore.sh --help` for options.

## Quickstart

After installing (and activating the environment if you used conda), validate
the installation using the included test PDBs (single file):

```bash
python -m protenixscore score \
  --input ./test_pdbs/1_PDL1-freebindcraft-2_l141_s788784_mpnn6_model1.pdb \
  --output ./score_out
```

Validate the installation using the included test PDBs (entire folder):

```bash
python -m protenixscore score \
  --input ./test_pdbs \
  --output ./score_out \
  --recursive
```

Interactive guided mode:

```bash
python -m protenixscore interactive
```

## Outputs

For each input structure `sample`, outputs are written to:

```
<output>/
  summary.csv
  failed_records.txt
  <sample>/
    summary_confidence.json
    full_confidence.json
    chain_id_map.json
    missing_atoms.json   (only if missing atoms were detected)
```

Notes:
- `summary.csv` is written when at least one structure is successfully scored.
- `failed_records.txt` is written only if one or more inputs fail.
- `chain_id_map.json` records the mapping between Protenix internal chain IDs
  and source chain IDs.
- `missing_atoms.json` is written when coordinates are missing and a fallback
  policy is used.

## Common options

- `--model_name` (default: `protenix_base_default_v0.5.0`)
- `--checkpoint_dir` (optional, overrides default checkpoint location)
- `--device` (`cpu|cuda:N|auto`, default: `auto`)
- `--dtype` (`fp32|bf16|fp16`, default: `bf16`)
- `--use_msa` (default: true; injects single-sequence MSA if none provided)
- `--msa_path` (optional; use precomputed MSA instead of dummy)
- `--chain_sequence` (optional; override chain sequences, format `A=SEQUENCE`, repeatable)
- `--target_chains` (optional; comma-separated chain IDs to treat as target)
- `--target_chain_sequences` (optional; FASTA of target sequences to match by sequence)
- `--target_msa_path` (optional; precomputed target MSAs in entity_1/, entity_2/, ...)
- `--binder_msa_mode` (default: single; single or none)
- `--msa_cache_dir` (optional; cache target MSAs by sequence hash)
- `--msa_source` (none|colabfold; how to generate target MSAs when cache miss)
- `--msa_host` (ColabFold server URL)
- `--msa_use_env` / `--msa_use_filter` (ColabFold controls, default true)
- `--msa_cache_refresh` (force re-fetch MSAs even if cached)
- `--use_esm` (optional)
- `--convert_pdb_to_cif` (always on for PDB input)
- `--missing_atom_policy` (`reference|zero|error`, default: `reference`)
- `--max_tokens` / `--max_atoms` (optional safety caps)

## How it works (high level)

1. Parse PDB/CIF (PDB is always converted to CIF).
2. Extract per-chain sequences from coordinates (or override via `--chain_sequence`).
3. Build Protenix features from CIF.
4. Map source atom coordinates to Protenix atom ordering.
5. Run Protenix confidence head with the provided coordinates.

## Target/binder batch workflow (recommended for many binders vs one target)

If you are scoring many binders against a fixed target, reuse a single target MSA
and use single-sequence MSAs for binders:

```bash
python -m protenixscore score \
  --input ./ranked \
  --output ./scores \
  --recursive \
  --target_chains A \
  --binder_msa_mode single \
  --msa_cache_dir ./msa_cache \
  --msa_source colabfold
```

Notes:
- Target MSAs are cached by sequence hash under `--msa_cache_dir`.
- Binder chains are kept in single-sequence mode for speed.

MSA notes:
- Protenix confidence scoring relies on MSAs; use precomputed MSAs or enable fetching
  so the target chain has a real MSA.
- Use `--msa_path` to point to precomputed MSAs laid out as `entity_1/`, `entity_2/`, etc.
  For batch runs you can also provide per-sample subfolders under `/path/to/msa/<sample_name>/entity_*`.
- When you set `--target_chains` (or `--target_chain_sequences`) and do not provide
  `--target_msa_path` / `--msa_cache_dir`, ProtenixScore defaults to fetching target
  MSAs from the ColabFold server and caches them under `<output>/msa_cache`.

## Troubleshooting

- If you see missing atom warnings, consider switching
  `--missing_atom_policy` to `error` to fail fast.
- If you hit device issues, try `--device cpu` to verify your setup.
- If checkpoints are not found, specify `--checkpoint_dir`.
