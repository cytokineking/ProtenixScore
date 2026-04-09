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
- Deterministic MSA resolution with map/shared/cache/fetch fallback.
- ipSAE metrics (AF3Score-style) computed from Protenix token-pair PAE.

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

This clones the pinned Protenix fork (with the Protenix v2 merge and score-only support), installs dependencies, and downloads
weights/CCD data unless skipped. By default it installs the `protenix-v2` checkpoint. It also wires up `PROTENIX_CHECKPOINT_DIR` and
`PROTENIX_DATA_ROOT_DIR` (conda activation or printed for manual export).
See `./install_protenixscore.sh --help` for options.

By default, `install_protenixscore.sh` pins the Protenix fork to a specific git commit for reproducibility.
Override with `--commit <sha>` (or pass an empty commit string to follow `--branch`).
The installer downloads the `protenix-v2` checkpoint by default, and the CLI also
defaults to `--model_name protenix-v2`. If you want to score with the older v1 base
model, install that checkpoint with
`--model-name protenix_base_default_v1.0.0` and then pass
`--model_name protenix_base_default_v1.0.0` when scoring. The installer downloads
only the selected checkpoint, so if you later want to score with another model you
will need to rerun the installer with `--model-name` or place that model's `.pt`
file into `PROTENIX_CHECKPOINT_DIR`.

If you keep multiple Protenix checkouts around, set `PROTENIX_REPO_DIR=/path/to/Protenix_fork`
before running `python -m protenixscore ...` or `python benchmark.py ...` to force
discovery to use that exact checkout.

Protenix original repository:
https://github.com/bytedance/Protenix

Pinned fork used by the install script:
https://github.com/cytokineking/Protenix

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
  msa_resolution_summary.json
  <sample>/
    summary_confidence.json
    full_confidence.json
    chain_id_map.json
    msa_resolution.json
    missing_atoms.json   (only if missing atoms were detected)
```

Notes:
- `summary.csv` is written when at least one structure is successfully scored.
- `failed_records.txt` is written only if one or more inputs fail.
- `chain_id_map.json` records the mapping between Protenix internal chain IDs
  and source chain IDs.
- `msa_resolution.json` records where each chain's MSA came from (`single|map|shared|cache|fetched`).
- `msa_resolution_summary.json` aggregates run-level MSA source counts and fetch/cache stats.
- `missing_atoms.json` is written when coordinates are missing and a fallback
  policy is used.

### ipSAE (Interface Predicted Structural Alignment Error)

ProtenixScore computes ipSAE using the same definition as AF3Score's `calculate_ipsae`
(inspired by the `IPSAE` script family), but using Protenix's `token_pair_pae`
from `full_confidence.json` instead of AlphaFold JSON outputs.

Definition (directional, chain1 -> chain2):
- Let `PAE(i,j)` be the token-pair PAE from chain1 token `i` to chain2 token `j`.
- Keep only "valid" interface pairs where `PAE(i,j) < pae_cutoff` (default `10.0` Angstrom).
- For each chain1 token `i`, compute `n0res(i) = count_j valid(i,j)`.
- Compute a TM-score-like normalization per token:
  `d0(i) = max(1.0, 1.24 * cbrt(max(27, n0res(i)) - 15) - 1.8)`.
- Convert PAE to a PTM-like score:
  `ptm(i,j) = 1 / (1 + (PAE(i,j) / d0(i))^2)`.
- Per-token ipSAE is the mean `ptm(i,j)` over valid `j` (0 if no valid pairs).
- Final ipSAE for the directional chain pair is `max_i per_token_ipSAE(i)`.

Outputs:
- `summary_confidence.json` includes:
  - `ipsae_by_chain_pair`: map of directional chain-pair scores, keyed by source chain IDs (e.g. `A_B`, `B_A`).
  - `ipsae_target_to_binder`, `ipsae_binder_to_target`, `ipsae_interface_max` when `--target_chains` is provided.
- `summary.csv` includes `ipsae_interface_max`, `ipsae_target_to_binder`, `ipsae_binder_to_target`.

Which ipSAE metric should you use?
- For the common "many binders vs one target" setup (you pass `--target_chains A`),
  the binder-focused score is `ipsae_binder_to_target` (direction: binder -> target).

## Common options

- `--model_name` (default: `protenix-v2`; pass `protenix_base_default_v1.0.0` to opt into v1)
- `--checkpoint_dir` (optional, overrides default checkpoint location)
- `--device` (`cpu|cuda:N|auto`, default: `auto`)
- `--dtype` (`fp32|bf16|fp16`, default: `bf16`)
- `--use_msas` (`both|target|binder|false`, default: `both`)
- `--msa_map_csv` (optional; CSV map for chain/sequence-provided MSAs)
- `--target_msa_shared_dir` / `--binder_msa_shared_dir` (optional shared MSA dirs by role)
- `--msa_provider` (`mmseqs2|none`, default: `mmseqs2`)
- `--msa_host_url` (default: `https://api.colabfold.com`)
- `--msa_cache_mode` (`readwrite|read|write|none`, default: `readwrite`)
- `--msa_cache_dir` (optional; defaults to `<output>/msa_cache` when cache mode is not `none`)
- `--msa_missing_policy` (`error|single`, default: `error`)
- `--validate_msa_inputs` (`true|false`, default: `true`)
- `--chain_sequence` (optional; override chain sequences, format `A=SEQUENCE`, repeatable)
- `--target_chains` (optional; comma-separated chain IDs to treat as target)
- `--target_chain_sequences` (optional; FASTA of target sequences to match by sequence)
- `--msa_use_env` / `--msa_use_filter` (MMseqs2/ColabFold controls, default true)
- `--msa_cache_refresh` (force re-fetch when fetching into cache/write paths)
- `--use_esm` (optional)
- `--convert_pdb_to_cif` (always on for PDB input)
- `--missing_atom_policy` (`reference|zero|error`, default: `reference`)
- `--max_tokens` / `--max_atoms` (optional safety caps)
- `--write_ipsae` (true/false, default: true)
- `--ipsae_pae_cutoff` (default: 10.0 Angstrom)

## How it works (high level)

1. Parse PDB/CIF (PDB is always converted to CIF).
2. Extract per-chain sequences from coordinates (or override via `--chain_sequence`).
3. Build Protenix features from CIF.
4. Map source atom coordinates to Protenix atom ordering.
5. Run Protenix confidence head with the provided coordinates.

## MSA handling (important)

Use `--use_msas` to control which roles need real MSAs:

- `both`: target and binder chains use real MSAs.
- `target`: only target chains use real MSAs, binders are single-sequence.
- `binder`: only binder chains use real MSAs, targets are single-sequence.
- `false`: all chains use single-sequence MSAs.

Resolution order for enabled roles is deterministic:

1. `--msa_map_csv` exact `sample_id + chain_id`.
2. `--msa_map_csv` `role + sequence`.
3. `--msa_map_csv` `sequence` (must be unique).
4. shared role directory (`--target_msa_shared_dir` / `--binder_msa_shared_dir`).
5. cache (`--msa_cache_dir`) according to `--msa_cache_mode`.
6. fetch from provider (`--msa_provider mmseqs2`).
7. unresolved -> `--msa_missing_policy` (`error` by default).

If `--msa_provider none` is set, unresolved enabled-role chains still obey
`--msa_missing_policy` (`error` fails fast, `single` falls back to single-sequence).
`--msa_provider none` still reads existing mmseqs2 cache entries in read/readwrite modes.

Ambiguous map lookups are hard errors.

## MSA map CSV

Supported columns:

- match selectors: `sample_id` (or `sample`), `chain_id`, `sequence`, `role`
- location: `msa_dir` OR (`pairing_path` + `non_pairing_path`)

Rules:

- At least one selector strategy is required: `sample_id+chain_id` and/or `sequence`.
- Duplicate keys are hard errors (`sample_id+chain_id`, `role+sequence`, sequence-only).
- `sample_id` is normalized using the same sample sanitizer as scoring.
- `sequence` matching normalizes by uppercasing and removing spaces/gaps.

Template:

```csv
sample_id,chain_id,role,msa_dir
complex_0001,A,target,/data/msa/complex_0001_A
complex_0001,H,binder,/data/msa/complex_0001_H
```

Sequence-level reuse:

```csv
role,sequence,msa_dir
target,EVQLVESGGGLVQPGGSLRLS...,/data/msa/target_shared_1
binder,QVQLQQSGAELVKPGASVK...,/data/msa/binder_shared_heavy_chain
```

## Target/binder batch workflow (recommended for many binders vs one target)

If you are scoring many binders against a fixed target:

```bash
python -m protenixscore score \
  --input ./ranked \
  --output ./scores \
  --recursive \
  --use_msas both \
  --target_chains A \
  --target_msa_shared_dir ./msas/target_shared \
  --msa_cache_dir ./msa_cache \
  --msa_provider mmseqs2
```

Notes:
- Target chains use the shared target MSA.
- Binder chains resolve via map/cache/fetch based on your flags.

If you have explicit per-chain mappings, use `--msa_map_csv`:

```bash
python -m protenixscore score \
  --input ./ranked \
  --output ./scores \
  --recursive \
  --use_msas both \
  --target_chains A \
  --msa_map_csv ./examples/msa_map_template.csv
```

## Troubleshooting

- If you see missing atom warnings, consider switching
  `--missing_atom_policy` to `error` to fail fast.
- If you hit device issues, try `--device cpu` to verify your setup.
- If checkpoints are not found, specify `--checkpoint_dir`.
