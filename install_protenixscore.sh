#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ENV_NAME="protenixscore"
USE_EXISTING_ENV=0
NO_CONDA=0
FORK_URL="https://github.com/cytokineking/Protenix"
FORK_BRANCH="main"
PROTENIX_DIR="${SCRIPT_DIR}/Protenix_fork"
PROTENIXSCORE_DIR="${SCRIPT_DIR}"
CHECKPOINT_DIR=""
DATA_DIR="${SCRIPT_DIR}/protenix_data"
SKIP_WEIGHTS=0
SKIP_CCD=0
CPU_MODE=0
MODEL_NAME="protenix_base_default_v0.5.0"

print_usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Options:
  --env-name NAME             Conda env name (default: ${ENV_NAME})
  --use-existing-env          Use existing conda env (skip create)
  --no-conda                  Use current Python env (no conda)
  --fork URL                  Protenix fork URL (default: ${FORK_URL})
  --branch NAME               Protenix fork branch (default: ${FORK_BRANCH})
  --protenix-dir PATH         Clone/install Protenix here (default: ${PROTENIX_DIR})
  --protenixscore-dir PATH    ProtenixScore dir (default: ${PROTENIXSCORE_DIR})
  --checkpoint-dir PATH       Checkpoint dir (default: <protenix-dir>/release_data/checkpoint)
  --data-dir PATH             Data dir for CCD/cache (default: ${DATA_DIR})
  --skip-weights              Skip downloading model weights
  --skip-checkpoints          Alias for --skip-weights
  --skip-ccd                  Skip downloading CCD cache files
  --cpu                       Install Protenix in CPU-only mode
  --model-name NAME           Model name to download (default: ${MODEL_NAME})
  -h, --help                  Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-name)
      ENV_NAME="$2"; shift 2 ;;
    --use-existing-env)
      USE_EXISTING_ENV=1; shift ;;
    --no-conda)
      NO_CONDA=1; shift ;;
    --fork)
      FORK_URL="$2"; shift 2 ;;
    --branch)
      FORK_BRANCH="$2"; shift 2 ;;
    --protenix-dir)
      PROTENIX_DIR="$2"; shift 2 ;;
    --protenixscore-dir)
      PROTENIXSCORE_DIR="$2"; shift 2 ;;
    --checkpoint-dir)
      CHECKPOINT_DIR="$2"; shift 2 ;;
    --data-dir)
      DATA_DIR="$2"; shift 2 ;;
    --skip-weights|--skip-checkpoints)
      SKIP_WEIGHTS=1; shift ;;
    --skip-ccd)
      SKIP_CCD=1; shift ;;
    --cpu)
      CPU_MODE=1; shift ;;
    --model-name)
      MODEL_NAME="$2"; shift 2 ;;
    -h|--help)
      print_usage; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2
      print_usage; exit 1 ;;
  esac
 done

if [[ -z "${CHECKPOINT_DIR}" ]]; then
  CHECKPOINT_DIR="${PROTENIX_DIR}/release_data/checkpoint"
fi

if [[ ! -d "${PROTENIXSCORE_DIR}" ]]; then
  echo "ProtenixScore directory not found: ${PROTENIXSCORE_DIR}" >&2
  exit 1
fi

if [[ ${NO_CONDA} -eq 0 ]]; then
  if ! command -v conda >/dev/null 2>&1; then
    echo "conda not found. Install Miniforge/Conda or use --no-conda." >&2
    exit 1
  fi
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  if [[ ${USE_EXISTING_ENV} -eq 0 ]]; then
    if conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
      echo "Conda env ${ENV_NAME} already exists; using existing env."
      USE_EXISTING_ENV=1
    fi
  fi
  if [[ ${USE_EXISTING_ENV} -eq 0 ]]; then
    echo "Creating conda env ${ENV_NAME} (python=3.11)"
    conda create -y -n "${ENV_NAME}" python=3.11
  fi
  conda activate "${ENV_NAME}"
else
  echo "Using current Python environment (no conda)."
  python - <<'PY'
import sys
if sys.version_info < (3, 11):
    raise SystemExit("Python >=3.11 required by Protenix")
PY
fi

if [[ -d "${PROTENIX_DIR}" ]]; then
  if [[ -d "${PROTENIX_DIR}/.git" ]]; then
    echo "Updating existing Protenix repo at ${PROTENIX_DIR}"
    git -C "${PROTENIX_DIR}" remote set-url origin "${FORK_URL}"
    git -C "${PROTENIX_DIR}" fetch origin
    git -C "${PROTENIX_DIR}" checkout "${FORK_BRANCH}"
    git -C "${PROTENIX_DIR}" pull --ff-only origin "${FORK_BRANCH}"
  else
    echo "${PROTENIX_DIR} exists but is not a git repo. Remove it or choose --protenix-dir." >&2
    exit 1
  fi
else
  echo "Cloning Protenix fork: ${FORK_URL} (branch ${FORK_BRANCH})"
  git clone --branch "${FORK_BRANCH}" "${FORK_URL}" "${PROTENIX_DIR}"
fi

should_install_protenix=1
if python - <<PY >/dev/null 2>&1; then
import importlib.util
spec = importlib.util.find_spec("protenix")
if spec and spec.origin:
    print(spec.origin)
PY
  installed_path=$(python - <<PY
import importlib.util
spec = importlib.util.find_spec("protenix")
print(spec.origin if spec and spec.origin else "")
PY
)
  if [[ -n "${installed_path}" && "${installed_path}" == "${PROTENIX_DIR}"* ]]; then
    should_install_protenix=0
  fi
fi

if [[ ${should_install_protenix} -eq 1 ]]; then
  echo "Installing Protenix (this will enforce pinned torch==2.7.1 per requirements.txt)"
  cd "${PROTENIX_DIR}"
  if [[ ${CPU_MODE} -eq 1 ]]; then
    python setup.py develop --cpu
  else
    python -m pip install -e .
  fi
else
  echo "Protenix already installed from ${PROTENIX_DIR}; skipping reinstall."
fi

# Ensure protenixscore is importable
python - <<PY
import site
from pathlib import Path
root = Path('${SCRIPT_DIR}').resolve()
site_pkgs = site.getsitepackages()[0]
Path(site_pkgs, 'protenixscore.pth').write_text(str(root.parent))
print(f"Wrote protenixscore.pth to {site_pkgs}")
PY

# Download weights + CCD from official Protenix URLs (unless skipped)
get_url() {
  local key="$1"
  python - <<PY
import importlib.util
from pathlib import Path
spec = importlib.util.spec_from_file_location('dep', '${PROTENIX_DIR}/protenix/web_service/dependency_url.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
print(mod.URL['${key}'])
PY
}

fetch_file() {
  local url="$1"
  local dest="$2"
  if [[ -f "${dest}" ]]; then
    echo "Exists: ${dest}"
    return
  fi
  mkdir -p "$(dirname "${dest}")"
  echo "Downloading ${url} -> ${dest}"
  if command -v curl >/dev/null 2>&1; then
    curl -L --fail -o "${dest}" "${url}"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "${dest}" "${url}"
  else
    echo "Need curl or wget to download ${url}" >&2
    exit 1
  fi
}

if [[ ${SKIP_WEIGHTS} -eq 0 ]]; then
  MODEL_PATH="${CHECKPOINT_DIR}/${MODEL_NAME}.pt"
  if [[ -f "${MODEL_PATH}" ]]; then
    echo "Weights already present: ${MODEL_PATH}"
  else
    MODEL_URL=$(get_url "${MODEL_NAME}")
    fetch_file "${MODEL_URL}" "${MODEL_PATH}"
  fi
else
  echo "Skipping weights download"
fi

if [[ ${SKIP_CCD} -eq 0 ]]; then
  CCD_PATH="${DATA_DIR}/components.v20240608.cif"
  CCD_RD_PATH="${DATA_DIR}/components.v20240608.cif.rdkit_mol.pkl"
  CLUSTER_PATH="${DATA_DIR}/clusters-by-entity-40.txt"
  if [[ -f "${CCD_PATH}" && -f "${CCD_RD_PATH}" && -f "${CLUSTER_PATH}" ]]; then
    echo "CCD/data already present in ${DATA_DIR}; skipping download"
  else
    CCD_URL=$(get_url "ccd_components_file")
    CCD_RD_URL=$(get_url "ccd_components_rdkit_mol_file")
    CLUSTER_URL=$(get_url "pdb_cluster_file")
    fetch_file "${CCD_URL}" "${CCD_PATH}"
    fetch_file "${CCD_RD_URL}" "${CCD_RD_PATH}"
    fetch_file "${CLUSTER_URL}" "${CLUSTER_PATH}"
  fi
else
  echo "Skipping CCD/data download"
fi

# Write conda activation env vars if using conda
if [[ ${NO_CONDA} -eq 0 ]]; then
  ACTIVATE_DIR="${CONDA_PREFIX}/etc/conda/activate.d"
  mkdir -p "${ACTIVATE_DIR}"
  cat > "${ACTIVATE_DIR}/protenixscore.sh" <<EOF
export PROTENIX_CHECKPOINT_DIR="${CHECKPOINT_DIR}"
export PROTENIX_DATA_ROOT_DIR="${DATA_DIR}"
export LAYERNORM_TYPE="torch"
EOF
  echo "Wrote env activation script to ${ACTIVATE_DIR}/protenixscore.sh"
else
  echo "Set these env vars in your shell:"
  echo "  export PROTENIX_CHECKPOINT_DIR=${CHECKPOINT_DIR}"
  echo "  export PROTENIX_DATA_ROOT_DIR=${DATA_DIR}"
  echo "  export LAYERNORM_TYPE=torch"
fi

echo "Install complete."
