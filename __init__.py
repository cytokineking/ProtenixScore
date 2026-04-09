"""ProtenixScore: score-only confidence metrics for existing structures."""

import os
from pathlib import Path
import sys

__all__ = ["__version__", "discover_protenix_dir", "get_loaded_protenix_dir"]

__version__ = "0.2.0"

_REPO_ROOT = Path(__file__).resolve().parent
_PROTENIX_REPO_ENV_VAR = "PROTENIX_REPO_DIR"


def _is_valid_protenix_dir(candidate: Path) -> bool:
    return (
        candidate.is_dir()
        and (candidate / "protenix").is_dir()
        and (candidate / "runner" / "inference.py").is_file()
    )


def _candidate_protenix_dirs():
    seen = set()
    env_dir_raw = os.environ.get(_PROTENIX_REPO_ENV_VAR)
    if env_dir_raw:
        env_candidate = Path(env_dir_raw).expanduser().resolve()
        if not _is_valid_protenix_dir(env_candidate):
            raise FileNotFoundError(
                f"{_PROTENIX_REPO_ENV_VAR} does not point to a valid Protenix checkout: {env_candidate}"
            )
        seen.add(str(env_candidate))
        yield env_candidate

    for root in (_REPO_ROOT, _REPO_ROOT.parent):
        for name in ("Protenix_fork", "Protenix"):
            candidate = (root / name).resolve()
            key = str(candidate)
            if key in seen:
                continue
            seen.add(key)
            yield candidate


def discover_protenix_dir():
    for candidate in _candidate_protenix_dirs():
        if _is_valid_protenix_dir(candidate):
            return candidate
    return None


def get_loaded_protenix_dir():
    module = sys.modules.get("protenix")
    module_file = getattr(module, "__file__", None)
    if not module_file:
        return None

    candidate = Path(module_file).resolve().parent.parent
    if _is_valid_protenix_dir(candidate):
        return candidate
    return None


_PROTENIX_DIR = discover_protenix_dir()
if _PROTENIX_DIR is not None and str(_PROTENIX_DIR) not in sys.path:
    sys.path.insert(0, str(_PROTENIX_DIR))
