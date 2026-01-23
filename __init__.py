"""ProtenixScore: score-only confidence metrics for existing structures."""

from pathlib import Path
import sys

__all__ = ["__version__"]

__version__ = "0.1.0"

_ROOT = Path(__file__).resolve().parents[1]
_PROTENIX_DIR = _ROOT / "Protenix"
if _PROTENIX_DIR.exists() and str(_PROTENIX_DIR) not in sys.path:
    sys.path.insert(0, str(_PROTENIX_DIR))
