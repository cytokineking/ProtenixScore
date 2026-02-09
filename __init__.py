"""ProtenixScore: score-only confidence metrics for existing structures."""

from pathlib import Path
import sys

__all__ = ["__version__"]

__version__ = "0.1.0"

_ROOT = Path(__file__).resolve().parents[1]
for _name in ("Protenix_fork", "Protenix"):
    _protenix_dir = _ROOT / _name
    if _protenix_dir.exists() and str(_protenix_dir) not in sys.path:
        sys.path.insert(0, str(_protenix_dir))
        break
