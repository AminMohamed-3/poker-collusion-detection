"""Use the existing Python 3.12 package set without changing historic environments."""
import os
import site
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for name in ("POLARS_MAX_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(name, "2")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
sys.dont_write_bytecode = True
site.addsitedir(str(ROOT / ".venv/Lib/site-packages"))
sys.path.insert(0, str(ROOT / "src"))

