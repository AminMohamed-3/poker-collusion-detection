"""Isolated TabPFN9 source overlay on the existing compatible torch environment."""
import os
import site
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
site.addsitedir(str(HERE.parents[1] / ".venv/Lib/site-packages"))
for name in ("POLARS_MAX_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "2"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["DO_NOT_TRACK"] = "1"
os.environ["TABPFN_NO_BROWSER"] = "1"
os.environ["SKB_DATA_DIRECTORY"] = str(HERE / "cache/skrub")
os.environ["MPLCONFIGDIR"] = str(HERE / "cache/matplotlib")
os.environ["TABPFN_MODEL_CACHE_DIR"] = str(HERE)
sys.path[:0] = [str(HERE / "vendor_deps"), str(HERE / "vendor")]
sys.dont_write_bytecode = True

if __name__ == "__main__":
    import tabpfn
    import torch
    import skrub
    from tabpfn.constants import ModelVersion
    print({"source": tabpfn.__file__, "torch": torch.__version__, "skrub": skrub.__version__, "models": [x.value for x in ModelVersion]})
