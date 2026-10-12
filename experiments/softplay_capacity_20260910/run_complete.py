"""Finish the frozen analysis and write its result, without adaptive work."""
from pathlib import Path
import subprocess,sys
H=Path(__file__).resolve().parent
subprocess.run([sys.executable,str(H/'run_analysis.py')],check=True)
subprocess.run([sys.executable,str(H/'compare.py')],check=True)
