import os
import sys
from pathlib import Path

# Make `src/` importable without requiring an installed package.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

# Avoid accidental CUDA use in CI-like environments.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
