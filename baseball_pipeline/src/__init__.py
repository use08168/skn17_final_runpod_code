# src/__init__.py
from pathlib import Path

# /workspace/baseball_pipeline
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
FAISS_DIR = PROJECT_ROOT / "faiss_index"
FISH_ROOT = PROJECT_ROOT / "fish-speech"

__all__ = [
    "PROJECT_ROOT",
    "DATA_DIR",
    "FAISS_DIR",
    "FISH_ROOT",
]
