# src/__init__.py
from pathlib import Path
import os

# 현재 파일의 위치에서 프로젝트 루트 찾기
# /workspace/skn17_final_runpod_code/baseball_pipeline/src/__init__.py
# -> /workspace/skn17_final_runpod_code/baseball_pipeline
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# RunPod 환경에서는 /workspace/skn17_final_runpod_code/baseball_pipeline
# 로컬에서는 사용자의 경로
DATA_DIR = PROJECT_ROOT / "data"
FAISS_DIR = PROJECT_ROOT / "faiss_index"
FISH_ROOT = PROJECT_ROOT / "fish-speech"

# 환경 확인용 (디버깅)
if os.environ.get("DEBUG_PATHS"):
    print(f"[DEBUG] PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"[DEBUG] DATA_DIR: {DATA_DIR}")

__all__ = [
    "PROJECT_ROOT",
    "DATA_DIR",
    "FAISS_DIR",
    "FISH_ROOT",
]
