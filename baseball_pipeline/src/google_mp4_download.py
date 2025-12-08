from __future__ import annotations

import re
import gdown
from pathlib import Path

from src import DATA_DIR
INPUT_VIDEO_DIR = DATA_DIR / "input_videos"

def download_gdrive_video(gdrive_url: str, dest_name: str | None = None) -> Path:
    """구글 드라이브 공유 링크에서 영상 다운로드"""
    m = re.search(r"id=([a-zA-Z0-9_-]+)", gdrive_url)
    if not m:
        m = re.search(r"/d/([a-zA-Z0-9_-]+)/", gdrive_url)
    
    if not m:
        raise ValueError(f"구글 드라이브 링크에서 file id를 찾지 못했습니다.\n현재 url: {gdrive_url}")
    
    file_id = m.group(1)
    
    if dest_name is None:
        dest_name = f"{file_id}.mp4"
    
    out_path = INPUT_VIDEO_DIR / dest_name
    
    if out_path.exists():
        print(f"[GDRIVE] 파일이 이미 존재합니다: {out_path}")
        return out_path
    
    url = f"https://drive.google.com/uc?id={file_id}"
    print("[GDRIVE] file_id:", file_id)
    print("[GDRIVE] url    :", url)
    print("[GDRIVE] output :", out_path)
    
    gdown.download(url, str(out_path), quiet=False)
    
    print("[GDRIVE] 다운로드 완료:", out_path)
    return out_path
