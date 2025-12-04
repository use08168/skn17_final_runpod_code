# src/demucs.py

from __future__ import annotations

import sys
from pathlib import Path
import subprocess

from src import DATA_DIR  # 야구 프로젝트에서 이미 사용 중인 상수

# ==== 디렉토리 설정 ====
PYTHON_EXEC = sys.executable  # 현재 커널이 사용하는 파이썬 경로
INPUT_VIDEO_DIR       = DATA_DIR / "input_videos"           # clip.mp4 등이 있는 곳
DEMUCS_ROOT           = DATA_DIR / "demucs"
DEMUCS_INPUT_WAV_DIR  = DEMUCS_ROOT / "input_wav"           # mp4 -> wav 추출
DEMUCS_OUTPUT_ROOT    = DEMUCS_ROOT / "outputs"             # demucs 분리 결과

DEMUCS_INPUT_WAV_DIR.mkdir(parents=True, exist_ok=True)
DEMUCS_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)



def extract_audio_from_mp4(
    video_name: str,
    sample_rate: int = 44100,
    channels: int = 2,
) -> Path:
    """
    DATA_DIR/input_videos/{video_name}.mp4 -> DATA_DIR/demucs/input_wav/{video_name}.wav

    Args:
        video_name: "clip.mp4" 같은 파일 이름
        sample_rate: 출력 wav 샘플레이트 (기본 44100)
        channels: 2 = 스테레오, 1 = 모노
    """
    mp4_path = INPUT_VIDEO_DIR / video_name
    wav_path = DEMUCS_INPUT_WAV_DIR / (Path(video_name).stem + ".wav")

    if not mp4_path.exists():
        raise FileNotFoundError(f"입력 비디오를 찾을 수 없습니다: {mp4_path}")

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(mp4_path),
        "-ac", str(channels),
        "-ar", str(sample_rate),
        str(wav_path),
    ]
    print("[ffmpeg]", " ".join(cmd))
    subprocess.run(cmd, check=True)

    return wav_path


def run_demucs_two_stems(
    wav_path: Path,
    device: str = "cuda",
    model: str = "htdemucs",
) -> Path:
    cmd = [
        PYTHON_EXEC,         # "python3" 말고 현재 파이썬
        "-m", "demucs.separate",
        "-n", model,
        "--two-stems=vocals",
        "-d", device,
        "-o", str(DEMUCS_OUTPUT_ROOT),
        str(wav_path),
    ]
    print("[demucs]", " ".join(cmd))
    completed = subprocess.run(cmd, text=True, capture_output=True)

    print("===== demucs stdout =====")
    print(completed.stdout)
    print("===== demucs stderr =====")
    print(completed.stderr)

    if completed.returncode != 0:
        raise RuntimeError(
            f"demucs 실행 실패 (returncode={completed.returncode})\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )

    track_dir = DEMUCS_OUTPUT_ROOT / model / wav_path.stem
    print("[result] track dir:", track_dir)
    print("  -", track_dir / "vocals.wav")
    print("  -", track_dir / "no_vocals.wav")
    return track_dir


def separate_video_with_demucs(
    video_name: str,
    device: str = "cuda",
    model: str = "htdemucs",
) -> Path:
    """
    단일 비디오 파일에 대해
    1) mp4 -> wav 변환
    2) demucs 2 stems 분리
    까지 한 번에 수행.

    Returns:
        track_dir: Demucs 결과 디렉토리
    """
    wav_path = extract_audio_from_mp4(video_name)
    track_dir = run_demucs_two_stems(wav_path, device=device, model=model)
    return track_dir


def separate_all_videos_in_input(
    device: str = "cuda",
    pattern: str = "*.mp4",
):
    """
    DATA_DIR/input_videos 아래의 모든 mp4 에 대해
    demucs 2 stems 분리를 수행.
    """
    mp4_files = sorted(INPUT_VIDEO_DIR.glob(pattern))
    print(f"[info] 발견된 mp4 파일 수: {len(mp4_files)} (경로: {INPUT_VIDEO_DIR})")

    for p in mp4_files:
        video_name = p.name
        print(f"\n===== {video_name} =====")
        separate_video_with_demucs(video_name, device=device)
