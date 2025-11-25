# src/tts_video_mixer.py

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import subprocess
import shlex
import os
import sys
from typing import Optional

import pandas as pd
from pydub import AudioSegment


@dataclass
class TTSMixConfig:
    csv_path: Path                     # LLM 출력 CSV 경로
    tts_root: Path                     # TTS wav 루트: .../data/tts_audio/clip
    input_video: Path                  # 원본 영상: .../data/input_videos/clip.mp4
    output_video: Path                 # 최종 출력 영상
    work_dir: Optional[Path] = None    # 임시 wav 저장 위치 (None이면 tts_root)
    keep_intermediate: bool = False    # True면 중간 wav 파일 삭제하지 않음

    # 열 이름 커스터마이즈 (기본값은 지금 CSV 기준)
    col_utt: str = "utterance_id"
    col_role: str = "role"
    col_start: str = "start_sec"
    col_end: str = "end_sec"


def _run_ffmpeg(cmd: list[str]) -> None:
    """
    ffmpeg 명령 실행 헬퍼. 에러 시 stderr를 같이 보여줌.
    """
    print("[ffmpeg] running:", " ".join(shlex.quote(c) for c in cmd))
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        print(proc.stdout)
        raise RuntimeError(f"ffmpeg failed with code {proc.returncode}")
    else:
        # 로그가 길면 필요 시 주석 처리해도 됨
        print(proc.stdout)


def extract_audio_from_video(video_path: Path, out_audio_path: Path) -> None:
    """
    원본 mp4에서 오디오만 wav로 추출.
    """
    out_audio_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        str(out_audio_path),
    ]
    _run_ffmpeg(cmd)
    print(f"[INFO] extracted original audio -> {out_audio_path}")


def build_mixed_audio_from_csv(
    cfg: TTSMixConfig,
    orig_audio_path: Path,
    out_audio_path: Path,
) -> None:
    """
    원본 오디오를 타임라인 기준으로 잘라서 재조립:
      - 각 row의 [start_sec, end_sec] 구간은 TTS로 '교체'
      - 그 사이 구간은 원본 오디오 그대로 유지

    => TTS 구간에서는 원본 음성과 겹치지 않고, 중간중간 빈 구간은 원본 소리만 유지.
    """
    out_audio_path.parent.mkdir(parents=True, exist_ok=True)

    # 1) 원본 오디오 로드
    orig = AudioSegment.from_file(orig_audio_path)
    total_len_ms = len(orig)
    print(f"[INFO] original audio length: {total_len_ms / 1000:.3f} sec")

    # 2) CSV 로드 (BOM 제거 포함)
    df = pd.read_csv(cfg.csv_path)
    df.columns = [c.replace("\ufeff", "").strip() for c in df.columns]

    required_cols = {cfg.col_utt, cfg.col_role, cfg.col_start, cfg.col_end}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV에 필요한 컬럼 {required_cols} 중 일부가 없습니다. 현재 컬럼: {df.columns.tolist()}")

    # 시작 시각 기준으로 정렬
    df = df.sort_values(cfg.col_start).reset_index(drop=True)
    print(f"[INFO] CSV rows: {len(df)}")

    tts_root = cfg.tts_root

    # 3) 타임라인 재조립
    result = AudioSegment.silent(duration=0, frame_rate=orig.frame_rate)
    cursor_ms = 0

    for i, row in df.iterrows():
        utt_id = str(row[cfg.col_utt])
        role = str(row[cfg.col_role])
        start_sec = float(row[cfg.col_start])
        end_sec = float(row[cfg.col_end])

        start_ms = int(start_sec * 1000)
        end_ms = int(end_sec * 1000)
        if end_ms <= start_ms:
            print(f"[WARN] 잘못된 구간 (end <= start), 스킵: {utt_id} {start_sec}-{end_sec}")
            continue

        # 타임라인 범위를 벗어나면 잘라주기
        start_ms = max(0, min(start_ms, total_len_ms))
        end_ms = max(0, min(end_ms, total_len_ms))

        # 3-1) cursor ~ start_ms 구간: 원본 오디오 그대로 붙이기
        if start_ms > cursor_ms:
            gap_seg = orig[cursor_ms:start_ms]
            result += gap_seg
        else:
            # start_ms가 cursor보다 앞이면 겹치는 구간이므로, 그냥 덮어쓰기 기준으로 진행
            start_ms = cursor_ms

        # 3-2) TTS 구간: [start_ms, end_ms]는 TTS로 '교체'
        wav_path = tts_root / role / f"{utt_id}.wav"
        target_dur_ms = end_ms - start_ms

        if not wav_path.exists():
            print(f"[WARN] TTS WAV 없음, 이 구간은 원본 사용: {wav_path}")
            # WAV 없으면 이 구간도 원본 사용
            result += orig[start_ms:end_ms]
            cursor_ms = end_ms
            continue

        tts_seg = AudioSegment.from_file(wav_path)

        # TTS 길이를 [start_sec, end_sec] 길이에 맞추기
        if len(tts_seg) > target_dur_ms:
            tts_seg = tts_seg[:target_dur_ms]
        elif len(tts_seg) < target_dur_ms:
            # 부족한 길이는 침묵으로 채워서 원본 음성이 섞이지 않게 함
            pad = AudioSegment.silent(duration=(target_dur_ms - len(tts_seg)), frame_rate=orig.frame_rate)
            tts_seg = tts_seg + pad

        print(
            f"[INFO] segment[{i}]: {wav_path.name} | role={role} | "
            f"start={start_sec:.3f}s ~ {end_sec:.3f}s "
            f"(tts_len={len(tts_seg)/1000:.3f}s)"
        )

        # 이 구간에서는 원본 오디오를 쓰지 않고 TTS만 붙인다
        result += tts_seg
        cursor_ms = end_ms

    # 3-3) 마지막 구간 이후 ~ 원본 끝까지는 다시 원본 오디오
    if cursor_ms < total_len_ms:
        result += orig[cursor_ms:total_len_ms]

    # 4) 최종 믹스 오디오 저장
    result.export(out_audio_path, format="wav")
    print(f"[INFO] mixed(replaced) audio saved -> {out_audio_path} | len={len(result)/1000:.3f} sec")



def mux_video_with_audio(
    input_video: Path,
    input_audio: Path,
    output_video: Path,
) -> None:
    """
    비디오는 원본 그대로 두고, 오디오는 우리가 만든 믹스 오디오로 교체.
    """
    output_video.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_video),
        "-i",
        str(input_audio),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-shortest",
        str(output_video),
    ]
    _run_ffmpeg(cmd)
    print(f"[INFO] final video saved -> {output_video}")


def build_tts_overlay_video(cfg: TTSMixConfig) -> None:
    """
    high-level 편의 함수:
      1) 원본 영상에서 오디오 추출
      2) CSV + TTS wav 기준으로 믹스 오디오 생성
      3) 믹스 오디오를 영상에 덮어쓰기
    """
    work_dir = cfg.work_dir or cfg.tts_root
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    orig_audio_path = work_dir / "orig_audio.wav"
    mixed_audio_path = work_dir / "mixed_tts_audio.wav"

    print("[STEP 1] extract original audio")
    extract_audio_from_video(cfg.input_video, orig_audio_path)

    print("[STEP 2] build mixed audio from CSV + TTS WAVs")
    build_mixed_audio_from_csv(cfg, orig_audio_path, mixed_audio_path)

    print("[STEP 3] mux video with mixed audio")
    mux_video_with_audio(cfg.input_video, mixed_audio_path, cfg.output_video)

    if not cfg.keep_intermediate:
        for p in [orig_audio_path, mixed_audio_path]:
            try:
                p.unlink()
                print(f"[CLEANUP] removed {p}")
            except FileNotFoundError:
                pass
