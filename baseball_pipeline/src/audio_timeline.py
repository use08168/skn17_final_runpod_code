# src/audio_timeline.py

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from pydub import AudioSegment


def _find_tts_wav_for_utt(
    tts_audio_dir: Path,
    utt_id: str,
    role: Optional[str] = None,
) -> Optional[Path]:
    """
    utterance_id 에 대응하는 wav 파일을 찾는다.

    우선순위:
    1) {tts_audio_dir}/{utt_id}.wav
    2) {tts_audio_dir}/{role}/{utt_id}.wav  (role=caster/analyst 등)
    """
    cand = tts_audio_dir / f"{utt_id}.wav"
    if cand.exists():
        return cand

    if role:
        role = str(role).strip().lower()
        cand2 = tts_audio_dir / role / f"{utt_id}.wav"
        if cand2.exists():
            return cand2

    # 추가로 폴더를 더 뒤지고 싶으면 여기에 glob 로직을 넣을 수 있음
    return None


def _stretch_segment_to_duration(
    seg: AudioSegment,
    target_ms: float,
    max_stretch_factor: float = 1.3,
) -> AudioSegment:
    """
    seg 의 길이가 target_ms 보다 짧을 때,
    최대 max_stretch_factor 배까지 타임스트레치해서 길이를 맞추려는 함수.

    pydub 의 frame_rate 조작을 이용한 간단한 방식.
    """
    orig_ms = len(seg)
    if orig_ms <= 0:
        return seg

    ratio = target_ms / orig_ms

    # 늘리는 비율이 너무 크면 무리하게 맞추지 않고 원본 유지
    if ratio <= 1.0 or ratio > max_stretch_factor:
        return seg

    new_frame_rate = int(seg.frame_rate / ratio)
    stretched = seg._spawn(seg.raw_data, overrides={"frame_rate": new_frame_rate})
    stretched = stretched.set_frame_rate(seg.frame_rate)
    return stretched


def build_tts_audio_timeline(
    llm_csv_path: Path | str,
    tts_audio_dir: Path | str,
    out_wav_path: Path | str,
    fade_out_ms: int = 300,
    max_stretch_factor: float = 1.3,
) -> Path:
    """
    LLM 결과 CSV + TTS 개별 wav 들을 사용해
    전체 타임라인 기반의 하나의 wav 파일을 만든다.

    - llm_csv_path: clip.tts_phrases.llm_kanana.csv 등
      (필수 컬럼: utterance_id, role, start_sec, end_sec)
    - tts_audio_dir: 개별 TTS wav 들이 있는 디렉토리
      (기본: data/tts_audio/{video_stem})
    - out_wav_path: 생성할 전체 wav 경로
    - fade_out_ms: 구간 끝에서 적용할 페이드아웃 길이(ms)
    - max_stretch_factor: 원본 TTS 가 짧을 경우,
      최대 몇 배까지 타임스트레치 허용할지 (1.3 = 최대 30% 늘리기)

    동작:
    - 전체 길이 = CSV 에서 max(end_sec) 기준으로 계산
    - 해당 길이의 silent 오디오를 만들고
    - 각 utterance 를 start_sec 위치에 overlay
    - 길이가 너무 길면 target_duration 에 맞춰 자르고 페이드아웃
    - 길이가 너무 짧으면 (허용 범위 안에서만) 살짝 늘려서 맞춤
    """
    llm_csv_path = Path(llm_csv_path)
    tts_audio_dir = Path(tts_audio_dir)
    out_wav_path = Path(out_wav_path)

    df = pd.read_csv(llm_csv_path)
    df.columns = [c.replace("\ufeff", "").strip() for c in df.columns]

    required_cols = {"utterance_id", "role", "start_sec", "end_sec"}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"[TIMELINE] CSV에 {required_cols} 컬럼이 필요합니다. "
            f"현재 컬럼: {df.columns.tolist()}"
        )

    # 시간 정렬
    df = df.dropna(subset=["start_sec", "end_sec"]).copy()
    df["start_sec"] = df["start_sec"].astype(float)
    df["end_sec"] = df["end_sec"].astype(float)
    df = df.sort_values("start_sec")

    if df.empty:
        raise ValueError("[TIMELINE] CSV 에 유효한 구간이 없습니다.")

    total_duration_sec = float(df["end_sec"].max()) + 1.0  # 여유 1초
    total_duration_ms = int(total_duration_sec * 1000)

    print(f"[TIMELINE] total_duration_sec ≈ {total_duration_sec:.2f} s")
    base = AudioSegment.silent(duration=total_duration_ms)  # 44.1kHz, mono 기본

    for row in df.itertuples(index=False):
        utt_id = str(getattr(row, "utterance_id"))
        role = getattr(row, "role", "")
        start_sec = float(getattr(row, "start_sec"))
        end_sec = float(getattr(row, "end_sec"))
        if end_sec <= start_sec:
            continue

        start_ms = int(start_sec * 1000)
        target_ms = int((end_sec - start_sec) * 1000)

        wav_path = _find_tts_wav_for_utt(tts_audio_dir, utt_id, role=role)
        if wav_path is None:
            print(f"[TIMELINE] WARN: wav not found for utt={utt_id}, role={role}")
            continue

        seg = AudioSegment.from_file(wav_path)
        orig_ms = len(seg)

        # 길이 맞추기: 원본이 너무 길면 자르면서 페이드아웃
        if orig_ms > target_ms:
            # 페이드아웃 적용: 마지막 fade_out_ms 구간
            cut = seg[:target_ms]
            if fade_out_ms > 0:
                fade_ms = min(fade_out_ms, target_ms // 2)
                if fade_ms > 0:
                    cut = cut.fade_out(fade_ms)
            seg = cut
        else:
            # 원본이 짧을 경우, 허용 범위 안에서만 늘리기
            if max_stretch_factor is not None and max_stretch_factor > 1.0:
                seg = _stretch_segment_to_duration(
                    seg, target_ms=target_ms, max_stretch_factor=max_stretch_factor
                )
                # 늘린 뒤에도 너무 길면 다시 한 번 잘라주고 페이드아웃
                if len(seg) > target_ms:
                    cut = seg[:target_ms]
                    if fade_out_ms > 0:
                        fade_ms = min(fade_out_ms, target_ms // 2)
                        if fade_ms > 0:
                            cut = cut.fade_out(fade_ms)
                    seg = cut

        # base 위에 overlay
        print(
            f"[TIMELINE] overlay utt={utt_id} role={role} "
            f"start={start_sec:.2f}s dur≈{len(seg)/1000:.2f}s"
        )
        base = base.overlay(seg, position=start_ms)

    out_wav_path.parent.mkdir(parents=True, exist_ok=True)
    base.export(out_wav_path, format="wav")
    print("[TIMELINE] saved:", out_wav_path)
    return out_wav_path


def mux_tts_audio_to_video(
    input_video_path: Path | str,
    tts_audio_path: Path | str,
    out_video_path: Path | str,
    mute_original: bool = True,
) -> Path:
    """
    ffmpeg 를 이용해:

    - 원본 비디오의 영상을 그대로 사용하고
    - 오디오는 TTS wav 를 붙인다.
      - mute_original=True  → 원본 음소거 + TTS만 사용
      - mute_original=False → 원본 + TTS를 amix 로 섞기
    """
    input_video_path = Path(input_video_path)
    tts_audio_path = Path(tts_audio_path)
    out_video_path = Path(out_video_path)

    out_video_path.parent.mkdir(parents=True, exist_ok=True)

    if mute_original:
        # 영상은 그대로, 오디오는 TTS만
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(input_video_path),
            "-i",
            str(tts_audio_path),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            str(out_video_path),
        ]
    else:
        # 원본 + TTS 를 amix 로 섞기
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(input_video_path),
            "-i",
            str(tts_audio_path),
            "-filter_complex",
            "[0:a][1:a]amix=inputs=2:duration=longest:dropout_transition=0[aout]",
            "-map",
            "0:v:0",
            "-map",
            "[aout]",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            str(out_video_path),
        ]

    print("[MUX] CMD:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print("[MUX] saved video:", out_video_path)
    return out_video_path
