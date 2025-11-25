# src/tts_fishspeech_npy_api.py

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import requests


def call_tts_npy_server(
    text: str,
    voice: str,
    api_url: str,
    out_wav_path: Path | str,
) -> Path:
    """ .npy 프롬프트 기반 TTS 서버(/v1/tts)에 요청해서 wav 파일로 저장 """
    out_wav_path = Path(out_wav_path)
    out_wav_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "text": text,
        "role": voice,          # "caster" or "analyst"
        "utterance_id": out_wav_path.stem,
    }

    # connect 10초, read 30분
    resp = requests.post(api_url, json=payload, timeout=(10, 1800))
    if resp.status_code != 200:
        try:
            print("[TTS_NPY_CLIENT] Error body:", resp.json())
        except Exception:
            print("[TTS_NPY_CLIENT] Raw error len:", len(resp.content))
        resp.raise_for_status()

    with open(out_wav_path, "wb") as f:
        f.write(resp.content)

    return out_wav_path


def run_tts_batch_via_npy_server(
    tts_csv_path: Path | str,
    api_url: str = "http://127.0.0.1:8080/v1/tts",
    max_rows: Optional[int] = None,
) -> Path:
    """
    clip.tts_phrases.llm_kanana.csv 를 읽어서

      - caster  → orig_text
      - analyst → llm_text

    를 TTS 서버로 보내고,
    data/tts_audio/{video_stem}/{utterance_id}.wav 로 저장.
    """

    tts_csv_path = Path(tts_csv_path)
    df = pd.read_csv(tts_csv_path)
    df.columns = [c.replace("\ufeff", "").strip() for c in df.columns]

    required_cols = {"utterance_id", "role", "orig_text", "llm_text"}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"[TTS_NPY_CLIENT] 필수 컬럼 없음: {required_cols} / 실제: {df.columns.tolist()}"
        )

    # video_stem 추론
    if "source_video" in df.columns and pd.notna(df["source_video"].iloc[0]):
        first_vid = str(df["source_video"].iloc[0])
        video_stem = Path(first_vid).stem
    else:
        # 예: clip.tts_phrases.llm_kanana.csv → clip
        video_stem = tts_csv_path.stem.split(".")[0]

    print("[TTS_NPY_CLIENT] 입력 CSV:", tts_csv_path)
    print("[TTS_NPY_CLIENT] video_stem:", video_stem)

    data_dir = tts_csv_path.parent.parent  # data/llm_outputs 기준
    tts_audio_dir = data_dir / "tts_audio" / video_stem
    tts_audio_dir.mkdir(parents=True, exist_ok=True)
    print("[TTS_NPY_CLIENT] 출력 디렉토리:", tts_audio_dir)

    if "tts_wav_path" not in df.columns:
        df["tts_wav_path"] = ""

    rows_iter = list(df.itertuples(index=True))
    if max_rows is not None:
        rows_iter = rows_iter[:max_rows]

    for row in rows_iter:
        idx = row.Index
        utt_id = row.utterance_id
        role_raw = str(row.role).strip()
        role = role_raw.lower()

        # --- 1) 역할에 따라 텍스트 컬럼 하나만 선택 ---
        if role == "caster":
            raw = row.orig_text
            voice = "caster"
        elif role == "analyst":
            raw = row.llm_text
            voice = "analyst"
        else:
            # 혹시 모르는 이상 케이스: analyst처럼 llm_text 우선
            raw = row.llm_text
            voice = "analyst"

        # --- 2) NaN / 빈 문자열 / "nan" 필터 ---
        text = "" if pd.isna(raw) else str(raw).strip()
        if not text or text.lower() == "nan":
            print(f"[TTS_NPY_CLIENT] (skip empty/nan) idx={idx} utt={utt_id} role={role_raw}")
            continue

        out_wav = tts_audio_dir / f"{utt_id}.wav"

        print(f"[TTS_NPY_CLIENT] ({idx}) role={role} voice={voice} utt={utt_id}")
        print(f"[TTS_NPY_CLIENT] Text: {text[:80]}{'...' if len(text) > 80 else ''}")

        try:
            call_tts_npy_server(
                text=text,
                voice=voice,
                api_url=api_url,
                out_wav_path=out_wav,
            )
            df.at[idx, "tts_wav_path"] = str(out_wav)
        except Exception as e:
            print(f"[TTS_NPY_CLIENT] ERROR utt={utt_id}: {e}")

    out_csv = tts_audio_dir / f"{video_stem}.tts_phrases.with_tts_npy.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print("[TTS_NPY_CLIENT] saved CSV:", out_csv)

    return out_csv
