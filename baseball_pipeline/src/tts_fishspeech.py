# src/tts_fishspeech.py

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import pandas as pd

from .tts_fish_cli import FishSpeechTTS, FishTTSConfig, PROJECT_ROOT


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.replace("\ufeff", "").strip() for c in df.columns]
    return df


def _pick_tts_text(row) -> str:
    """
    llm_text가 있으면 우선 사용, 없으면 orig_text, 그것도 없으면 text 사용.
    """
    for col in ("llm_text", "orig_text", "text"):
        if col in row.index:
            val = row[col]
            if isinstance(val, float) and math.isnan(val):
                continue
            if isinstance(val, str) and val.strip():
                return val.strip()
    return ""


def _infer_video_stem(df: pd.DataFrame, csv_path: Path) -> str:
    if "source_video" in df.columns and df["source_video"].notna().any():
        name = str(df["source_video"].dropna().iloc[0])
        return Path(name).stem
    # fallback: clip.tts_phrases.llm_kanana.csv → clip
    return csv_path.stem.split(".")[0]


def run_tts_batch(
    tts_csv_path: Path | str,
    caster_ref_wav: Path | str,
    analyst_ref_wav: Path | str,
    caster_prompt_text: Optional[str] = None,
    analyst_prompt_text: Optional[str] = None,
    max_rows: Optional[int] = None,
) -> Path:
    """
    llm_outputs 에서 만든 CSV 기준으로
    - role == "caster"  → caster_ref_wav 화자로 TTS
    - 그 외(analyst 등) → analyst_ref_wav 화자로 TTS
    를 생성하고, 결과 wav 경로를 추가한 CSV를 반환.

    Parameters
    ----------
    tts_csv_path : CSV 경로 (예: data/llm_outputs/clip.tts_phrases.llm_kanana.csv)
    caster_ref_wav : 캐스터 참조 음성 wav
    analyst_ref_wav : 해설 참조 음성 wav
    caster_prompt_text / analyst_prompt_text : 각 참조 음성에서 읽은 문장 (없으면 기본 문구 사용)
    max_rows : 일부만 테스트하고 싶을 때 상한 (None이면 전체)
    """
    tts_csv_path = Path(tts_csv_path)
    caster_ref_wav = Path(caster_ref_wav)
    analyst_ref_wav = Path(analyst_ref_wav)

    if not tts_csv_path.exists():
        raise FileNotFoundError(f"TTS CSV가 없음: {tts_csv_path}")
    if not caster_ref_wav.exists():
        raise FileNotFoundError(f"caster_ref_wav 없음: {caster_ref_wav}")
    if not analyst_ref_wav.exists():
        raise FileNotFoundError(f"analyst_ref_wav 없음: {analyst_ref_wav}")

    print(f"[TTS_BATCH] 입력 CSV: {tts_csv_path}")
    df = pd.read_csv(tts_csv_path)
    df = _clean_columns(df)

    video_stem = _infer_video_stem(df, tts_csv_path)
    print(f"[TTS_BATCH] video_stem 추론: {video_stem}")

    # data/llm_outputs/ → data/tts_audio/{video_stem}/
    data_root = tts_csv_path.parent.parent
    tts_root = data_root / "tts_audio" / video_stem
    print(f"[TTS_BATCH] TTS 출력 루트: {tts_root}")
    tts_root.mkdir(parents=True, exist_ok=True)

    # 캐스터 / 해설용 엔진 각각 준비
    caster_cfg = FishTTSConfig(
        prompt_wav=caster_ref_wav,
        prompt_tokens=caster_ref_wav.with_suffix(".npy"),
        prompt_text=caster_prompt_text or "캐스터 프롬프트 음성입니다.",
    )
    analyst_cfg = FishTTSConfig(
        prompt_wav=analyst_ref_wav,
        prompt_tokens=analyst_ref_wav.with_suffix(".npy"),
        prompt_text=analyst_prompt_text or "해설 프롬프트 음성입니다.",
    )

    caster_tts = FishSpeechTTS(caster_cfg)
    analyst_tts = FishSpeechTTS(analyst_cfg)

    # 결과 경로 칼럼 추가
    if "tts_wav_path" not in df.columns:
        df["tts_wav_path"] = ""

    processed = 0
    total = len(df)

    # 역할별 인덱스 나누기
    caster_indices = []
    analyst_indices = []

    for idx, row in df.iterrows():
        role = str(row.get("role", "")).lower()
        if role == "caster":
            caster_indices.append(idx)
        else:
            analyst_indices.append(idx)

    def _process_indices(indices, engine, role_dir: str, processed: int) -> int:
        for idx in indices:
            if max_rows is not None and processed >= max_rows:
                break

            row = df.loc[idx]
            utt_id = str(row.get("utterance_id", f"utt_{idx}"))
            text = _pick_tts_text(row)

            if not text:
                print(f"[TTS_BATCH] 빈 텍스트 → 스킵: utterance_id={utt_id}")
                continue

            out_dir = tts_root / role_dir
            out_dir.mkdir(parents=True, exist_ok=True)
            out_wav = out_dir / f"{utt_id}.wav"

            print(
                f"\n[TTS_BATCH] ({processed+1}/{total}) role={role_dir} "
                f"utterance_id={utt_id}"
            )

            try:
                engine.tts(text, out_wav)
                # data/ 기준 상대 경로로 저장
                try:
                    rel_path = out_wav.relative_to(data_root)
                except ValueError:
                    rel_path = out_wav
                df.at[idx, "tts_wav_path"] = str(rel_path)
                processed += 1
            except Exception as e:
                print(f"[TTS_BATCH] ERROR utterance_id={utt_id}: {e}")

        return processed

    # 1) 캐스터 전부
    processed = _process_indices(caster_indices, caster_tts, "caster", processed)

    # 2) 해설 전부
    processed = _process_indices(analyst_indices, analyst_tts, "analyst", processed)

    # 이하 out_csv 저장 부분은 그대로 유지
    out_csv = tts_csv_path.with_name(tts_csv_path.stem + ".tts_done.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print(f"[TTS_BATCH] 완료 rows={processed} / {total}")
    print(f"[TTS_BATCH] 결과 CSV: {out_csv}")
    return out_csv
