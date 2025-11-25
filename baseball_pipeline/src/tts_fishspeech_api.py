# src/tts_fishspeech_api.py
from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Union

import base64
import pandas as pd
import requests


# ---------------------------
# 유틸: 텍스트 선택 (caster/analyst 규칙)
# ---------------------------

def _choose_text_by_role(row) -> str:
    """
    CSV 한 줄(row)에서 실제 TTS에 쓸 텍스트를 고른다.

    - role == 'caster'  → orig_text
    - role == 'analyst' → llm_text
    """
    role_raw = str(getattr(row, "role", "")).strip().lower()

    if role_raw == "caster":
        raw = getattr(row, "orig_text", "")
    elif role_raw == "analyst":
        raw = getattr(row, "llm_text", "")
    else:
        # 혹시 이상한 값이면 analyst 룰로
        raw = getattr(row, "llm_text", "")

    if pd.isna(raw):
        return ""
    s = str(raw).strip()
    if not s or s.lower() == "nan":
        return ""
    return s


# ---------------------------
# 유틸: 참조 wav → base64 리스트
# ---------------------------

def _load_audio_b64_list(
    wav_paths: List[Union[str, Path]],
    label: str,
) -> List[str]:
    """
    여러 개의 wav 경로를 받아서,
    존재하는 파일만 base64 로 인코딩해서 리스트로 반환.

    아무것도 없으면 예외.
    """
    valid_paths: List[Path] = []
    for p in wav_paths:
        p = Path(p)
        if p.exists():
            valid_paths.append(p)
        else:
            print(f"[TTS_API] (skip missing {label} ref) {p}")

    if not valid_paths:
        raise FileNotFoundError(f"[TTS_API] {label} 참조 wav가 하나도 없습니다.")

    b64_list: List[str] = []
    for p in valid_paths:
        with open(p, "rb") as f:
            data = f.read()
        b64_list.append(base64.b64encode(data).decode("ascii"))
        print(f"[TTS_API] loaded {label} ref:", p)

    return b64_list


def _build_references(
    voice: str,
    caster_refs_b64: List[str],
    analyst_refs_b64: List[str],
) -> List[dict]:
    """
    Fish-Speech /v1/tts 의 references 필드 생성.

    voice:
      - "caster"  → caster_refs_b64 사용
      - "analyst" → analyst_refs_b64 사용
    """
    if voice == "caster":
        refs_b64 = caster_refs_b64
        base_text = "캐스터 프롬프트"
    else:
        refs_b64 = analyst_refs_b64
        base_text = "해설 프롬프트"

    refs = [
        {
            "audio": b64,
            "text": f"{base_text} {i+1}",
        }
        for i, b64 in enumerate(refs_b64)
    ]
    return refs


# ---------------------------
# 실제 API 호출
# ---------------------------

def call_fish_tts_api(
    text: str,
    references: List[dict],
    api_url: str,
    out_wav_path: Path | str,
) -> Path:
    """
    Fish-Speech api_server 의 /v1/tts 를 호출해서
    단일 문장을 wav 로 저장.
    """
    out_wav_path = Path(out_wav_path)
    out_wav_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "text": text,
        "chunk_length": 200,        # 문장이 길 때 쪼개는 기준
        "format": "wav",
        "references": references,
        "reference_id": None,
        "seed": 42,
        "use_memory_cache": "on",   # 참조 캐시 → 반복 호출 빨라짐
        "normalize": True,
        "streaming": False,
        "max_new_tokens": 1024,
        "top_p": 0.7,
        "repetition_penalty": 1.1,
        "temperature": 0.7,
    }

    headers = {
        "accept": "*/*",
        "Content-Type": "application/json",
    }

    resp = requests.post(api_url, json=payload, headers=headers, timeout=(10, 600))
    if resp.status_code != 200:
        try:
            print("[TTS_API] error body:", resp.json())
        except Exception:
            print("[TTS_API] raw error (len):", len(resp.content))
        resp.raise_for_status()

    with open(out_wav_path, "wb") as f:
        f.write(resp.content)

    return out_wav_path


# ---------------------------
# 배치 실행: CSV + 여러 참조 wav
# ---------------------------

def run_tts_batch_via_api(
    tts_csv_path: Path | str,
    caster_ref_wavs: Union[Path, str, List[Union[Path, str]]],
    analyst_ref_wavs: Union[Path, str, List[Union[Path, str]]],
    api_url: str = "http://127.0.0.1:8080/v1/tts",
    max_rows: Optional[int] = None,
) -> Path:
    """
    clip.tts_phrases.llm_kanana.csv 를 읽어서
    Fish-Speech api_server(/v1/tts)에 배치로 요청.

    - role == "caster"  → orig_text 사용 + 캐스터 참조 wav(1~N개)
    - role == "analyst" → llm_text 사용 + 해설 참조 wav(1~N개)

    `caster_ref_wavs`, `analyst_ref_wavs`:
      - Path 또는 str 하나만 줘도 되고
      - [Path, Path, Path] 리스트로 여러 개 줘도 됨
      - 존재하는 파일만 자동으로 사용

    출력:
      data/tts_audio/{video_stem}/{utterance_id}.wav
      + tts_wav_path 컬럼이 추가된 CSV 저장
    """
    tts_csv_path = Path(tts_csv_path)
    df = pd.read_csv(tts_csv_path)
    df.columns = [c.replace("\ufeff", "").strip() for c in df.columns]

    required_cols = {"utterance_id", "role", "orig_text", "llm_text"}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"[TTS_API] CSV에 {required_cols} 가 필요합니다. "
            f"현재 컬럼: {df.columns.tolist()}"
        )

    # video_stem 추론
    if "source_video" in df.columns and pd.notna(df["source_video"].iloc[0]):
        first_vid = str(df["source_video"].iloc[0])
        video_stem = Path(first_vid).stem
    else:
        video_stem = tts_csv_path.stem.split(".")[0]

    print("[TTS_API] 입력 CSV:", tts_csv_path)
    print("[TTS_API] video_stem:", video_stem)

    data_dir = tts_csv_path.parent.parent  # data/llm_outputs 기준
    tts_audio_dir = data_dir / "tts_audio" / video_stem
    tts_audio_dir.mkdir(parents=True, exist_ok=True)
    print("[TTS_API] 출력 디렉토리:", tts_audio_dir)

    # ---- 참조 wav 경로 정규화 (하나만 들어와도 리스트로 바꾸기) ----
    if not isinstance(caster_ref_wavs, (list, tuple)):
        caster_ref_list: List[Union[str, Path]] = [caster_ref_wavs]
    else:
        caster_ref_list = list(caster_ref_wavs)

    if not isinstance(analyst_ref_wavs, (list, tuple)):
        analyst_ref_list: List[Union[str, Path]] = [analyst_ref_wavs]
    else:
        analyst_ref_list = list(analyst_ref_wavs)

    # ---- 실제 파일 존재하는 것만 base64 로딩 ----
    caster_refs_b64 = _load_audio_b64_list(caster_ref_list, label="caster")
    analyst_refs_b64 = _load_audio_b64_list(analyst_ref_list, label="analyst")

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

        # 1) role에 맞춰 텍스트 선택
        text = _choose_text_by_role(row)
        if not text:
            print(f"[TTS_API] (skip empty) idx={idx} utt={utt_id} role={role_raw}")
            continue

        # 2) voice (서버 쪽 역할)
        voice = "caster" if role == "caster" else "analyst"

        # 3) 역할에 맞는 references 구성
        references = _build_references(
            voice=voice,
            caster_refs_b64=caster_refs_b64,
            analyst_refs_b64=analyst_refs_b64,
        )

        out_wav = tts_audio_dir / f"{utt_id}.wav"

        print(f"[TTS_API] ({idx}) role={role} voice={voice} utt={utt_id}")
        print(f"[TTS_API] Text: {text[:80]}{'...' if len(text) > 80 else ''}")

        try:
            call_fish_tts_api(
                text=text,
                references=references,
                api_url=api_url,
                out_wav_path=out_wav,
            )
            df.at[idx, "tts_wav_path"] = str(out_wav)
        except Exception as e:
            print(f"[TTS_API] ERROR utt={utt_id}: {e}")

    out_csv = tts_audio_dir / f"{video_stem}.tts_phrases.with_tts_api.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print("[TTS_API] saved CSV:", out_csv)

    return out_csv
