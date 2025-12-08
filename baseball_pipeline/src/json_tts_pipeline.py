# src/json_tts_pipeline.py

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd

from src.tts_fishspeech_api import run_tts_batch_via_api
from src.llm_preprocess_align import preprocess_and_align_llm_csv
from src.tts_align_wsola_gap import build_wsola_tts_timeline_gap


def json_sets_to_tts_csv(
    json_path: Path | str,
    video_path: Path | str,
    out_csv_path: Path | str | None = None,
) -> Path:
    """
    최종 세트 JSON (set_id, caster_text, analyst_text, set_start_sec, set_end_sec ...)
    -> TTS 파이프라인에서 사용하는 CSV 포맷으로 변환.

    CSV 스키마:
      - utterance_id : 고유 ID (set_id + 역할)
      - set_id       : 세트 ID
      - video_id     : 원본 video_id
      - source_video : 원본 비디오 풀 경로 (TTS용 metadata)
      - role         : 'caster' 또는 'analyst'
      - orig_text    : 캐스터 텍스트 (또는 analyst 에서는 비워 둠)
      - llm_text     : 해설 텍스트 (analyst용; 캐스터는 orig_text 복사)
      - start_sec, end_sec : 초기 슬롯 (세트 내에서 캐스터/해설 비율로 나눔)
    """
    json_path = Path(json_path)
    video_path = Path(video_path)

    with json_path.open("r", encoding="utf-8") as f:
        sets: List[Dict[str, Any]] = json.load(f)

    rows: List[Dict[str, Any]] = []

    for s in sets:
        set_id = str(s["set_id"])
        video_id = str(s.get("video_id", ""))
        set_start = float(s["set_start_sec"])
        set_end = float(s["set_end_sec"])
        total_dur = max(set_end - set_start, 0.2)  # 너무 짧으면 최소 0.2초

        caster_text = (s.get("caster_text") or "").strip()
        analyst_text_raw = s.get("analyst_text", None)
        analyst_text = analyst_text_raw.strip() if isinstance(analyst_text_raw, str) else None

        # 캐릭터 수 기반 비율 계산 (공백 제거)
        caster_chars = len("".join(caster_text.split()))
        analyst_chars = len("".join(analyst_text.split())) if analyst_text else 0

        # ---- 캐스터 슬롯 ----
        if caster_text:
            if analyst_text:
                # 캐스터/해설 둘 다 있는 경우 → 글자수 비율로 시간 분배
                total_chars = max(caster_chars + analyst_chars, 1)
                caster_ratio = caster_chars / total_chars
                # 너무 극단적이지 않게 클램핑 (25~60%)
                caster_ratio = min(max(caster_ratio, 0.25), 0.6)
                caster_dur = total_dur * caster_ratio
            else:
                # 해설이 없으면 전체 세트 시간 사용
                caster_dur = total_dur

            caster_start = set_start
            caster_end = caster_start + caster_dur

            rows.append(
                {
                    "utterance_id": f"{set_id}_C",
                    "set_id": set_id,
                    "video_id": video_id,
                    "source_video": str(video_path),
                    "role": "caster",
                    "orig_text": caster_text,
                    "llm_text": caster_text,  # 캐스터는 orig == llm 로 둬도 무방
                    "start_sec": caster_start,
                    "end_sec": caster_end,
                }
            )

        # ---- 해설 슬롯 ----
        if analyst_text:
            # 해설이 있는 경우에만 추가
            # 위에서 caster_dur 를 계산해두었다고 가정
            if caster_text:
                analyst_start = caster_end
                analyst_end = set_end
            else:
                # 캐스터가 없는 세트 (거의 없겠지만 안전)
                analyst_start = set_start
                analyst_end = set_end

            rows.append(
                {
                    "utterance_id": f"{set_id}_A",
                    "set_id": set_id,
                    "video_id": video_id,
                    "source_video": str(video_path),
                    "role": "analyst",
                    "orig_text": "",          # analyst 는 llm_text 기준 사용
                    "llm_text": analyst_text,
                    "start_sec": analyst_start,
                    "end_sec": analyst_end,
                }
            )

    if not rows:
        raise ValueError("[JSON_TTS] 변환 결과가 비어 있습니다. JSON 내용을 확인하세요.")

    df = pd.DataFrame(rows)

    if out_csv_path is None:
        # 예: vocals_timeline_set_split_scoreboard_pakchanho.json
        #  -> vocals.tts_phrases.from_json.csv
        video_stem = video_path.stem
        out_csv_path = json_path.with_name(f"{video_stem}.tts_phrases.from_json.csv")

    out_csv_path = Path(out_csv_path)
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv_path, index=False, encoding="utf-8-sig")

    print(f"[JSON_TTS] JSON → CSV 변환 완료: {out_csv_path}")
    print(f"[JSON_TTS] 총 발화 수: {len(df)} (rows)")
    return out_csv_path


def run_full_tts_pipeline_from_json(
    json_sets_path: Path | str,
    video_path: Path | str,
    caster_ref_wavs,
    analyst_ref_wavs,
    fish_api_url: str = "http://127.0.0.1:8080/v1/tts",
    *,
    # LLM 정렬 옵션
    min_text_chars: int = 2,
    merge_same_role: bool = True,
    merge_gap_thresh_sec: float = 0.25,
    merge_short_thresh_sec: float = 1.0,
    min_gap_sec: float = 0.02,
    caster_extra_ratio: float = 0.2,
    analyst_extra_ratio: float = 2.0,
    max_analyst_expand_sec: float = 7.0,
    analyst_priority_min_overlap_sec: float = 0.5,
    # WSOLA 옵션
    min_gap_ms: int = 60,
    tail_margin_ms: int = 80,
    caster_max_speedup: float = 1.3,
    analyst_max_speedup: float = 1.8,
) -> tuple[Path, Path, Path]:
    """
    JSON 세트 파일 → (TTS CSV → TTS wav → 정렬 CSV → WSOLA 타임라인 wav)까지 한 번에 실행.

    반환:
      (final_tts_wav_path, aligned_csv_path, tts_csv_with_paths)
    """
    json_sets_path = Path(json_sets_path)
    video_path = Path(video_path)
    video_stem = video_path.stem

    # 1) JSON → TTS CSV
    tts_format_csv = json_sets_to_tts_csv(
        json_path=json_sets_path,
        video_path=video_path,
        out_csv_path=None,
    )

    # 2) TTS (Fish-Speech API)
    tts_csv_with_paths = run_tts_batch_via_api(
        tts_csv_path=tts_format_csv,
        caster_ref_wavs=caster_ref_wavs,
        analyst_ref_wavs=analyst_ref_wavs,
        api_url=fish_api_url,
    )

    # 3) 시간 정렬 전처리
    aligned_csv = preprocess_and_align_llm_csv(
        llm_csv_path=tts_csv_with_paths,
        out_csv_path=None,
        start_col="start_sec",
        end_col="end_sec",
        role_col="role",
        uttid_col="utterance_id",
        min_text_chars=min_text_chars,
        merge_same_role=merge_same_role,
        merge_gap_thresh_sec=merge_gap_thresh_sec,
        merge_short_thresh_sec=merge_short_thresh_sec,
        min_gap_sec=min_gap_sec,
        caster_extra_ratio=caster_extra_ratio,
        analyst_extra_ratio=analyst_extra_ratio,
        max_analyst_expand_sec=max_analyst_expand_sec,
        analyst_priority_min_overlap_sec=analyst_priority_min_overlap_sec,
    )

    # 4) WSOLA 타임라인 합성
    # run_tts_batch_via_api 에서:
    #   data_dir = tts_csv_path.parent.parent  (예: data/llm_outputs -> data)
    #   tts_audio_dir = data_dir / "tts_audio" / video_stem
    tts_audio_dir = Path(tts_csv_with_paths).parent  # = data/tts_audio/{video_stem}
    tts_timeline_wav = tts_audio_dir / f"{video_stem}.tts_timeline.wav"

    final_tts_wav = build_wsola_tts_timeline_gap(
        llm_csv_path=aligned_csv,
        tts_audio_dir=tts_audio_dir,
        out_wav_path=tts_timeline_wav,
        start_col="start_sec",
        end_col="end_sec",
        role_col="role",
        uttid_col="utterance_id",
        min_gap_ms=min_gap_ms,
        tail_margin_ms=tail_margin_ms,
        caster_max_speedup=caster_max_speedup,
        analyst_max_speedup=analyst_max_speedup,
    )

    return final_tts_wav, aligned_csv, tts_csv_with_paths