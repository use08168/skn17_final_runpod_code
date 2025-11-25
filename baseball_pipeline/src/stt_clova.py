# src/stt_clova.py

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import pandas as pd
import requests

# -----------------------------
# 0) 엑셀(stt.xlsx) → boostings 리스트 변환
# -----------------------------
def load_boostings_from_xlsx(xlsx_path: str | Path) -> List[Dict]:
    """
    CLOVA Speech boostings 형식으로 변환:
    [
      {"words": "키워드1,키워드2", "weight": 3},
      {"words": "이순철", "weight": 5},
      ...
    ]
    """
    xlsx_path = Path(xlsx_path)
    if not xlsx_path.exists():
        raise FileNotFoundError(f"키워드 엑셀 파일을 찾을 수 없습니다: {xlsx_path}")

    df = pd.read_excel(xlsx_path)

    # 키워드 컬럼 이름 추론
    if "words" in df.columns:
        word_col = "words"
    elif "keyword" in df.columns:
        word_col = "keyword"
    elif "키워드" in df.columns:
        word_col = "키워드"
    else:
        raise ValueError("엑셀에 'words', 'keyword', '키워드' 중 하나 컬럼이 있어야 합니다.")

    # 가중치 컬럼(선택)
    weight_col = None
    for cand in ["weight", "가중치", "weight(1~5)"]:
        if cand in df.columns:
            weight_col = cand
            break

    boostings: List[Dict] = []
    for _, row in df.iterrows():
        if pd.isna(row[word_col]):
            continue

        entry: Dict = {"words": str(row[word_col]).strip()}

        if weight_col is not None and not pd.isna(row[weight_col]):
            entry["weight"] = float(row[weight_col])  # 0~5.0 사이 실수

        boostings.append(entry)

    return boostings


# -----------------------------
# 1) CLOVA Speech 클라이언트
# -----------------------------
class ClovaSpeechClient:
    """
    NAVER CLOVA Speech REST 클라이언트 (long-sentence)
    - invoke_url: 콘솔에서 복사한 Invoke URL
    - secret_key: 콘솔에서 발급받은 Secret Key
    """

    def __init__(self, invoke_url: str, secret_key: str):
        self.invoke_url = invoke_url.rstrip("/")
        self.secret = secret_key

    def recognize_local_file(
        self,
        audio_path: str | Path,
        *,
        language: str = "ko-KR",
        completion: str = "sync",
        diarization_enable: bool = True,
        speaker_count_min: int = -1,        # -1 / -1 = 화자 수 자동
        speaker_count_max: int = -1,
        boostings: Optional[List[Dict]] = None,   # 엑셀/직접 지정 부스팅
        use_domain_boostings: bool = False,       # 도메인 부스팅 사용할지
        timeout: int = 600,
    ) -> dict:
        """
        로컬에 있는 음성/영상 파일을 그대로 업로드해서 인식.
        (mp4 / wav / mp3 다 가능)
        """

        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {audio_path}")

        # boostings + use_domain_boostings 동시 사용 방지
        if boostings is not None and use_domain_boostings:
            raise ValueError("boostings와 use_domain_boostings는 동시에 사용할 수 없습니다.")

        params: Dict = {
            "language": language,
            "completion": completion,
            "diarization": {
                "enable": diarization_enable,
                "speakerCountMin": speaker_count_min,
                "speakerCountMax": speaker_count_max,
            },
        }

        if boostings is not None:
            params["boostings"] = boostings
        elif use_domain_boostings:
            params["useDomainBoostings"] = True

        headers = {
            "Accept": "application/json; charset=UTF-8",
            "X-CLOVASPEECH-API-KEY": self.secret,
        }

        url = self.invoke_url + "/recognizer/upload"

        with open(audio_path, "rb") as f:
            files = {
                "media": (audio_path.name, f, "application/octet-stream"),
                "params": (
                    None,
                    json.dumps(params, ensure_ascii=False).encode("utf-8"),
                    "application/json",
                ),
            }

            resp = requests.post(
                url,
                headers=headers,
                files=files,
                timeout=timeout,
            )

        resp.raise_for_status()
        return resp.json()


# -----------------------------
# 2) JSON → segment DataFrame
# -----------------------------
def clova_segments_to_dataframe(result_json: dict) -> pd.DataFrame:
    """
    CLOVA Speech long-sentence 결과 JSON에서
    화자/타임스탬프/텍스트를 뽑아서 DataFrame으로 변환.
    """
    segments = result_json.get("segments", []) or []

    rows = []
    for i, seg in enumerate(segments, start=1):
        start_ms = seg.get("start")
        end_ms = seg.get("end")
        speaker_info = seg.get("speaker") or {}
        diar_info = seg.get("diarization") or {}

        rows.append(
            {
                "segment_id": i,
                "speaker_label": speaker_info.get("label") or diar_info.get("label"),
                "speaker_name": speaker_info.get("name"),
                "start_ms": start_ms,
                "end_ms": end_ms,
                "start_sec": start_ms / 1000.0 if isinstance(start_ms, (int, float)) else None,
                "end_sec": end_ms / 1000.0 if isinstance(end_ms, (int, float)) else None,
                "text": seg.get("textEdited") or seg.get("text"),
                "confidence": seg.get("confidence"),
                "words": seg.get("words"),
            }
        )

    df = pd.DataFrame(rows)
    return df


# -----------------------------
# 3) words 기반 phrase 쪼개기
# -----------------------------
EOS_SUFFIXES = (
    "입니다",
    "입니다.",
    "있습니다",
    "있습니다.",
    "합니다",
    "합니다.",
    "했어요.",
    "했죠.",
    "했지요.",
    "같습니다.",
    "되겠습니다.",
    "되겠는데요.",
)

def clova_words_to_phrases_df(result_json: dict) -> pd.DataFrame:
    """
    result_json['segments'][*]['words']를 이용해서
    더 짧은 '구절(phrase)' 단위로 자른 DataFrame 생성.
    """
    segments = result_json.get("segments", []) or []
    rows = []

    for seg_idx, seg in enumerate(segments, start=1):
        speaker_info = seg.get("speaker") or {}
        diar_info = seg.get("diarization") or {}

        speaker_label = speaker_info.get("label") or diar_info.get("label")
        speaker_name = speaker_info.get("name")
        seg_conf = seg.get("confidence")

        words = seg.get("words") or []
        # words가 없으면 segment 전체를 하나의 phrase로 사용
        if not words:
            start_ms = seg.get("start")
            end_ms = seg.get("end")
            rows.append(
                {
                    "segment_id": seg_idx,
                    "phrase_id": 1,
                    "speaker_label": speaker_label,
                    "speaker_name": speaker_name,
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "start_sec": start_ms / 1000.0 if isinstance(start_ms, (int, float)) else None,
                    "end_sec": end_ms / 1000.0 if isinstance(end_ms, (int, float)) else None,
                    "text": seg.get("textEdited") or seg.get("text"),
                    "confidence": seg_conf,
                }
            )
            continue

        cur_start = None
        cur_end = None
        cur_tokens = []
        phrase_id = 0

        for (w_start, w_end, w_text) in words:
            if cur_start is None:
                cur_start = w_start
            cur_end = w_end
            cur_tokens.append(str(w_text))

            # 문장 끝 후보인지 체크
            is_eos = any(str(w_text).endswith(suf) for suf in EOS_SUFFIXES)

            if is_eos:
                phrase_id += 1
                text = " ".join(cur_tokens).strip()
                rows.append(
                    {
                        "segment_id": seg_idx,
                        "phrase_id": phrase_id,
                        "speaker_label": speaker_label,
                        "speaker_name": speaker_name,
                        "start_ms": cur_start,
                        "end_ms": cur_end,
                        "start_sec": cur_start / 1000.0,
                        "end_sec": cur_end / 1000.0,
                        "text": text,
                        "confidence": seg_conf,
                    }
                )
                cur_start = None
                cur_end = None
                cur_tokens = []

        # 마지막 덩어리가 EOS로 안 끝났다면 그냥 한 덩어리로 추가
        if cur_tokens:
            phrase_id += 1
            text = " ".join(cur_tokens).strip()
            rows.append(
                {
                    "segment_id": seg_idx,
                    "phrase_id": phrase_id,
                    "speaker_label": speaker_label,
                    "speaker_name": speaker_name,
                    "start_ms": cur_start,
                    "end_ms": cur_end,
                    "start_sec": cur_start / 1000.0,
                    "end_sec": cur_end / 1000.0,
                    "text": text,
                    "confidence": seg_conf,
                }
            )

    df_phrase = pd.DataFrame(rows)
    return df_phrase


# -----------------------------
# 4) 역할 부여 + sandwiched 수정
# -----------------------------
def assign_roles_caster_analyst(
    df: pd.DataFrame,
    fixed_caster_id: str | None = None,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    화자별 총 발화시간(end_ms - start_ms)을 기준으로:
      - 가장 많이 말한 화자 1명: "caster"
      - 나머지 모든 화자: "analyst"

    fixed_caster_id 가 있으면 그 화자를 caster로 고정.
    """
    if df.empty:
        return df.copy(), {}

    df = df.copy()
    df["duration_ms"] = df["end_ms"] - df["start_ms"]

    if "speaker_id" not in df.columns:
        df["speaker_id"] = df["speaker_name"].fillna(df["speaker_label"])

    duration_by_spk = (
        df.groupby("speaker_id")["duration_ms"]
        .sum()
        .sort_values(ascending=False)
    )

    if duration_by_spk.empty:
        return df, {}

    if fixed_caster_id is None:
        caster_id = duration_by_spk.index[0]
    else:
        caster_id = fixed_caster_id
        if caster_id not in duration_by_spk.index:
            caster_id = duration_by_spk.index[0]

    role_map = {
        spk: ("caster" if spk == caster_id else "analyst")
        for spk in duration_by_spk.index
    }

    df["role"] = df["speaker_id"].map(role_map).fillna("analyst")
    return df, role_map


def relabel_short_sandwiched_as_caster(
    df: pd.DataFrame,
    caster_id: str,
    max_duration_ms: int = 1200,  # 1.2초 이하만 후보
    max_gap_ms: int = 400,        # 앞/뒤 gap 0.4초 이하
) -> pd.DataFrame:
    """
    시간 순으로 정렬된 df 에서
      - speaker_id != caster_id 이고
      - 앞/뒤 segment 의 speaker_id 가 모두 caster_id 이고
      - segment 길이가 max_duration_ms 이하이고
      - 앞/뒤와의 gap 도 max_gap_ms 이하이면
    => 해당 segment 를 caster 로 재라벨링
    """
    if df.empty:
        return df.copy()

    df = df.sort_values("start_ms").reset_index(drop=True).copy()

    if "speaker_id" not in df.columns:
        df["speaker_id"] = df["speaker_name"].fillna(df["speaker_label"])

    for i in range(1, len(df) - 1):
        row = df.loc[i]
        if row["speaker_id"] == caster_id:
            continue

        dur = row["end_ms"] - row["start_ms"]
        if dur > max_duration_ms:
            continue

        prev_row = df.loc[i - 1]
        next_row = df.loc[i + 1]

        if prev_row["speaker_id"] != caster_id or next_row["speaker_id"] != caster_id:
            continue

        gap_prev = row["start_ms"] - prev_row["end_ms"]
        gap_next = next_row["start_ms"] - row["end_ms"]

        if (gap_prev is not None and gap_prev > max_gap_ms) or (
            gap_next is not None and gap_next > max_gap_ms
        ):
            continue

        # 조건 만족 → caster로 통합
        df.at[i, "speaker_id"] = caster_id

    return df


# -----------------------------
# 5) TTS CSV / Timeline JSON
# -----------------------------
def save_tts_csv(
    df_phrase_roles: pd.DataFrame,
    audio_file: str | Path,
    out_dir: str | Path,
    encoding: str = "utf-8-sig",
) -> Path:
    """
    TTS/영상 인코딩용 CSV:
      - utterance_id
      - source_video
      - role (caster/analyst)
      - speaker_id / speaker_name
      - start_sec / end_sec
      - text
    """
    audio_path = Path(audio_file)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{audio_path.stem}.tts_phrases.csv"

    df = df_phrase_roles.copy()
    df["source_video"] = audio_path.name

    # utterance_id: clip_세그먼트_프레이즈
    df["utterance_id"] = [
        f"{audio_path.stem}_{int(seg)}_{int(phr)}"
        for seg, phr in zip(df["segment_id"], df["phrase_id"])
    ]

    cols = [
        "utterance_id",
        "source_video",
        "role",
        "speaker_id",
        "speaker_name",
        "start_sec",
        "end_sec",
        "text",
        "confidence",
    ]
    df[cols].to_csv(out_path, index=False, encoding=encoding)
    print(f"[TTS CSV] 저장 완료 → {out_path}")
    return out_path


def build_timeline_json(
    df_phrase_roles: pd.DataFrame,
    video_id: str,
    meta: dict | None = None,
) -> dict:
    """
    sLLM 컨텍스트용 타임라인 구조:
      {
        "video_id": ...,
        "meta": {...},
        "segments": [
          { start_sec, end_sec, role, speaker_id, text, ... },
          ...
        ]
      }
    """
    if meta is None:
        meta = {}

    df_sorted = df_phrase_roles.sort_values("start_sec").reset_index(drop=True)

    segments = []
    for _, row in df_sorted.iterrows():
        segments.append(
            {
                "segment_id": int(row.get("segment_id", 0)),
                "phrase_id": int(row.get("phrase_id", 0)),
                "start_sec": float(row.get("start_sec")) if row.get("start_sec") is not None else None,
                "end_sec": float(row.get("end_sec")) if row.get("end_sec") is not None else None,
                "role": row.get("role"),
                "speaker_id": row.get("speaker_id"),
                "speaker_name": row.get("speaker_name"),
                "text": row.get("text"),
                "confidence": float(row.get("confidence")) if row.get("confidence") is not None else None,
            }
        )

    timeline = {
        "video_id": video_id,
        "meta": meta,
        "segments": segments,
    }
    return timeline


def save_timeline_json(
    df_phrase_roles: pd.DataFrame,
    audio_file: str | Path,
    meta: dict | None,
    out_dir: str | Path,
    indent: int = 2,
) -> Path:
    """
    clip.mp4 -> <out_dir>/clip.clova_timeline.json 저장
    """
    audio_path = Path(audio_file)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    video_id = audio_path.stem
    doc = build_timeline_json(df_phrase_roles, video_id=video_id, meta=meta or {})

    out_path = out_dir / f"{video_id}.clova_timeline.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=indent)

    print(f"[Timeline JSON] 저장 완료 → {out_path}")
    return out_path


# -----------------------------
# 6) 한 번에 STT → CSV/JSON 까지 해주는 함수
# -----------------------------
def run_stt_pipeline(
    audio_path: str | Path,
    *,
    invoke_url: str,
    secret_key: str,
    stt_raw_dir: str | Path,
    stt_seg_dir: str | Path,
    xlsx_keywords_path: str | Path | None = None,
    use_domain_boostings: bool = True,
    speaker_count_min: int = 2,
    speaker_count_max: int = 3,
    save_raw_json: bool = True,
) -> Tuple[Path, Path]:
    """
    MP4/WAV/MP3 파일 → CLOVA STT → 
      1) stt_raw_dir/{stem}.clova_raw.json
      2) stt_seg_dir/{stem}.tts_phrases.csv
      3) stt_seg_dir/{stem}.clova_timeline.json

    을 만든 다음 (tts_csv_path, timeline_json_path)를 반환.
    """

    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {audio_path}")

    stt_raw_dir = Path(stt_raw_dir)
    stt_seg_dir = Path(stt_seg_dir)
    stt_raw_dir.mkdir(parents=True, exist_ok=True)
    stt_seg_dir.mkdir(parents=True, exist_ok=True)

    client = ClovaSpeechClient(invoke_url, secret_key)

    # 엑셀 키워드 사용 여부
    boostings = None
    if xlsx_keywords_path is not None:
        boostings = load_boostings_from_xlsx(xlsx_keywords_path)
        print(f"[STT] 키워드 부스팅 {len(boostings)}개 로드 (엑셀)")
        use_domain_boostings = False
    else:
        print("[STT] 엑셀 부스팅 미사용 → 도메인 부스팅 설정에 따름")

    print(f"[STT] CLOVA 요청: {audio_path.name}")
    result_json = client.recognize_local_file(
        audio_path=audio_path,
        completion="sync",
        diarization_enable=True,
        speaker_count_min=speaker_count_min,
        speaker_count_max=speaker_count_max,
        boostings=boostings,
        use_domain_boostings=use_domain_boostings,
    )

    stem = audio_path.stem

    # 0) 원본 JSON 저장
    if save_raw_json:
        raw_path = stt_raw_dir / f"{stem}.clova_raw.json"
        raw_path.write_text(
            json.dumps(result_json, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"[STT] 원본 JSON 저장 → {raw_path}")

    # 1) segment DF (caster 후보 찾기용)
    df_seg = clova_segments_to_dataframe(result_json)
    if df_seg.empty:
        raise RuntimeError("STT 결과에 segments가 없습니다. (df_seg empty)")

    df_seg["speaker_id"] = df_seg["speaker_name"].fillna(df_seg["speaker_label"])
    dur = (df_seg["end_ms"] - df_seg["start_ms"]).groupby(df_seg["speaker_id"]).sum()
    caster_id_initial = dur.sort_values(ascending=False).index[0]
    print("[STT] 초기 caster 후보 speaker_id:", caster_id_initial)
    print("[STT] 화자별 총 발화시간(ms):")
    print(dur)

    # 2) phrase DF 생성
    df_phrase = clova_words_to_phrases_df(result_json)
    df_phrase["speaker_id"] = df_phrase["speaker_name"].fillna(df_phrase["speaker_label"])

    # 3) sandwiched 단발 analyst → caster로 통합
    df_phrase_clean = relabel_short_sandwiched_as_caster(
        df_phrase,
        caster_id=caster_id_initial,
        max_duration_ms=1200,
        max_gap_ms=400,
    )

    # 4) 최종 역할 부여
    df_phrase_roles, role_map = assign_roles_caster_analyst(
        df_phrase_clean,
        fixed_caster_id=caster_id_initial,
    )
    print("[STT] role_map:", role_map)

    # 5) TTS용 CSV & timeline JSON 저장
    tts_csv_path = save_tts_csv(
        df_phrase_roles,
        audio_file=audio_path,
        out_dir=stt_seg_dir,
    )
    timeline_json_path = save_timeline_json(
        df_phrase_roles,
        audio_file=audio_path,
        meta={},
        out_dir=stt_seg_dir,
    )

    print("[STT] 파이프라인 완료")
    return tts_csv_path, timeline_json_path
