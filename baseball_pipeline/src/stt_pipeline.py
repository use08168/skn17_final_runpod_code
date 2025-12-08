from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

# =========================
# 기본 설정
# =========================

# word 사이 간격이 이 값(ms) 이상이면 "쉼"으로 보고 phrase 분할
DEFAULT_PAUSE_THRESH_MS = 50000  # 0.7초 정도 기준


# =========================
# 데이터 구조
# =========================

@dataclass
class Word:
    start_ms: int
    end_ms: int
    text: str


@dataclass
class SegmentInfo:
    seg_idx: int
    start_ms: int
    end_ms: int
    text: str
    confidence: Optional[float]
    speaker_label: Optional[str]
    speaker_name: Optional[str]
    words: List[Word]


# =========================
# Clova raw JSON 파싱
# =========================

def load_clova_raw(json_path: Path) -> Dict[str, Any]:
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_segments(data: Dict[str, Any]) -> List[SegmentInfo]:
    segments_raw = data.get("segments", [])
    segments: List[SegmentInfo] = []

    for idx, seg in enumerate(segments_raw, start=1):
        start_ms = int(seg.get("start", 0))
        end_ms = int(seg.get("end", 0))
        text = seg.get("text", "") or ""
        confidence = seg.get("confidence", None)

        diar = seg.get("diarization", {}) or {}
        speaker = seg.get("speaker", {}) or {}

        speaker_label = str(diar.get("label")) if diar.get("label") is not None else None
        speaker_name = speaker.get("name", None)

        words_list: List[Word] = []
        for w in seg.get("words", []) or []:
            # w = [start, end, "단어"] 형태
            if not isinstance(w, (list, tuple)) or len(w) < 3:
                continue
            w_start, w_end, w_text = w[0], w[1], w[2]
            try:
                w_start = int(w_start)
                w_end = int(w_end)
            except Exception:
                continue
            words_list.append(Word(start_ms=w_start, end_ms=w_end, text=str(w_text)))

        if not words_list:
            continue

        segments.append(
            SegmentInfo(
                seg_idx=idx,
                start_ms=start_ms,
                end_ms=end_ms,
                text=text,
                confidence=confidence,
                speaker_label=speaker_label,
                speaker_name=speaker_name,
                words=words_list,
            )
        )

    return segments


def compute_speaker_roles(segments: List[SegmentInfo]) -> Dict[str, str]:
    """
    speaker_name(A/B/C…) 별로 전체 발화 시간을 집계해서
    가장 많이 말한 화자를 'caster', 나머지를 'analyst' 로 태깅.
    """
    stats: Dict[str, Dict[str, int]] = {}

    for seg in segments:
        name = seg.speaker_name or seg.speaker_label or "UNK"
        dur = max(seg.end_ms - seg.start_ms, 0)

        if name not in stats:
            stats[name] = {"time": 0, "words": 0}
        stats[name]["time"] += dur
        stats[name]["words"] += len(seg.words)

    if not stats:
        return {}

    # 가장 말이 많은 사람 = caster
    main_speaker = max(stats.items(), key=lambda kv: (kv[1]["time"], kv[1]["words"]))[0]

    roles: Dict[str, str] = {}
    for name in stats.keys():
        roles[name] = "caster" if name == main_speaker else "analyst"

    return roles


# =========================
# word 기반 phrase 분할
# =========================

def split_words_into_phrases(
    words: List[Word],
    pause_thresh_ms: int = DEFAULT_PAUSE_THRESH_MS,
) -> List[List[Word]]:
    """
    연속된 word 리스트를, 단어 사이 무음(pause) 길이에 따라 여러 phrase로 분할.
    """
    if not words:
        return []

    phrases: List[List[Word]] = []
    current: List[Word] = [words[0]]

    for prev, cur in zip(words, words[1:]):
        gap = cur.start_ms - prev.end_ms
        if gap >= pause_thresh_ms:
            phrases.append(current)
            current = [cur]
        else:
            current.append(cur)

    if current:
        phrases.append(current)

    return phrases


def build_phrase_rows(
    segments: List[SegmentInfo],
    source_video: str,
    pause_thresh_ms: int = DEFAULT_PAUSE_THRESH_MS,
) -> pd.DataFrame:
    """
    SegmentInfo 리스트를 word 기반 phrase 단위로 쪼개서
    pandas DataFrame(row = phrase)로 변환.
    """
    speaker_roles = compute_speaker_roles(segments)

    rows: List[Dict[str, Any]] = []
    for seg in segments:
        spk_name = seg.speaker_name or seg.speaker_label or "UNK"
        role = speaker_roles.get(spk_name, "analyst")
        spk_id = seg.speaker_label or ""

        word_chunks = split_words_into_phrases(seg.words, pause_thresh_ms=pause_thresh_ms)
        if not word_chunks:
            continue

        for phrase_idx, chunk in enumerate(word_chunks, start=1):
            start_ms = chunk[0].start_ms
            end_ms = chunk[-1].end_ms
            texts = [w.text for w in chunk]
            joined = " ".join(texts)

            rows.append(
                {
                    "utterance_id": f"clip_{seg.seg_idx}_{phrase_idx}",
                    "source_video": source_video,
                    "role": role,
                    "speaker_id": spk_id,
                    "speaker_name": spk_name,
                    "start_sec": start_ms / 1000.0,
                    "end_sec": end_ms / 1000.0,
                    "text": joined,
                    "confidence": seg.confidence,
                    "orig_text": joined,
                    "llm_text": "",
                }
            )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["start_sec", "end_sec"]).reset_index(drop=True)
    return df


def clova_raw_dict_to_phrase_df(
    data: Dict[str, Any],
    source_video: str,
    pause_thresh_ms: int = DEFAULT_PAUSE_THRESH_MS,
) -> pd.DataFrame:
    segments = parse_segments(data)
    return build_phrase_rows(
        segments=segments,
        source_video=source_video,
        pause_thresh_ms=pause_thresh_ms,
    )


# =========================
# 부스팅 키워드 (엑셀)
# =========================

def load_boostings_from_xlsx(xlsx_path: Path) -> Optional[List[Dict[str, Any]]]:
    """
    stt.xlsx 같은 엑셀에서 키워드 읽어서 boostings 형식으로 변환.
    - 모든 컬럼의 값을 문자열로 모아서 ',' 로 join.
    - use_domain_boostings=True이면 이 값은 무시하는 게 안전.
    """
    if not xlsx_path.exists():
        return None

    df = pd.read_excel(xlsx_path)
    if df.empty:
        return None

    keywords: List[str] = []
    for col in df.columns:
        col_vals = (
            df[col]
            .dropna()
            .astype(str)
            .map(lambda s: s.strip())
        )
        keywords.extend([s for s in col_vals if s])

    keywords = sorted(set(keywords))
    if not keywords:
        return None

    joined = ", ".join(keywords)
    return [{"words": joined}]


# =========================
# ClovaSpeechClient (local upload)
# =========================

class ClovaSpeechClient:
    def __init__(self, invoke_url: str, secret_key: str):
        self.invoke_url = invoke_url.rstrip("/")
        self.secret = secret_key

    def recognize_local_file(
        self,
        audio_path: str,
        completion: str = "sync",
        diarization_enable: bool = True,
        speaker_count_min: int = -1,
        speaker_count_max: int = -1,
        boostings: Optional[List[Dict[str, Any]]] = None,
        use_domain_boostings: bool = True,
    ) -> Dict[str, Any]:
        """
        공식 문서의 req_upload 래퍼.
        """
        request_body: Dict[str, Any] = {
            "language": "ko-KR",
            "completion": completion,
            "callback": "",
            "userdata": None,
            "wordAlignment": True,
            "fullText": True,
            "forbiddens": None,
            "boostings": boostings,
            "diarization": {
                "enable": diarization_enable,
                "speakerCountMin": speaker_count_min,
                "speakerCountMax": speaker_count_max,
            },
            # 도메인 부스팅 사용 여부
            "useDomainBoostings": use_domain_boostings,
        }

        headers = {
            "Accept": "application/json;UTF-8",
            "X-CLOVASPEECH-API-KEY": self.secret,
        }

        upload_url = self.invoke_url + "/recognizer/upload"
        audio_path = str(audio_path)

        with open(audio_path, "rb") as f:
            files = {
                "media": f,
                "params": (
                    None,
                    json.dumps(request_body, ensure_ascii=False).encode("UTF-8"),
                    "application/json",
                ),
            }
            resp = requests.post(
                headers=headers,
                url=upload_url,
                files=files,
                timeout=120,
            )

        resp.raise_for_status()
        return resp.json()


# =========================
# High-level: run_stt_pipeline
# =========================

def run_stt_pipeline(
    audio_path: Path | str,
    invoke_url: str,
    secret_key: str,
    stt_raw_dir: Path,
    stt_seg_dir: Path,
    xlsx_keywords_path: Optional[Path] = None,
    use_domain_boostings: bool = True,
    speaker_count_min: int = 2,
    speaker_count_max: int = 3,
    save_raw_json: bool = True,
    pause_thresh_ms: int = DEFAULT_PAUSE_THRESH_MS,
) -> Path:
    """
    baseball_pipeline.ipynb 셀 1에서 쓰는 one-shot 파이프라인.

    1) Clova Speech에 audio_path(mp4/wav 등)를 업로드해서 STT 실행
    2) raw json을 stt_raw_dir/{stem}.clova_raw.json 로 저장 (옵션)
    3) word 기반 phrase를 이용해 타임라인용 JSON(stt_seg_dir/{stem}.timeline.json) 생성

    return: timeline_json_path (Path)
    """
    audio_path = Path(audio_path)
    stt_raw_dir.mkdir(parents=True, exist_ok=True)
    stt_seg_dir.mkdir(parents=True, exist_ok=True)

    stem = audio_path.stem
    raw_json_path = stt_raw_dir / f"{stem}.clova_raw.json"
    timeline_json_path = stt_seg_dir / f"{stem}.timeline.json"

    # --- 부스팅 키워드 로드 (옵션) ---
    boostings: Optional[List[Dict[str, Any]]] = None
    if xlsx_keywords_path is not None and xlsx_keywords_path.exists():
        if not use_domain_boostings:
            boostings = load_boostings_from_xlsx(xlsx_keywords_path)
        else:
            print("[STT_PIPELINE] use_domain_boostings=True 이므로 엑셀 boostings는 무시됨")

    # --- Clova 호출 ---
    client = ClovaSpeechClient(invoke_url, secret_key)
    print(f"[STT_PIPELINE] Clova STT 요청 시작: {audio_path}")

    result_json = client.recognize_local_file(
        audio_path=str(audio_path),
        completion="sync",
        diarization_enable=True,
        speaker_count_min=speaker_count_min,
        speaker_count_max=speaker_count_max,
        boostings=boostings,
        use_domain_boostings=use_domain_boostings,
    )

    # --- raw JSON 저장 (옵션) ---
    if save_raw_json:
        raw_json_path.parent.mkdir(parents=True, exist_ok=True)
        with raw_json_path.open("w", encoding="utf-8") as f:
            json.dump(result_json, f, ensure_ascii=False, indent=2)
        print(f"[STT_PIPELINE] raw JSON 저장 -> {raw_json_path}")

    # --- phrase DF 생성 (word 기반 분할) ---
    df = clova_raw_dict_to_phrase_df(
        data=result_json,
        source_video=audio_path.name,
        pause_thresh_ms=pause_thresh_ms,
    )

    # --- 타임라인 JSON 생성 (test 파일과 동일한 스키마) ---
    segments_for_timeline: List[Dict[str, Any]] = []
    for seg_idx, row in enumerate(df.itertuples(index=False), start=0):
        # df 컬럼: utterance_id, source_video, role, speaker_id, speaker_name,
        #          start_sec, end_sec, text, confidence, ...
        speaker_label = str(row.speaker_id) if row.speaker_id else None
        confidence_val: Optional[float]
        if row.confidence is None or pd.isna(row.confidence):
            confidence_val = None
        else:
            confidence_val = float(row.confidence)

        segments_for_timeline.append(
            {
                "segment_id": seg_idx,
                "start_sec": float(row.start_sec),
                "end_sec": float(row.end_sec),
                "role": row.role,
                "speaker_label": speaker_label,
                "text": row.text,
                "confidence": confidence_val,
            }
        )

    timeline_obj: Dict[str, Any] = {
        "video_id": stem,   # 예: "vocals"
        "meta": {},
        "segments": segments_for_timeline,
    }

    timeline_json_path.parent.mkdir(parents=True, exist_ok=True)
    with timeline_json_path.open("w", encoding="utf-8") as f:
        json.dump(timeline_obj, f, ensure_ascii=False, indent=2)

    print(f"[STT_PIPELINE] timeline JSON 저장 -> {timeline_json_path}")
    return timeline_json_path