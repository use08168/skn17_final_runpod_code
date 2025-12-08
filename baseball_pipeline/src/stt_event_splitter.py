from __future__ import annotations

from typing import List, Dict, Any, Tuple, Optional


# ============================================
# 1. 유틸: 문장부호 기준으로 "첫 문장" 찾기 (여러 segment를 가로질러)
# ============================================

def find_first_sentence_across_segments(
    segments: List[Dict[str, Any]],
) -> Tuple[str, int, str]:
    """
    segments(시간순 정렬된 이벤트 세그먼트들)를 앞에서부터 보면서,
    처음 등장하는 문장부호(., ?, !, …, ？, ！)까지를 하나의 문장으로 묶는다.

    Returns
    -------
    caster_text : str
        첫 문장 전체 텍스트 (여러 segment를 합친 것일 수 있음)
    last_idx : int
        이 문장이 끝나는 segment의 인덱스 (segments 기준)
    tail_text : str
        문장부호 이후, 같은 segment 안에 남은 뒷부분 텍스트
        (analyst_text의 맨 앞에 들어가야 할 부분)
    """
    terminators = ".?!…？！"

    sentence_parts: List[str] = []
    tail_text = ""
    last_idx = -1

    for i, seg in enumerate(segments):
        text = (seg.get("text") or "").strip()
        if not text:
            continue

        # 이 segment 안에서 처음 등장하는 문장부호 위치 찾기
        pos = None
        for idx, ch in enumerate(text):
            if ch in terminators:
                pos = idx
                break

        if pos is None:
            # 문장부호 없음 → 이 segment 전체가 아직 첫 문장에 포함
            sentence_parts.append(text)
            last_idx = i
        else:
            # 문장부호 발견 → 여기서 첫 문장 끝
            main = text[: pos + 1].strip()
            tail = text[pos + 1 :].strip()

            if main:
                sentence_parts.append(main)
                last_idx = i
            if tail:
                tail_text = tail  # 이건 나중에 analyst_text 맨 앞에 들어감
            break

    # 문장부호를 전혀 못 찾았을 수도 있음
    if not sentence_parts:
        return "", -1, ""

    caster_text = " ".join(sentence_parts).strip()
    return caster_text, last_idx, tail_text


# ============================================
# 2. 1차: 이벤트(세트) 단위로 묶기 (이전과 동일)
# ============================================

def group_segments_by_event(
    segments: List[Dict[str, Any]],
    caster_gap: float = 2.0,
    silence_gap: float = 4.0,
) -> List[List[Dict[str, Any]]]:
    """
    STT segment들을 '이벤트(세트)' 단위로 1차 그룹핑한다.
    - 새 이벤트 시작은 caster에서만 일어난다.
    - analyst는 항상 가장 가까운 caster가 있는 이벤트에 붙인다.
    - confidence 필터는 사용하지 않는다.
    """
    # 1) caster / analyst 만 사용
    filtered: List[Dict[str, Any]] = []
    for seg in segments:
        role = (seg.get("role") or "").lower()
        if role not in ("caster", "analyst"):
            continue
        seg = {
            **seg,
            "role": role,
            "start_sec": float(seg.get("start_sec", 0.0)),
            "end_sec": float(seg.get("end_sec", 0.0)),
            "text": seg.get("text", "") or "",
        }
        filtered.append(seg)

    if not filtered:
        return []

    # 2) 시간 순 정렬
    filtered.sort(key=lambda s: s["start_sec"])

    # 3) 첫 caster 이전의 analyst 발언은 버린다.
    first_caster_idx = None
    for i, seg in enumerate(filtered):
        if seg["role"] == "caster":
            first_caster_idx = i
            break

    if first_caster_idx is None:
        # caster 자체가 없으면 이벤트를 만들 수 없음
        return []

    segs = filtered[first_caster_idx:]

    events: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []

    def last_caster_in_current(cur: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        for s in reversed(cur):
            if s["role"] == "caster":
                return s
        return None

    for seg in segs:
        if not current:
            # 첫 이벤트는 반드시 caster일 때 시작
            if seg["role"] != "caster":
                continue
            current = [seg]
            continue

        prev = current[-1]
        gap = seg["start_sec"] - prev["end_sec"]
        start_new = False

        if seg["role"] == "caster":
            # 1) 긴 침묵 후 caster 등장 → 새 이벤트
            if gap >= silence_gap:
                start_new = True
            else:
                # 2) 같은 이벤트 안 '마지막 caster'와의 거리
                last_c = last_caster_in_current(current)
                if last_c is not None:
                    caster_gap_val = seg["start_sec"] - last_c["start_sec"]
                    if caster_gap_val >= caster_gap:
                        start_new = True

        # analyst로는 새 이벤트를 시작하지 않는다.
        if start_new:
            events.append(current)
            current = [seg]
        else:
            current.append(seg)

    if current:
        events.append(current)

    return events


# ============================================
# 3. 2차: 각 이벤트를 "caster 1 + (optional) analyst 1"로 정규화
#    - caster_text: 이벤트 전체에서 첫 문장(여러 segment 포함 가능)
#    - analyst_text: 그 이후 나온 모든 말(순서 유지)
# ============================================

def normalize_event_to_single_caster_analyst(
    event_segments: List[Dict[str, Any]],
    video_id: str,
    index: int,
) -> Optional[Dict[str, Any]]:
    """
    하나의 이벤트(segment 리스트)를
    'caster 1개 + (있으면) analyst 1개' 구조의 세트(dict)로 변환한다.

    - caster_text : 이벤트 전체에서 "첫 문장" (여러 segment를 이어 붙일 수 있음).
    - analyst_text : 첫 문장 이후의 나머지 모든 발화를 시간 순서대로 이어 붙인 것.
      (첫 문장이 끝난 segment의 tail + 그 뒤 segment 전체)
    - 여기서는 role(caster/analyst)을 무시하고, 첫 문장/나머지로만 자른다.
    """
    if not event_segments:
        return None

    # 시간 순 정렬 (안전용)
    event_segments = sorted(event_segments, key=lambda s: s["start_sec"])

    # 이벤트 전체 시간 범위
    set_start_sec = min(s["start_sec"] for s in event_segments)
    set_end_sec = max(s["end_sec"] for s in event_segments)

    # ---------- caster_text: 첫 문장 찾기 ----------
    caster_text, last_idx, tail_text = find_first_sentence_across_segments(
        event_segments
    )

    if not caster_text:
        # 문장부호가 하나도 없고, 텍스트도 없으면 스킵
        return None

    # ---------- analyst_text: 첫 문장 이후 모든 발화 ----------
    analyst_pieces: List[str] = []
    analyst_time_candidates: List[Tuple[float, float]] = []

    # 같은 segment 안 tail부터
    if tail_text:
        analyst_pieces.append(tail_text)
        # tail의 시간은 대략 문장 끝난 시점으로 본다
        analyst_time_candidates.append(
            (event_segments[last_idx]["end_sec"], event_segments[last_idx]["end_sec"])
        )

    # 그 뒤 segment 전체를 순서대로 추가
    if last_idx >= 0:
        start_j = last_idx + 1
    else:
        start_j = 0

    for j in range(start_j, len(event_segments)):
        seg = event_segments[j]
        txt = (seg.get("text") or "").strip()
        if not txt:
            continue
        analyst_pieces.append(txt)
        analyst_time_candidates.append((seg["start_sec"], seg["end_sec"]))

    # analyst_text는 "첫 캐스터의 첫 문장 이후에 나온 모든 말"을
    # role과 상관 없이 시간 순서대로 이어붙인 것.
    if analyst_pieces:
        analyst_text: Optional[str] = " ".join(analyst_pieces).strip()
    else:
        # 이 이벤트는 caster만 존재 (analyst 없음)
        analyst_text = None

    set_id = f"{video_id}-{index}"

    return {
        "set_id": set_id,
        "video_id": video_id,
        "set_start_sec": set_start_sec,
        "set_end_sec": set_end_sec,
        "caster_text": caster_text,
        "analyst_text": analyst_text,
    }


# ============================================
# 4. STT JSON → 이벤트 세트 리스트 변환 함수
# ============================================

def stt_json_to_event_sets(
    stt_json: Dict[str, Any],
    caster_gap: float = 2.0,
    silence_gap: float = 4.0,
) -> List[Dict[str, Any]]:
    """
    clova STT JSON을 '이벤트 세트 리스트'로 변환한다.
    """
    video_id = stt_json.get("video_id", "clip")
    segments = stt_json.get("segments", [])

    # 1) 1차 이벤트 그룹핑
    raw_events = group_segments_by_event(
        segments,
        caster_gap=caster_gap,
        silence_gap=silence_gap,
    )

    # 2) 각 이벤트를 "caster 1 + (optional) analyst 1" 세트로 정규화
    event_sets: List[Dict[str, Any]] = []
    for idx, ev in enumerate(raw_events, start=1):
        norm = normalize_event_to_single_caster_analyst(
            ev,
            video_id=video_id,
            index=idx,
        )
        if norm is not None:
            event_sets.append(norm)

    return event_sets