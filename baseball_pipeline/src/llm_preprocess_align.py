# src/llm_preprocess_align.py

from __future__ import annotations

from pathlib import Path
from typing import Optional, Iterable, List

import numpy as np
import pandas as pd

# 역할 세트 (필요하면 프로젝트에 맞게 수정)
DEFAULT_CASTER_ROLES: set[str] = {"caster", "A"}
DEFAULT_ANALYST_ROLES: set[str] = {"analyst", "B", "C"}


def _choose_text_for_len(row) -> str:
    """
    길이 판단용 텍스트 선택:
    - 캐스터: orig_text
    - 해설:   llm_text
    """
    role_raw = str(getattr(row, "role", "")).strip().lower()
    if role_raw == "caster":
        raw = getattr(row, "orig_text", "")
    else:
        raw = getattr(row, "llm_text", "")

    if pd.isna(raw):
        return ""
    s = str(raw).strip()
    if not s or s.lower() == "nan":
        return ""
    return s


def _estimate_slot_duration_from_text(
    row,
    *,
    role_col: str,
    caster_roles: set[str],
    analyst_roles: set[str],
    caster_chars_per_sec: float,
    analyst_chars_per_sec: float,
    dur_margin_ratio: float,
) -> float:
    """
    텍스트 길이(발화량) 기반으로 "이 정도는 줘야 말할 수 있다" 하는
    슬롯 길이를 추정한다.

    - 공백 제거한 글자 수 / (역할별 글자/초) * (1 + margin) 구조
    """
    txt = _choose_text_for_len(row)
    # 공백 제거 후 글자 수
    n_chars = len("".join(txt.split()))
    if n_chars <= 0:
        return 0.0

    role_raw = str(getattr(row, role_col, "")).strip().lower()

    if role_raw in caster_roles:
        cps = caster_chars_per_sec
    elif role_raw in analyst_roles:
        cps = analyst_chars_per_sec
    else:
        cps = analyst_chars_per_sec  # 기본 해설 속도로

    if cps <= 0:
        return 0.0

    base = n_chars / cps  # 초 단위
    return base * (1.0 + dur_margin_ratio)


def _apply_analyst_priority_pre(
    df: pd.DataFrame,
    *,
    start_col: str,
    end_col: str,
    role_col: str,
    caster_roles: set[str],
    analyst_roles: set[str],
    min_overlap_sec: float,
) -> pd.DataFrame:
    """
    (정렬/슬롯 재계산 전) 원래 start/end 기준으로
    해설 구간과 겹치는 캐스터 구간을 제거한다.

    - min_overlap_sec <= 0: "조금이라도" 겹치면 캐스터 드롭
    - min_overlap_sec > 0:  겹치는 길이가 이 값 이상일 때만 캐스터 드롭
    """
    if min_overlap_sec is None:
        return df

    if min_overlap_sec < 0:
        min_overlap_sec = 0.0

    roles = df[role_col].astype(str).str.strip().str.lower().to_numpy()
    starts = df[start_col].to_numpy(float)
    ends = df[end_col].to_numpy(float)

    caster_roles = {r.lower() for r in caster_roles}
    analyst_roles = {r.lower() for r in analyst_roles}

    is_caster = np.isin(roles, list(caster_roles))
    is_analyst = np.isin(roles, list(analyst_roles))

    drop_mask = np.zeros(len(df), dtype=bool)

    for i in range(len(df)):
        if not is_analyst[i]:
            continue

        a_start = starts[i]
        a_end = ends[i]

        # 모든 구간과의 겹치는 길이
        overlap_len = np.minimum(ends, a_end) - np.maximum(starts, a_start)

        if min_overlap_sec <= 0:
            conflict = overlap_len > 0
        else:
            conflict = overlap_len >= min_overlap_sec

        conflict &= is_caster
        conflict[i] = False  # 자기 자신은 제외

        if np.any(conflict):
            drop_mask |= conflict

    if drop_mask.any():
        before = len(df)
        df = df.loc[~drop_mask].copy()
        df = df.sort_values(start_col).reset_index(drop=True)
        print(f"[LLM_PRE_ALIGN] analyst priority drop casters: {before} -> {len(df)}")
    else:
        print("[LLM_PRE_ALIGN] analyst priority: no caster rows dropped")

    return df


def preprocess_and_align_llm_csv(
    llm_csv_path: Path | str,
    out_csv_path: Optional[Path | str] = None,
    *,
    # 시간/컬럼 이름
    start_col: str = "start_sec",
    end_col: str = "end_sec",
    role_col: str = "role",
    uttid_col: str = "utterance_id",
    caster_roles: Iterable[str] = DEFAULT_CASTER_ROLES,
    analyst_roles: Iterable[str] = DEFAULT_ANALYST_ROLES,
    # ===== 텍스트/구간 전처리 파라미터 =====
    min_text_chars: int = 2,
    # 같은 화자가 아주 짧게 쪼개진 구간을 merge 할 기준
    merge_same_role: bool = True,
    merge_gap_thresh_sec: float = 0.25,   # 앞/뒤 구간 사이 간격이 이 이하면 합치기
    merge_short_thresh_sec: float = 1.0,  # 둘 중 하나라도 이보다 짧으면 합치기 후보
    # ===== 슬롯 길이 튜닝 파라미터 =====
    min_gap_sec: float = 0.02,           # 발화들 사이 최소 간격
    caster_extra_ratio: float = 0.0,     # 캐스터 slot 늘리는 비율 (0.2 → 1.2배)
    analyst_extra_ratio: float = 0.5,    # 해설 slot 늘리는 비율 (2.0 → 3배)
    max_analyst_expand_sec: float = 7.0, # 해설 1줄당 최대 +7초까지만 확장
    # ===== 텍스트 길이 기반 duration 추정 파라미터 =====
    caster_chars_per_sec: float = 9.0,   # 캐스터 평균 9글자/초 정도
    analyst_chars_per_sec: float = 7.0,  # 해설 평균 7글자/초 정도
    dur_margin_ratio: float = 0.2,       # 예측 시간에 20% 여유
    # 전역 슬롯 스케일
    global_slot_scale: float = 1.0,
    # ===== 해설 우선 전략 (전처리 단계에서 캐스터 drop) =====
    analyst_priority_min_overlap_sec: Optional[float] = None,
) -> Path:
    """
    1) LLM CSV를 전처리 (이상하게 짧게 쪼개진 구간 merge 등)
    2) (선택) 해설 구간과 겹치는 캐스터 구간 drop
    3) start_sec / end_sec를 역할별 슬롯 정책에 맞게 재계산
       - 원래 STT duration
       - 텍스트 길이 기반 예측 duration
       - extra_ratio 기반 duration
       이 셋을 섞어서, "발화량에 맞는 슬롯"을 만든다.

    출력 CSV:
      - 원본 주요 컬럼 유지 (utterance_id, role, orig_text, llm_text, start_sec, end_sec, ...)
      - orig_start_sec, orig_end_sec 는 내부 계산용으로만 사용하고,
        최종 CSV에는 포함되지 않는다.
    """
    llm_csv_path = Path(llm_csv_path)

    df = pd.read_csv(llm_csv_path)
    df.columns = [c.replace("\ufeff", "").strip() for c in df.columns]

    required = {uttid_col, role_col, start_col, end_col}
    if not required.issubset(df.columns):
        raise ValueError(
            f"[LLM_PRE_ALIGN] CSV에 {required} 컬럼이 필요합니다. "
            f"현재 컬럼: {df.columns.tolist()}"
        )

    caster_roles = {r.lower() for r in caster_roles}
    analyst_roles = {r.lower() for r in analyst_roles}

    # 시간 정리
    df = df.dropna(subset=[start_col, end_col]).copy()
    df[start_col] = df[start_col].astype(float)
    df[end_col] = df[end_col].astype(float)
    df = df.sort_values(start_col).reset_index(drop=True)

    if df.empty:
        raise ValueError("[LLM_PRE_ALIGN] CSV 에 유효한 구간이 없습니다.")

    # 원본 start/end 백업 (내부 계산용)
    df["orig_start_sec"] = df[start_col]
    df["orig_end_sec"] = df[end_col]

    # ====== 1단계: 텍스트 기반 전처리 (빈 줄/매우 짧은 줄 제거) ======
    keep_mask: List[bool] = []
    for row in df.itertuples(index=False):
        txt = _choose_text_for_len(row)
        keep = len(txt) >= min_text_chars
        keep_mask.append(keep)

    df = df[keep_mask].reset_index(drop=True)
    print(f"[LLM_PRE_ALIGN] very short/empty rows removed: {len(keep_mask) - len(df)}")

    if df.empty:
        raise ValueError("[LLM_PRE_ALIGN] 전처리 후 남은 구간이 없습니다.")

    # ====== 2단계: 같은 화자의 짧은 구간 merge ======
    if merge_same_role:
        merged_rows = []
        rows = list(df.to_dict(orient="records"))

        for r in rows:
            if not merged_rows:
                merged_rows.append(r)
                continue

            prev = merged_rows[-1]
            role_prev = str(prev[role_col]).strip().lower()
            role_cur = str(r[role_col]).strip().lower()

            same_role = (role_prev == role_cur) and role_prev != ""

            gap = float(r[start_col]) - float(prev[end_col])
            prev_dur = float(prev[end_col]) - float(prev[start_col])
            cur_dur = float(r[end_col]) - float(r[start_col])

            should_merge = (
                same_role
                and gap >= 0.0
                and gap <= merge_gap_thresh_sec
                and (prev_dur <= merge_short_thresh_sec or cur_dur <= merge_short_thresh_sec)
            )

            if should_merge:
                # 시간 합치기
                prev[end_col] = max(float(prev[end_col]), float(r[end_col]))
                prev["orig_end_sec"] = max(
                    float(prev.get("orig_end_sec", prev[end_col])),
                    float(r.get("orig_end_sec", r[end_col])),
                )

                # 텍스트 합치기 (orig_text / llm_text / text 모두 시도)
                for col in ["orig_text", "llm_text", "text"]:
                    if col in r:
                        prev_val = str(prev.get(col, "") or "").strip()
                        cur_val = str(r.get(col, "") or "").strip()
                        if prev_val and cur_val:
                            prev[col] = (prev_val + " " + cur_val).strip()
                        elif cur_val:
                            prev[col] = cur_val
            else:
                merged_rows.append(r)

        df = pd.DataFrame(merged_rows)
        df = df.sort_values(start_col).reset_index(drop=True)
        print(f"[LLM_PRE_ALIGN] merged rows count: {len(keep_mask)} -> {len(df)}")

    # ====== 2.5단계: 해설 우선 전략 (겹치는 캐스터 drop) ======
    if analyst_priority_min_overlap_sec is not None:
        df = _apply_analyst_priority_pre(
            df,
            start_col=start_col,
            end_col=end_col,
            role_col=role_col,
            caster_roles=caster_roles,
            analyst_roles=analyst_roles,
            min_overlap_sec=analyst_priority_min_overlap_sec,
        )

    # ====== 3단계: 역할/텍스트 기반으로 슬롯 길이 재계산 (TTS 이전 align) ======
    new_starts: list[float] = []
    new_ends: list[float] = []

    prev_end = float(df[start_col].min()) - min_gap_sec  # 이전 발화 끝 시각

    for i, row in enumerate(df.itertuples(index=False)):
        role_raw = str(getattr(row, role_col, "")).strip()
        role = role_raw.lower()
    
        orig_start = float(getattr(row, start_col))
        orig_end = float(getattr(row, end_col))
        orig_dur = max(orig_end - orig_start, 0.01)
    
        # 3-1) 텍스트 길이 기반 duration 추정
        pred_dur = _estimate_slot_duration_from_text(
            row,
            role_col=role_col,
            caster_roles=caster_roles,
            analyst_roles=analyst_roles,
            caster_chars_per_sec=caster_chars_per_sec,
            analyst_chars_per_sec=analyst_chars_per_sec,
            dur_margin_ratio=dur_margin_ratio,
        )
    
        # 3-2) extra_ratio 기반 duration
        if role in caster_roles:
            extra_ratio = caster_extra_ratio
        elif role in analyst_roles:
            extra_ratio = analyst_extra_ratio
        else:
            extra_ratio = 0.0
    
        dur_from_ratio = orig_dur * (1.0 + extra_ratio)
    
        # 3-3) 최종적으로 쓰고 싶은 "이 줄은 최소 이 정도는 줘야 한다" 길이
        desired_dur = max(orig_dur, pred_dur, dur_from_ratio)
    
        # 🔥 전역 스케일 적용 (전체적으로 슬롯 넓히기)
        desired_dur *= global_slot_scale
    
        # 해설은 너무 과도하게 안 나가게 상한
        if role in analyst_roles and max_analyst_expand_sec is not None:
            desired_dur = min(desired_dur, orig_dur + max_analyst_expand_sec)
    
        # 🔥 start는 "절대" 건드리지 않고, 원래 STT start를 그대로 사용
        start_aligned = orig_start
    
        # desired_dur만큼 오른쪽으로 end를 늘린다
        end_aligned = start_aligned + desired_dur
    
        # 만약 이상하게 너무 짧아지면 최소 길이 보장
        if end_aligned <= start_aligned + 0.02:
            end_aligned = start_aligned + max(0.02, orig_dur * 0.3)
    
        new_starts.append(start_aligned)
        new_ends.append(end_aligned)
    
        # prev_end는 이제 "참고용"으로만 업데이트 (다음 줄 start에는 사용 안 함)
        prev_end = end_aligned
    
        print(
            f"[LLM_PRE_ALIGN] role={role_raw:8s} "
            f"orig=({orig_start:.3f}~{orig_end:.3f}, dur={orig_dur:.3f}) "
            f"pred_dur={pred_dur:.3f} "
            f"-> aligned=({start_aligned:.3f}~{end_aligned:.3f}, "
            f"dur={end_aligned-start_aligned:.3f})"
        )

    df[start_col] = new_starts
    df[end_col] = new_ends

    # ====== 4단계: CSV 저장 (orig_* 컬럼은 제거) ======
    if out_csv_path is None:
        # 예: clip.tts_phrases.llm_kanana.csv -> clip.tts_phrases.llm_kanana.pre_aligned.csv
        stem = llm_csv_path.stem
        out_csv_path = llm_csv_path.with_name(stem + ".pre_aligned.csv")

    out_csv_path = Path(out_csv_path)
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)

    # 최종 출력에서는 orig_start_sec, orig_end_sec 를 제거
    df_out = df.copy()
    df_out = df_out.drop(columns=["orig_start_sec", "orig_end_sec"], errors="ignore")

    df_out.to_csv(out_csv_path, index=False, encoding="utf-8-sig")
    print("[LLM_PRE_ALIGN] saved preprocessed+aligned CSV:", out_csv_path)

    return out_csv_path
