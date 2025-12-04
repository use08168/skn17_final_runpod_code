# src/tts_align_wsola_gap.py

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import soundfile as sf
from pytsmod import wsolatsm  # WSOLA TSM


# 화자 역할 이름 세트 (너 CSV에 맞게 필요하면 수정)
DEFAULT_CASTER_ROLES: set[str] = {"caster", "A"}
DEFAULT_ANALYST_ROLES: set[str] = {"analyst", "B", "C"}


def _wsola_tsm(audio: np.ndarray, rate: float) -> np.ndarray:
    """
    pytsmod.wsolatsm.wsola 래퍼

    - audio: (num_samples, channels) 또는 (num_samples,)
    - 항상 (num_samples, channels) 형태로 반환하도록 정규화
    """
    audio = np.asarray(audio)

    if np.isclose(rate, 1.0, atol=1e-3):
        # 그대로 반환하되, 모노면 (N, 1)로 맞춰줌
        if audio.ndim == 1:
            return audio[:, None]
        return audio

    if audio.ndim == 1:
        # (N,) 모노 입력
        y = wsolatsm.wsola(audio, rate)  # -> (N,) 예상
        y = np.asarray(y)
        if y.ndim == 1:
            y = y[:, None]  # (N, 1)로
        return y

    # 2D인 경우: (num_samples, channels) -> (channels, num_samples)
    x = audio.T
    y = wsolatsm.wsola(x, rate)      # (channels, new_len) 또는 (new_len,)
    y = np.asarray(y)

    if y.ndim == 1:
        # 채널이 1개인데 1D로 온 경우
        y = y[None, :]               # (1, new_len)
    # 다시 (num_samples, channels)
    y = y.T
    return y


def _resolve_wav_path(tts_dir: Path, p: str | Path) -> Path:
    p = Path(p)
    if p.is_absolute():
        return p
    return tts_dir / p


def build_wsola_tts_timeline_gap(
    llm_csv_path: Path,
    tts_audio_dir: Path,
    out_wav_path: Path,
    *,
    start_col: str = "start_sec",
    end_col: str = "end_sec",
    role_col: str = "role",
    uttid_col: str = "utterance_id",
    # gap/여유 설정
    min_gap_ms: int = 60,          # 이전 발화 끝과 다음 발화 시작 최소 간격
    tail_margin_ms: int = 80,      # 슬롯 끝에 남길 최소 여유
    # 역할별 최대 속도 증가 배수 (1.5 = 최대 1.5배 빨라지기)
    caster_max_speedup: float = 1.3,
    analyst_max_speedup: float = 1.8,
    caster_roles: Iterable[str] = DEFAULT_CASTER_ROLES,
    analyst_roles: Iterable[str] = DEFAULT_ANALYST_ROLES,
) -> Path:
    """
    LLM CSV + TTS 개별 wav 를 사용해서,
    '슬롯에 충분히 들어가면 속도 그대로, 안 들어가면 가능한 한 부드럽게 빠르게' WSOLA를 적용한
    전체 타임라인 wav 를 만든다.

    - start_sec / end_sec: 슬롯 시간
    - 발화 길이(orig_dur) <= 슬롯내 사용 가능 시간(target_dur) 이면 → rate=1.0 (원래 속도)
    - 발화 길이 > target_dur 이면 →
        ideal_speedup = orig_dur / target_dur
        - ideal_speedup 이 '귀에 거북하지 않은 범위(soft_speedup)' 이하면
          → slot에 딱 맞도록 속도 조절
        - 그 이상이면
          → soft_speedup 까지만 빠르게 하고, 나머지는 tail 에서 살짝 잘릴 수 있음
    """
    llm_csv_path = Path(llm_csv_path)
    tts_audio_dir = Path(tts_audio_dir)
    out_wav_path = Path(out_wav_path)

    caster_roles = {r.lower() for r in caster_roles}
    analyst_roles = {r.lower() for r in analyst_roles}

    df = pd.read_csv(llm_csv_path)
    df.columns = [c.replace("\ufeff", "").strip() for c in df.columns]

    required = {uttid_col, role_col, start_col, end_col}
    if not required.issubset(df.columns):
        raise ValueError(
            f"[WSOLA_GAP] CSV에 {required} 컬럼이 필요합니다. "
            f"현재 컬럼: {df.columns.tolist()}"
        )

    # 시간 정렬 + 정리
    df = df.dropna(subset=[start_col, end_col]).copy()
    df[start_col] = df[start_col].astype(float)
    df[end_col] = df[end_col].astype(float)
    df = df.sort_values(start_col).reset_index(drop=True)

    if df.empty:
        raise ValueError("[WSOLA_GAP] CSV 에 유효한 구간이 없습니다.")

    last_end = float(df[end_col].max())
    total_dur = last_end + (tail_margin_ms + min_gap_ms) / 1000.0
    print(f"[WSOLA_GAP] last_end={last_end:.3f}s, total≈{total_dur:.3f}s")

    # 첫 wav 에서 sr, 채널 결정
    first_utt = str(df.iloc[0][uttid_col])
    first_path = tts_audio_dir / f"{first_utt}.wav"
    if not first_path.exists():
        raise FileNotFoundError(f"[WSOLA_GAP] 첫 TTS 파일 없음: {first_path}")

    info = sf.info(first_path)
    sr = info.samplerate
    ch = info.channels
    print(f"[WSOLA_GAP] samplerate={sr}, channels={ch}")

    num_samples = int(np.ceil(total_dur * sr)) + sr
    timeline = np.zeros((num_samples, ch), dtype=np.float32)

    cursor_sec = 0.0  # 이전 발화가 실제로 끝난 시각 (sec)

    for row in df.itertuples(index=False):
        utt_id = str(getattr(row, uttid_col))
        role_raw = str(getattr(row, role_col, "")).strip()
        role = role_raw.lower()

        slot_start = float(getattr(row, start_col))
        slot_end = float(getattr(row, end_col))
        if slot_end <= slot_start:
            continue

        # 이전 발화 끝 + 최소 gap 과, 슬롯 시작 중 더 늦은 쪽을 실제 시작으로
        min_start = cursor_sec + (min_gap_ms / 1000.0)
        logical_start = max(slot_start, min_start)

        # 슬롯 끝에서 tail_margin 만큼 빼고 실제 쓸 수 있는 오른쪽 경계
        logical_end = slot_end - (tail_margin_ms / 1000.0)

        if logical_end <= logical_start + 0.01:
            print(
                f"[WSOLA_GAP][SKIP] utt={utt_id} slot 너무 좁음 "
                f"slot=({slot_start:.2f}~{slot_end:.2f}), "
                f"logical=({logical_start:.2f}~{logical_end:.2f})"
            )
            continue

        target_dur = logical_end - logical_start  # 이 안에 넣고 싶은 길이 (sec)

        wav_path = tts_audio_dir / f"{utt_id}.wav"
        if not wav_path.exists():
            print(f"[WSOLA_GAP][WARN] utt={utt_id} wav 없음:", wav_path)
            continue

        # ---- 오디오 로드 (항상 2D로 정규화) ----
        audio, this_sr = sf.read(wav_path, always_2d=False)
        audio = np.asarray(audio)
        if audio.ndim == 1:
            audio = audio[:, None]  # (N,) -> (N, 1)

        if this_sr != sr:
            raise RuntimeError(
                f"[WSOLA_GAP] sr 불일치: {wav_path} (expected {sr}, got {this_sr})"
            )

        orig_len = audio.shape[0]
        orig_dur = orig_len / sr

        # 기본: 속도는 그대로 (rate = 1.0)
        used_rate = 1.0
        used_speedup = 1.0

        if orig_dur > target_dur:
            # ---- soft speedup 로직 ----
            # slot 안에 딱 맞추려면 필요한 속도
            ideal_rate = target_dur / orig_dur         # < 1.0
            ideal_speedup = 1.0 / ideal_rate           # > 1.0

            # 역할별 최대/soft 속도 설정
            if role in caster_roles:
                hard_max_speedup = caster_max_speedup
                # 캐스터는 너무 빠르면 어색하니 1.4 정도를 soft 상한으로
                soft_speedup = min(hard_max_speedup, 1.4)
            elif role in analyst_roles:
                hard_max_speedup = analyst_max_speedup
                # 해설은 약간 더 빠른 것도 허용 (예: 1.5)
                soft_speedup = min(hard_max_speedup, 1.5)
            else:
                hard_max_speedup = analyst_max_speedup
                soft_speedup = min(hard_max_speedup, 1.5)

            # ideal_speedup 이 귀에 거북하지 않은 구간(soft_speedup) 안이면
            # → slot 에 딱 맞게 속도 조절 (클리핑 없음)
            if ideal_speedup <= soft_speedup:
                used_speedup = ideal_speedup
            else:
                # slot 이 너무 좁아서, 자연스러운 속도로도 다 못 들어가는 경우:
                # 속도는 soft_speedup 까지만 올리고, 나머지는 약간 잘릴 수 있게 둔다.
                used_speedup = soft_speedup

            # 그래도 1배보다 느리게 하진 않는다.
            if used_speedup < 1.0:
                used_speedup = 1.0

            # 최종적으로는 hard_max_speedup 을 절대 넘지 않도록 안전장치
            used_speedup = min(used_speedup, hard_max_speedup)

            used_rate = 1.0 / used_speedup
        else:
            # orig_dur <= target_dur → 그대로 두고, 남는 건 gap으로 둔다
            used_rate = 1.0
            used_speedup = 1.0

        # ---- WSOLA 적용 ----
        warped = _wsola_tsm(audio, used_rate)
        warped = np.asarray(warped)

        # 최종적으로 항상 (N, C) 형태 보장
        if warped.ndim == 1:
            warped = warped[:, None]      # (N,)   -> (N, 1)
        elif warped.ndim == 2 and warped.shape[0] == 1 and warped.shape[1] > 1:
            warped = warped.T             # (1, N) -> (N, 1)

        warped_len = warped.shape[0]
        warped_dur = warped_len / sr

        # 그래도 길면 tail 컷 (특히 rate 제한 때문에 ideal보다 길게 남는 경우)
        target_samples = int(target_dur * sr)
        if target_samples <= 0:
            print(f"[WSOLA_GAP][SKIP] target_samples<=0 utt={utt_id}")
            continue

        if warped_len > target_samples:
            warped = warped[:target_samples, :]
            warped_len = warped.shape[0]
            warped_dur = warped_len / sr

        start_sample = int(logical_start * sr)
        end_sample = start_sample + warped_len
        if end_sample > num_samples:
            end_sample = num_samples
            warped = warped[: end_sample - start_sample, :]
            warped_len = warped.shape[0]
            warped_dur = warped_len / sr

        # (N, C) + (N, C) 로 더하기
        timeline[start_sample:end_sample, :] += warped.astype(np.float32)

        cursor_sec = logical_start + warped_dur

        print(
            f"[WSOLA_GAP] utt={utt_id} role={role_raw:8s} "
            f"slot=({slot_start:7.3f}~{slot_end:7.3f}) "
            f"logical=({logical_start:7.3f}~{logical_end:7.3f}) "
            f"orig={orig_dur:6.3f}s -> target={target_dur:6.3f}s "
            f"speedup={used_speedup:4.2f} rate={used_rate:5.3f}"
        )

    out_wav_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_wav_path, timeline, sr)
    print("[WSOLA_GAP] saved timeline:", out_wav_path)
    return out_wav_path
