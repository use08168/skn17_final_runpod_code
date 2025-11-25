# src/audio_concat_test.py

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
from pydub import AudioSegment
import subprocess


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

    return None


def _get_video_duration_ms(input_video_path: Path) -> int:
    """
    비디오 파일에서 오디오 트랙을 읽어서 전체 길이(ms)를 구한다.
    (ffmpeg + pydub 사용)
    """
    seg = AudioSegment.from_file(str(input_video_path))
    return len(seg)


def build_sequential_tts_audio(
    llm_csv_path: Path | str,
    tts_audio_dir: Path | str,
    out_wav_path: Path | str,
    gap_ms_between_roles: int = 0,
    role_gain_db: Optional[dict[str, float]] = None,
) -> Path:
    """
    타임스탬프를 무시하고, CSV 에 있는 utterance 순서대로
    TTS wav들을 쭉 이어붙여 하나의 오디오를 만든다.

    - llm_csv_path: clip.tts_phrases.llm_kanana.csv 등
      (필수 컬럼: utterance_id, role)
      start_sec 이 있으면 그 순서대로 정렬, 없으면 CSV 순서를 그대로 사용
    - tts_audio_dir: 개별 TTS wav 가 있는 디렉토리
      (예: data/tts_audio/clip)
    - out_wav_path: 이어붙인 결과 wav 경로
    - gap_ms_between_roles:
      캐스터/해설위원 등 role 이 바뀔 때 삽입할 공백(ms).
      예: 150 → 캐스터→해설위원 전환마다 0.15초 침묵
    - role_gain_db:
      역할별 볼륨 조절(dB). 예:
        {"caster": +1.5, "analyst": -1.0}
      role 컬럼을 소문자로 normalize 해서 lookup 한다.
    """
    llm_csv_path = Path(llm_csv_path)
    tts_audio_dir = Path(tts_audio_dir)
    out_wav_path = Path(out_wav_path)

    df = pd.read_csv(llm_csv_path)
    df.columns = [c.replace("\ufeff", "").strip() for c in df.columns]

    if "utterance_id" not in df.columns:
        raise ValueError("[CONCAT] CSV 에 utterance_id 컬럼이 필요합니다.")

    # 정렬 기준: start_sec 이 있으면 그걸로, 아니면 입력 그대로
    if "start_sec" in df.columns:
        df = df.copy()
        df["start_sec"] = pd.to_numeric(df["start_sec"], errors="coerce")
        df = df.sort_values("start_sec")

    segments: list[AudioSegment] = []
    prev_role_norm: Optional[str] = None

    # role_gain_db 키는 소문자 기준으로 맞춰둔다
    if role_gain_db is not None:
        role_gain_db = {str(k).strip().lower(): float(v) for k, v in role_gain_db.items()}

    for row in df.itertuples(index=False):
        utt_id = str(getattr(row, "utterance_id"))
        role_raw = getattr(row, "role", "")
        role_norm = str(role_raw).strip().lower()

        wav_path = _find_tts_wav_for_utt(tts_audio_dir, utt_id, role=role_norm)
        if wav_path is None:
            print(f"[CONCAT] WARN: wav not found for utt={utt_id}, role={role_raw}")
            continue

        # 🔹 캐스터 ↔ 해설위원 등 화자가 바뀌는 순간엔 살짝 공백 삽입
        if (
            prev_role_norm is not None
            and role_norm
            and role_norm != prev_role_norm
            and gap_ms_between_roles > 0
        ):
            gap = AudioSegment.silent(duration=gap_ms_between_roles)
            segments.append(gap)
            print(
                f"[CONCAT] insert gap {gap_ms_between_roles}ms "
                f"between {prev_role_norm} -> {role_norm}"
            )

        seg = AudioSegment.from_file(wav_path)

        # 🔹 역할별 볼륨 조절 (dB 단위)
        gain_db = 0.0
        if role_gain_db is not None:
            gain_db = float(role_gain_db.get(role_norm, 0.0))
        if gain_db != 0.0:
            seg = seg + gain_db  # pydub: +dB / -dB
            print(
                f"[CONCAT] apply gain {gain_db:+.2f} dB "
                f"for role={role_raw} (utt={utt_id})"
            )

        segments.append(seg)
        print(
            f"[CONCAT] append utt={utt_id} role={role_raw} "
            f"dur={len(seg)/1000:.2f}s"
        )

        prev_role_norm = role_norm

    if not segments:
        raise ValueError("[CONCAT] 이어붙일 TTS 오디오가 없습니다.")

    full = segments[0]
    for seg in segments[1:]:
        full += seg

    out_wav_path.parent.mkdir(parents=True, exist_ok=True)
    full.export(out_wav_path, format="wav")
    print("[CONCAT] saved full TTS wav:", out_wav_path)
    return out_wav_path


def cut_audio_to_video_length(
    audio_path: Path | str,
    input_video_path: Path | str,
    out_wav_path: Optional[Path | str] = None,
) -> Path:
    """
    audio_path 의 길이가 비디오 길이보다 길면,
    비디오 길이에 맞춰 잘라낸 새 wav 를 생성한다.

    - 비디오 길이보다 짧으면 그대로 둔다.
    """
    audio_path = Path(audio_path)
    input_video_path = Path(input_video_path)
    if out_wav_path is None:
        out_wav_path = audio_path
    out_wav_path = Path(out_wav_path)

    video_ms = _get_video_duration_ms(input_video_path)
    audio = AudioSegment.from_file(str(audio_path))
    audio_ms = len(audio)

    print(
        f"[CUT] video_ms={video_ms}ms ({video_ms/1000:.2f}s), "
        f"audio_ms={audio_ms}ms ({audio_ms/1000:.2f}s)"
    )

    if audio_ms <= video_ms:
        if out_wav_path != audio_path:
            audio.export(out_wav_path, format="wav")
        print("[CUT] audio <= video, 그대로 사용:", out_wav_path)
        return out_wav_path

    trimmed = audio[:video_ms]
    out_wav_path.parent.mkdir(parents=True, exist_ok=True)
    trimmed.export(out_wav_path, format="wav")
    print("[CUT] trimmed audio to video length:", out_wav_path)
    return out_wav_path


def mux_tts_audio_to_video_concat(
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

    print("[MUX(CONCAT)] CMD:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print("[MUX(CONCAT)] saved video:", out_video_path)
    return out_video_path


def build_sequential_tts_video(
    llm_csv_path: Path | str,
    tts_audio_dir: Path | str,
    input_video_path: Path | str,
    out_wav_path: Path | str,
    out_video_path: Path | str,
    mute_original: bool = True,
    gap_ms_between_roles: int = 0,
    role_gain_db: Optional[dict[str, float]] = None,
) -> Path:
    """
    편의를 위한 one-shot 함수:

    1) CSV 순서대로 TTS wav 들을 이어붙여 하나의 오디오 생성
       - role 이 바뀔 때마다 gap_ms_between_roles 만큼 침묵 삽입
       - role_gain_db 로 역할별 볼륨 조절
    2) 그 오디오가 영상보다 길면, 영상 길이에 맞춰 잘라냄
    3) 잘라낸 오디오를 영상에 붙여서 최종 mp4 생성
    """
    llm_csv_path = Path(llm_csv_path)
    tts_audio_dir = Path(tts_audio_dir)
    input_video_path = Path(input_video_path)
    out_wav_path = Path(out_wav_path)
    out_video_path = Path(out_video_path)

    full_wav = build_sequential_tts_audio(
        llm_csv_path=llm_csv_path,
        tts_audio_dir=tts_audio_dir,
        out_wav_path=out_wav_path,
        gap_ms_between_roles=gap_ms_between_roles,
        role_gain_db=role_gain_db,
    )

    trimmed_wav = cut_audio_to_video_length(
        audio_path=full_wav,
        input_video_path=input_video_path,
        out_wav_path=out_wav_path,  # 같은 경로에 덮어쓰기
    )

    final_video = mux_tts_audio_to_video_concat(
        input_video_path=input_video_path,
        tts_audio_path=trimmed_wav,
        out_video_path=out_video_path,
        mute_original=mute_original,
    )
    return final_video
