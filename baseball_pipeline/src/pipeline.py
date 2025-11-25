# src/pipeline.py

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from . import PROJECT_ROOT, DATA_DIR, FAISS_DIR
from .stt_clova import run_stt_to_csv
from .rag_faiss import load_full_match_log
from .llm_kanana import KananaConfig, KananaCommentaryModel
from .tts_fishspeech import synthesize_with_openaudio

# 디렉토리들
INPUT_DIR = DATA_DIR / "input_videos"
STT_SEG_DIR = DATA_DIR / "stt_segments"
LLM_OUT_DIR = DATA_DIR / "llm_outputs"
TTS_DIR = DATA_DIR / "tts_audio"
FINAL_DIR = DATA_DIR / "final_videos"

for d in [INPUT_DIR, STT_SEG_DIR, LLM_OUT_DIR, TTS_DIR, FINAL_DIR]:
    d.mkdir(parents=True, exist_ok=True)


@dataclass
class PipelineConfig:
    # ==== STT (CLOVA) ====
    clova_invoke_url: str       # 🔥 여기에 invoke URL 하드코딩 or 호출 시 전달
    clova_secret_key: str       # 🔥 여기에 CLOVA 시크릿 키

    stt_keyword_xlsx: str       # 키워드 엑셀 경로 (예: "/workspace/baseball_pipeline/stt.xlsx")

    # ==== RAG (FAISS + OpenAI Embeddings) ====
    faiss_db_path: Optional[str] = None  # None이면 /faiss_index
    openai_api_key: Optional[str] = None # 🔥 필요하면 OpenAI API 키

    # ==== LLM (Kanana) ====
    hf_token: Optional[str] = None       # 🔥 Hugging Face 토큰
    match_title: str = "2025 한국시리즈 1차전"

    # ==== TTS (Fish-speech / OpenAudio) ====
    fish_checkpoint_dir: Optional[str] = None  # None이면 PROJECT_ROOT/fish-speech/checkpoints/openaudio-s1-mini
    caster_prompt_wav: str = str(PROJECT_ROOT / "fish-speech" / "references" / "caster_prompt.wav")
    commentator_prompt_wav: str = str(PROJECT_ROOT / "fish-speech" / "references" / "commentator_prompt.wav")

    # ==== 화자 라벨 → 역할 매핑 ====
    # 예: A=캐스터, B=해설
    speaker_role_map: Dict[str, str] = field(default_factory=lambda: {"A": "caster", "B": "commentator"})

    def get_faiss_db_path(self) -> str:
        if self.faiss_db_path is not None:
            return self.faiss_db_path
        return str(FAISS_DIR)

    def get_fish_checkpoint_dir(self) -> Path:
        if self.fish_checkpoint_dir is not None:
            return Path(self.fish_checkpoint_dir)
        return PROJECT_ROOT / "fish-speech" / "checkpoints" / "openaudio-s1-mini"


def replace_audio_with_ffmpeg(
    input_mp4: Path,
    new_audio_wav: Path,
    output_mp4: Path,
) -> Path:
    """
    기존 영상의 '영상 트랙'은 그대로 두고, 오디오를 new_audio_wav로 교체.
    """
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_mp4),
        "-i",
        str(new_audio_wav),
        "-c:v",
        "copy",
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-shortest",
        str(output_mp4),
    ]
    print("[FFMPEG] 명령:", " ".join(cmd))
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg 실패 (return code={proc.returncode})")
    print(f"[FFMPEG] 새 영상 생성: {output_mp4}")
    return output_mp4


def run_full_pipeline(
    video_filename: str,   # data/input_videos 밑에 있는 파일 이름 (예: "my_clip.mp4")
    cfg: PipelineConfig,
) -> Dict[str, Path]:
    """
    1) MP4 → CLOVA STT CSV (data/stt_segments)
    2) FAISS에서 경기 로그 로드
    3) 해설 멘트를 박찬호 스타일로 변환
    4) 캐스터+해설 통합 스크립트 → TTS
    5) 기존 mp4에 새 오디오 입혀서 최종 mp4 생성

    ⚠ 현재 버전은 "전체 스크립트를 한 번에 TTS" 하는 단순 버전.
      나중에 segment 단위로 잘라서 타임스탬프 기반으로 합성하는 버전으로 확장 가능.
    """
    video_path = INPUT_DIR / video_filename
    if not video_path.exists():
        raise FileNotFoundError(f"입력 영상이 없습니다: {video_path}")

    # ---------- 1) STT ----------
    stt_csv = run_stt_to_csv(
        audio_path=video_path,
        xlsx_keywords_path=cfg.stt_keyword_xlsx,
        invoke_url=cfg.clova_invoke_url,
        secret_key=cfg.clova_secret_key,
        speaker_count_min=2,
        speaker_count_max=2,
        save_raw_json=True,
    )

    df = pd.read_csv(stt_csv)
    print(f"[PIPE] STT segments: {len(df)} rows")

    # ---------- 2) 경기 로그 (RAG) ----------
    match_log = load_full_match_log(
        target_match=cfg.match_title,
        db_path=cfg.get_faiss_db_path(),
        openai_api_key=cfg.openai_api_key,
    ) or ""
    print(f"[PIPE] 경기 로그 길이: {len(match_log)} chars")

    # ---------- 3) LLM: 해설 멘트 변환 ----------
    llm_cfg = KananaConfig(
        hf_token=cfg.hf_token,
        max_new_tokens=256,
    )
    llm = KananaCommentaryModel(llm_cfg)

    # 화자 라벨 → 역할
    def map_role(row):
        label = str(row.get("speaker_label") or "").strip()
        return cfg.speaker_role_map.get(label, "unknown")

    df["role"] = df.apply(map_role, axis=1)

    # commentator 변환 결과를 순서대로 담아 두기
    park_text_list = []

    for _, row in df.iterrows():
        text = str(row.get("text") or "").strip()
        if not text:
            park_text_list.append("")  # 자리만 맞추기
            continue

        role = row["role"]
        if role == "commentator":
            park_text = llm.generate_park_style(
                match_log=match_log,
                original_text=text,
            )
            park_text_list.append(park_text)
        else:
            park_text_list.append("")

    # ---------- 3-2) 통합 스크립트 만들기 ----------
    merged_script_lines = []
    park_idx = 0
    for _, row in df.iterrows():
        text = str(row.get("text") or "").strip()
        if not text:
            continue
        role = row["role"]

        if role == "caster":
            merged_script_lines.append(f"[캐스터] {text}")
        elif role == "commentator":
            park_text = park_text_list[park_idx]
            park_idx += 1
            if not park_text:
                park_text = text
            merged_script_lines.append(f"[박찬호] {park_text}")
        else:
            # 역할 모를 때는 그냥 원본
            merged_script_lines.append(text)

    merged_script = "\n".join(merged_script_lines)
    script_path = LLM_OUT_DIR / f"{video_path.stem}_script.txt"
    script_path.write_text(merged_script, encoding="utf-8")
    print(f"[PIPE] 통합 스크립트 저장: {script_path}")

    # ---------- 4) TTS ----------
    tts_out_wav = TTS_DIR / f"{video_path.stem}_tts.wav"
    _ = synthesize_with_openaudio(
        text=merged_script,
        speaker="commentator",  # 전체 스크립트를 일단 박찬호 목소리로
        output_wav=tts_out_wav,
        prompt_wav_caster=Path(cfg.caster_prompt_wav),
        prompt_wav_commentator=Path(cfg.commentator_prompt_wav),
        checkpoint_dir=cfg.get_fish_checkpoint_dir(),
        prompt_text="야구 중계 참고 음성에 해당하는 텍스트",
    )

    # ---------- 5) ffmpeg로 새로운 mp4 만들기 ----------
    final_mp4 = FINAL_DIR / f"{video_path.stem}_park_version.mp4"
    _ = replace_audio_with_ffmpeg(
        input_mp4=video_path,
        new_audio_wav=tts_out_wav,
        output_mp4=final_mp4,
    )

    print("[PIPE] 전체 파이프라인 완료.")
    return {
        "stt_csv": Path(stt_csv),
        "script_txt": script_path,
        "tts_wav": tts_out_wav,
        "final_mp4": final_mp4,
    }
