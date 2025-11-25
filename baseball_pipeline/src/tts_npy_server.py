# src/tts_npy_server.py
"""
Fish-Speech OpenAudio-S1-mini + .npy 프롬프트 전용 TTS 서버

- 캐스터 / 해설위원용 VQ 토큰(.npy)을 미리 만들어둔 상태에서,
  HTTP 요청(text, role)을 받아서
  -> text2semantic CLI
  -> dac CLI
  를 호출해서 wav를 만들어서 바로 반환한다.

⚠️ 주의:
- 지금 버전은 fish-speech 모델을 "CLI로" 호출한다.
  (= 요청마다 모델을 다시 로딩하므로 느릴 수 있음)
- 장점: baseball_pipeline.ipynb에서 쓰던 CLI 명령을 그대로 쓰기 때문에
  음질 / 목소리는 예전에 잘 나왔던 것과 동일하게 맞춰진다.
"""

from __future__ import annotations

import os
import sys
import uuid
import subprocess as sp
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn

# -------------------------------------------------------------------
# 경로 설정
# -------------------------------------------------------------------

# 프로젝트 루트: /workspace/baseball_pipeline
PROJECT_ROOT = Path(__file__).resolve().parents[1]
FISH_ROOT = PROJECT_ROOT / "fish-speech"

CHECKPOINT_ROOT = FISH_ROOT / "checkpoints" / "openaudio-s1-mini"
DAC_CKPT = CHECKPOINT_ROOT / "codec.pth"

TTS_REFS_DIR = PROJECT_ROOT / "data" / "tts_refs"
CASTER_NPY = TTS_REFS_DIR / "caster_prompt.npy"
ANALYST_NPY = TTS_REFS_DIR / "analyst_pakchanho_prompt.npy"

TEMP_DIR = FISH_ROOT / "temp_npy_server"

print("[TTS_NPY_SERVER] PROJECT_ROOT:", PROJECT_ROOT)
print("[TTS_NPY_SERVER] FISH_ROOT   :", FISH_ROOT)
print("[TTS_NPY_SERVER] LLAMA_CKPT  :", CHECKPOINT_ROOT)
print("[TTS_NPY_SERVER] DAC_CKPT    :", DAC_CKPT)
print("[TTS_NPY_SERVER] CASTER_NPY  :", CASTER_NPY, "| exists:", CASTER_NPY.exists())
print("[TTS_NPY_SERVER] ANALYST_NPY :", ANALYST_NPY, "| exists:", ANALYST_NPY.exists())

# 기본 안전 체크
if not FISH_ROOT.exists():
    raise RuntimeError(f"fish-speech 폴더가 없음: {FISH_ROOT}")

if not CHECKPOINT_ROOT.exists():
    raise RuntimeError(f"OpenAudio S1-mini 체크포인트 폴더가 없음: {CHECKPOINT_ROOT}")

if not DAC_CKPT.exists():
    raise RuntimeError(f"DAC codec.pth 를 찾지 못함: {DAC_CKPT}")

if not CASTER_NPY.exists() or not ANALYST_NPY.exists():
    raise RuntimeError("caster_prompt.npy / analyst_pakchanho_prompt.npy 둘 다 준비돼 있어야 함.")

TEMP_DIR.mkdir(parents=True, exist_ok=True)

# torch는 단순히 device 확인용으로만 쓴다.
try:
    import torch  # type: ignore

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except Exception:
    DEVICE = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"

print("[TTS_NPY_SERVER] DEVICE:", DEVICE)

# -------------------------------------------------------------------
# 요청/응답 스키마
# -------------------------------------------------------------------


class NpyTTSRequest(BaseModel):
    """클라이언트에서 보내는 JSON 구조"""

    text: str
    role: Literal["caster", "analyst"] = "caster"
    # baseball_pipeline 쪽에서 파일 이름 정해두고 싶으면 같이 보낼 수 있음
    utterance_id: str | None = None


# response_model 대신 FileResponse를 바로 쓸 거라서 별도 모델은 안 씀


# -------------------------------------------------------------------
# 내부 유틸
# -------------------------------------------------------------------


def _build_text2semantic_cmd(
    text: str,
    prompt_text: str,
    prompt_npy: Path,
    out_dir: Path,
) -> list[str]:
    """
    예전에 ipynb에서 사용하던 text2semantic CLI와 똑같은 형식으로 명령어를 만든다.

    예시 (기억해 두자):
    python fish_speech/models/text2semantic/inference.py \
        --text "연속 탈삼진" \
        --prompt-text "캐스터 프롬프트 음성입니다." \
        --prompt-tokens "/workspace/.../caster_prompt.npy" \
        --checkpoint-path "/workspace/.../checkpoints/openaudio-s1-mini" \
        --num-samples 1 \
        --output-dir "/workspace/.../fish-speech/temp_tts" \
        --device cuda \
        --half
    """
    cmd = [
        sys.executable,
        "fish_speech/models/text2semantic/inference.py",
        "--text",
        text,
        "--prompt-text",
        prompt_text,
        "--prompt-tokens",
        str(prompt_npy),
        "--checkpoint-path",
        str(CHECKPOINT_ROOT),
        "--num-samples",
        "1",
        "--output-dir",
        str(out_dir),
        "--device",
        DEVICE,
    ]
    
    return cmd


def _build_dac_cmd(codes_path: Path, out_wav: Path) -> list[str]:
    """
    예전에 ipynb에서 사용하던 DAC CLI와 동일한 포맷으로 명령어 생성.
    대략 이런 느낌:

    python fish_speech/models/dac/inference.py \
        -i "codes_0.npy" \
        --checkpoint-path "checkpoints/openaudio-s1-mini/codec.pth" \
        --output-path "out.wav" \
        --device cuda \
        --half
    """
    cmd = [
        sys.executable,
        "fish_speech/models/dac/inference.py",
        "-i",
        str(codes_path),
        "--checkpoint-path",
        str(DAC_CKPT),
        "--output-path",
        str(out_wav),
        "--device",
        DEVICE,
    ]
    
    return cmd


def _run_cmd(cmd: list[str], cwd: Path) -> None:
    """공통 subprocess 실행 + 에러 메시지 정리."""
    print("[TTS_NPY_SERVER] [CMD]", " ".join(cmd))
    print("[TTS_NPY_SERVER] [CWD]", cwd)

    proc = sp.run(
        cmd,
        cwd=str(cwd),
        stdout=sp.PIPE,
        stderr=sp.STDOUT,
        text=True,
    )

    if proc.returncode != 0:
        print("[TTS_NPY_SERVER] COMMAND FAILED (returncode =", proc.returncode, ")")
        print("===== BEGIN OUTPUT =====")
        print(proc.stdout)
        print("=====  END  OUTPUT =====")
        raise RuntimeError(f"명령 실행 실패 (code={proc.returncode})")


# -------------------------------------------------------------------
# FastAPI 앱
# -------------------------------------------------------------------

app = FastAPI(title="Fish-Speech .npy TTS Server")


@app.post("/v1/tts", response_class=FileResponse)
def tts_endpoint(req: NpyTTSRequest):
    """
    text + role(caster/analyst)를 받아서
    .npy 프롬프트 기반으로 wav를 생성해서 바로 반환.
    """
    # 어떤 프롬프트 사용할지 선택
    if req.role == "caster":
        prompt_npy = CASTER_NPY
        prompt_text = "캐스터 프롬프트 음성입니다."
    else:
        prompt_npy = ANALYST_NPY
        prompt_text = "해설위원 프롬프트 음성입니다."

    if not prompt_npy.exists():
        raise HTTPException(status_code=500, detail=f"prompt npy 파일이 없음: {prompt_npy}")

    # 요청마다 개별 temp 디렉토리 생성
    utt_id = req.utterance_id or uuid.uuid4().hex
    work_dir = TEMP_DIR / utt_id
    work_dir.mkdir(parents=True, exist_ok=True)

    # 1단계: text -> semantic codes
    cmd_sem = _build_text2semantic_cmd(
        text=req.text,
        prompt_text=prompt_text,
        prompt_npy=prompt_npy,
        out_dir=work_dir,
    )
    try:
        _run_cmd(cmd_sem, cwd=FISH_ROOT)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=f"text2semantic 실패: {e}")

    # inference.py 기본 출력: codes_0.npy
    codes_path = work_dir / "codes_0.npy"
    if not codes_path.exists():
        # 혹시 이름이 다르면 나중에 여기서 조정
        raise HTTPException(status_code=500, detail=f"codes_0.npy를 찾지 못함: {codes_path}")

    # 2단계: semantic codes -> wav
    out_wav = work_dir / f"{utt_id}.wav"
    cmd_dac = _build_dac_cmd(codes_path=codes_path, out_wav=out_wav)
    try:
        _run_cmd(cmd_dac, cwd=FISH_ROOT)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=f"DAC(inference.py) 실패: {e}")

    if not out_wav.exists():
        raise HTTPException(status_code=500, detail=f"wav 파일 생성 실패: {out_wav}")

    # 바로 파일로 응답 (클라이언트에서 content를 그대로 저장하면 됨)
    return FileResponse(
        path=str(out_wav),
        media_type="audio/wav",
        filename=out_wav.name,
    )


# -------------------------------------------------------------------
# 엔트리포인트
# -------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=False,
    )


if __name__ == "__main__":
    main()
