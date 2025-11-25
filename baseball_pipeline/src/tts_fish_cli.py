# src/tts_fish_cli.py

from __future__ import annotations

import os
import subprocess as sp
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List
import sys   # ✅ 추가

# ✅ 현재 커널 파이썬 경로 고정
PYTHON_BIN = sys.executable

# ================================
# 기본 경로 설정
# ================================
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]          # /workspace/baseball_pipeline
FISH_ROOT = PROJECT_ROOT / "fish-speech"     # /workspace/baseball_pipeline/fish-speech

CHECKPOINT_DIR = FISH_ROOT / "checkpoints" / "openaudio-s1-mini"
CODEC_CKPT = CHECKPOINT_DIR / "codec.pth"

# 기본 참조 음성 위치 (필요하면 노트북에서 override)
DEFAULT_PROMPT_WAV = FISH_ROOT / "references" / "prompt.wav"
DEFAULT_PROMPT_TOKENS = FISH_ROOT / "references" / "prompt.npy"

DEFAULT_TEMP_DIR = FISH_ROOT / "temp_tts"


# ================================
# 유틸: subprocess 래퍼
# ================================
def _run_cmd(cmd: List[str], cwd: Path, env: Optional[dict] = None) -> None:
    """간단한 subprocess 래퍼 (stdout 실시간 프린트)."""
    print("\n[CMD]", " ".join(cmd))
    print("[CWD]", cwd)

    proc = sp.Popen(
        cmd,
        cwd=str(cwd),
        stdout=sp.PIPE,
        stderr=sp.STDOUT,
        text=True,
        env=env,
    )
    assert proc.stdout is not None

    for line in proc.stdout:
        print(line, end="")

    ret = proc.wait()
    if ret != 0:
        raise RuntimeError(f"Command failed with exit code {ret}: {' '.join(cmd)}")


# ================================
# 설정 / 클래스 정의
# ================================
@dataclass
class FishTTSConfig:
    fish_root: Path = FISH_ROOT
    llama_ckpt_dir: Path = CHECKPOINT_DIR
    codec_ckpt: Path = CODEC_CKPT

    prompt_wav: Path = DEFAULT_PROMPT_WAV
    prompt_tokens: Path = DEFAULT_PROMPT_TOKENS
    prompt_text: str = "프롬프트 녹음 때 읽었던 문장을 여기에 그대로 적어줘."

    temp_dir: Path = DEFAULT_TEMP_DIR

    device: str = "cuda"      # GPU 강제
    half: bool = True         # bf16 이슈 있을 때는 half 사용
    compile_t2s: bool = False # 많이 돌릴 때 True 로 바꿔도 됨


class FishSpeechTTS:
    """
    Fish-Speech / OpenAudio-S1-mini CLI 래퍼.
    - 참조 음성 → VQ 토큰(prompt_tokens)
    - 텍스트 → semantic codes → 최종 wav
    """

    def __init__(self, cfg: Optional[FishTTSConfig] = None):
        self.cfg = cfg or FishTTSConfig()
        self._check_paths()

    # ------------------------------
    # 내부 유틸
    # ------------------------------
    def _check_paths(self):
        if not self.cfg.fish_root.exists():
            raise FileNotFoundError(f"fish-speech 루트가 없음: {self.cfg.fish_root}")
        if not self.cfg.llama_ckpt_dir.exists():
            raise FileNotFoundError(
                f"OpenAudio S1-mini 체크포인트 폴더가 없음: {self.cfg.llama_ckpt_dir}"
            )
        if not self.cfg.codec_ckpt.exists():
            raise FileNotFoundError(
                f"codec.pth 를 찾을 수 없음: {self.cfg.codec_ckpt}"
            )
        self.cfg.temp_dir.mkdir(parents=True, exist_ok=True)
        self.cfg.prompt_tokens.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------
    # 1) 참조 음성 → prompt_tokens.npy
    # ------------------------------
    def ensure_prompt_tokens(self) -> Path:
        """
        prompt_tokens(.npy)가 없으면
        dac/inference.py로 fake.npy 생성 후 원하는 경로로 rename.
        """
        if self.cfg.prompt_tokens.exists():
            print(f"[TTS] prompt_tokens 존재: {self.cfg.prompt_tokens}")
            return self.cfg.prompt_tokens

        if not self.cfg.prompt_wav.exists():
            raise FileNotFoundError(f"prompt.wav 이 없음: {self.cfg.prompt_wav}")

        print("[TTS] prompt_tokens 없음 → 참조 음성에서 VQ 토큰 추출 시작")

        cmd = [
            PYTHON_BIN,
            "fish_speech/models/dac/inference.py",
            "-i",
            str(self.cfg.prompt_wav),
            "--checkpoint-path",
            str(self.cfg.codec_ckpt),
        ]

        env = os.environ.copy()
        env.setdefault("CUDA_VISIBLE_DEVICES", "0")

        _run_cmd(cmd, cwd=self.cfg.fish_root, env=env)

        fake_npy = self.cfg.fish_root / "fake.npy"
        if not fake_npy.exists():
            raise FileNotFoundError(
                f"참조 인코딩 후 fake.npy 가 안 보임: {fake_npy}"
            )

        print(f"[TTS] fake.npy → {self.cfg.prompt_tokens} 로 이동")
        fake_npy.rename(self.cfg.prompt_tokens)

        return self.cfg.prompt_tokens

    # ------------------------------
    # 2) 텍스트 → semantic codes
    # ------------------------------
    def text_to_codes(self, text: str, prompt_text: Optional[str] = None) -> Path:
        """
        한 문장을 semantic 토큰 codes_*.npy로 변환.
        가장 최근 생성된 codes_*.npy 경로를 반환.
        """
        text = text.strip()
        if not text:
            raise ValueError("빈 텍스트는 TTS 할 수 없음")

        prompt_tokens = self.ensure_prompt_tokens()
        p_text = (prompt_text or self.cfg.prompt_text).strip()
        if not p_text:
            raise ValueError("prompt_text 가 비어 있음. 참조 음성에서 읽은 문장을 넣어줘.")

        print(f"[TTS] Text→Codes: \"{text[:40]}...\"")

        # 예전 codes_* 정리
        for f in self.cfg.temp_dir.glob("codes_*.npy"):
            try:
                f.unlink()
            except Exception:
                pass

        cmd = [
            PYTHON_BIN,
            "fish_speech/models/text2semantic/inference.py",
            "--text",
            text,
            "--prompt-text",
            p_text,
            "--prompt-tokens",
            str(prompt_tokens),
            "--checkpoint-path",
            str(self.cfg.llama_ckpt_dir),
            "--num-samples",
            "1",
            "--output-dir",
            str(self.cfg.temp_dir),
            "--device",
            self.cfg.device,
        ]

        if self.cfg.half:
            cmd.append("--half")
        if self.cfg.compile_t2s:
            cmd.append("--compile")

        env = os.environ.copy()
        env.setdefault("CUDA_VISIBLE_DEVICES", "0")

        _run_cmd(cmd, cwd=self.cfg.fish_root, env=env)

        cand = sorted(self.cfg.temp_dir.glob("codes_*.npy"), key=lambda p: p.stat().st_mtime)
        if not cand:
            raise FileNotFoundError(
                f"semantic codes(codes_*.npy)를 {self.cfg.temp_dir} 에서 찾지 못했음"
            )

        latest = cand[-1]
        print(f"[TTS] 생성된 codes 파일: {latest}")
        return latest

    # ------------------------------
    # 3) semantic codes → wav
    # ------------------------------
    def codes_to_wav(self, codes_path: Path, out_wav: Path) -> Path:
        out_wav = Path(out_wav)
        out_wav.parent.mkdir(parents=True, exist_ok=True)

        print(f"[TTS] Codes→Wav: {codes_path.name} → {out_wav}")
        cmd = [
            PYTHON_BIN,
            "fish_speech/models/dac/inference.py",
            "-i",
            str(codes_path),
            "--output-path",
            str(out_wav),
        ]

        env = os.environ.copy()
        env.setdefault("CUDA_VISIBLE_DEVICES", "0")

        _run_cmd(cmd, cwd=self.cfg.fish_root, env=env)

        if not out_wav.exists():
            raise FileNotFoundError(f"TTS 결과 wav가 생성되지 않았음: {out_wav}")
        return out_wav

    # ------------------------------
    # 4) one-shot: text → wav
    # ------------------------------
    def tts(self, text: str, out_wav: Path, prompt_text: Optional[str] = None) -> Path:
        t0 = time.time()
        codes = self.text_to_codes(text, prompt_text=prompt_text)
        wav = self.codes_to_wav(codes, out_wav)
        dt = time.time() - t0
        print(f"[TTS] 전체 TTS 완료 ({dt:.1f}초): {wav}")
        return wav
