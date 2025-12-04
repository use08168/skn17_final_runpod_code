# src/llm_kanana.py

from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Optional

import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# =========================
#  모델 / 토크나이저 로더
# =========================

BASE_MODEL_ID = "kakaocorp/kanana-1.5-8b-instruct-2505"
LORA_MODEL_ID = "SeHee8546/kanana-1.5-8b-pakchanho-lora"

_TOKENIZER: Optional[AutoTokenizer] = None
_MODEL: Optional[torch.nn.Module] = None


def get_kanana_pakchanho_model():
    """
    kakaocorp/kanana-1.5-8b-instruct-2505 + pakchanho LoRA 를
    한 번만 로드해서 캐시한다.
    """
    global _TOKENIZER, _MODEL
    if _MODEL is not None and _TOKENIZER is not None:
        return _TOKENIZER, _MODEL

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[KANANA] loading base model on {device} ...")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)

    if device == "cuda":
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            torch_dtype=torch.float32,
        )

    print("[KANANA] loading LoRA:", LORA_MODEL_ID)
    if device == "cuda":
        model = PeftModel.from_pretrained(
            base_model,
            LORA_MODEL_ID,
            torch_dtype=torch.bfloat16,
        )
    else:
        model = PeftModel.from_pretrained(
            base_model,
            LORA_MODEL_ID,
            torch_dtype=torch.float32,
        )

    model.eval()

    _TOKENIZER = tokenizer
    _MODEL = model
    print("[KANANA] model + LoRA ready.")
    return _TOKENIZER, _MODEL


# =========================
#  프롬프트 (LLaMA 스타일 + 1인칭/메타 금지)
# =========================

BASE_SYSTEM_PROMPT = (
    "당신은 프로야구 해설위원 박찬호이다. "
    "입력으로 들어오는 문장은 캐스터가 방금 한 멘트이거나 현재 상황을 말하는 문장이다. "
    "당신은 시청자에게 야구 상황을 이해시키기 위해 실제 중계 방송처럼 해설 멘트를 말해야 한다. "
    "대답은 반드시 한국어로 하며, 박찬호 특유의 말투(말을 길게 하고, 중간에 숨을 고르듯 끊고, "
    "\"~같아요\", \"~하죠\" 등)를 사용합니다. "
    "「나, 저, 박찬호 입니다, 박찬호 위원」과 같은 1인칭 표현은 사용하지 말하면 절대 안된다. "
    "캐스터 멘트를 그대로 반복하지 말고 기존 해설위원의 정보를 바탕으로 박찬호라면 이런식으로 말할 것 같은 문잔을 만들어야 한다. "
    "한 번에 두세 문장 정도만 말한다. "
    "지침이나 규칙, 출력 형식, 요약 설명을 말하지 말고, 실제 방송에서 나올 법한 해설 멘트만 출력한다."
)


def build_pakchanho_messages(
    orig_text: str,
    game_desc: Optional[str] = None,
) -> List[Dict[str, str]]:
    """
    analyst row 의 orig_text 를 받아서 박찬호 스타일 해설을 생성하기 위한
    최소 messages 포맷을 만든다.
    """
    orig_text = str(orig_text).strip()

    if game_desc:
        user_content = (
            f"경기 상황 설명:\n{game_desc}\n\n"
            f"캐스터 또는 원본 멘트:\n{orig_text}\n\n"
            "위 멘트 직후에, 해설위원 박찬호가 실제 중계에서 말할 법한 해설 멘트를 한 번 작성하세요."
        )
    else:
        user_content = (
            f"캐스터 또는 원본 멘트:\n{orig_text}\n\n"
            "위 멘트 직후에, 해설위원 박찬호가 실제 중계에서 말할 법한 해설 멘트를 한 번 작성하세요."
        )

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": BASE_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    return messages


# =========================
#  후처리 (프롬프트 냄새 제거 + fallback)
# =========================

def _postprocess_pakchanho(text: str, orig_text: Optional[str] = None) -> str:
    """
    - 불릿("- ", "•")이나 번호("1.", "2.")로 시작하는 줄 제거
    - 프롬프트 지침/메타 같은 문장 제거
    - 만약 다 지워져버리면 orig_text 로 fallback
    """
    if not text:
        return orig_text.strip() if orig_text else ""

    lines = text.splitlines()
    cleaned_lines = []

    bad_keywords = [
        "지침",
        "규칙",
        "출력 형식",
        "사용 금지",
        "1 인칭",
        "1인칭",
        "입력된 상황",
        "맥락",
        "배경정보",
        "반영하여 답변하라",
    ]

    for ln in lines:
        s = ln.strip()
        if not s:
            continue

        # 불릿/번호 목록 제거
        if s.startswith("- ") or s.startswith("•"):
            continue
        if len(s) > 2 and s[0].isdigit() and s[1] in [".", ")"]:
            continue

        # 메타/지침 문장 제거
        if any(k in s for k in bad_keywords):
            continue

        cleaned_lines.append(s)

    cleaned = " ".join(cleaned_lines).strip()

    # 다 지워졌으면 → orig_text로 fallback
    if not cleaned:
        return orig_text.strip() if orig_text else text.strip()

    return cleaned


# =========================
#  단일 문장 생성 함수
# =========================

def generate_pakchanho_reply(
    orig_text: str,
    game_desc: Optional[str] = None,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    """
    orig_text (STT에서 온 해설/캐스터 멘트)를 넣으면
    박찬호 스타일 한 줄(2~3문장)을 생성해서 반환한다.
    """
    tokenizer, model = get_kanana_pakchanho_model()

    messages = build_pakchanho_messages(orig_text=orig_text, game_desc=game_desc)

    # LLaMA 계열 chat 템플릿 사용
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(prompt, return_tensors="pt")
    device = model.device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    pad_id = tokenizer.eos_token_id
    if pad_id is None:
        pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = 0

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=pad_id,
        )

    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    text = text.strip()
    text = _postprocess_pakchanho(text, orig_text=orig_text)
    return text


# =========================
#  CSV 배치 처리
# =========================

def run_llm_kanana_for_tts_csv(
    csv_path: Path | str,
    out_path: Optional[Path | str] = None,
    max_rows: Optional[int] = None,
    game_desc: Optional[str] = None,
) -> Path:
    """
    clip.tts_phrases.csv 용 LLM 단계.

    - 입력: STT/전처리까지 끝난 CSV
      (컬럼: utterance_id, role, orig_text, llm_text 등)
    - role == "analyst" 인 row 에 대해서만, orig_text 를 LLM에 넣어
      박찬호 스타일 멘트를 생성하고 llm_text 에 채운다.
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    df.columns = [c.replace("\ufeff", "").strip() for c in df.columns]

    required_cols = {"utterance_id", "role", "orig_text"}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"[KANANA] CSV에 {required_cols} 가 필요합니다. "
            f"현재 컬럼: {df.columns.tolist()}"
        )

    if "llm_text" not in df.columns:
        df["llm_text"] = ""

    rows = list(df.itertuples(index=True))
    if max_rows is not None:
        rows = rows[:max_rows]

    for row in rows:
        idx = row.Index
        role_raw = str(getattr(row, "role", "")).strip()
        role = role_raw.lower()

        # analyst 만 LLM 통과
        if role != "analyst":
            continue

        orig = str(getattr(row, "orig_text", "")).strip()
        if not orig or orig.lower() == "nan":
            continue

        utt_id = getattr(row, "utterance_id", "<?>")

        try:
            reply = generate_pakchanho_reply(
                orig_text=orig,
                game_desc=game_desc,
            )
            df.at[idx, "llm_text"] = reply
            print(
                f"[KANANA] idx={idx} utt={utt_id} role={role_raw} "
                f"orig='{orig[:30]}...' -> llm='{reply[:30]}...'"
            )
        except Exception as e:
            print(f"[KANANA] ERROR idx={idx} utt={utt_id}: {e}")

    if out_path is None:
        out_path = csv_path.with_name(csv_path.stem + ".llm_kanana.csv")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print("[KANANA] saved CSV:", out_path)
    return out_path
