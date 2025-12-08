from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import PeftModel


# =========================================
# 1. 데이터 구조
# =========================================

@dataclass
class SetItem:
    set_id: str
    video_id: str
    set_start_sec: float
    set_end_sec: float
    caster_text: str
    analyst_text: Optional[str]
    scoreboard: Dict[str, Any]


# =========================================
# 2. 모델 로드
# =========================================

def load_pakchanho_model(
    base_model_name: str = "kakaocorp/kanana-1.5-8b-instruct-2505",
    lora_model_id: str = "SeHee8546/kanana-1.5-8b-pakchanho-lora-v2",
    load_in_4bit: bool = True,
):
    """
    Kanana 8B + 박찬호 LoRA 로드 (4bit 양자화 옵션).
    """

    # =============================
    # 1) 디바이스 및 dtype 설정
    # =============================
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        if major >= 8:
            compute_dtype = torch.bfloat16
        else:
            compute_dtype = torch.float16
        device_map = "auto"
    else:
        compute_dtype = torch.float32
        device_map = "cpu"
        load_in_4bit = False  # CPU에서는 4bit 비권장

    # =============================
    # 2) 베이스 모델 로드
    # =============================
    if load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            device_map=device_map,
            trust_remote_code=True,
        )
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=compute_dtype,
            device_map=device_map,
            trust_remote_code=True,
        )

    # =============================
    # 3) LoRA 가중치 merge
    # =============================
    model = PeftModel.from_pretrained(
        base_model,
        lora_model_id,
        device_map=device_map,
    )

    # =============================
    # 4) 토크나이저 로드
    #    -> trust_remote_code / use_fast 옵션 빼고 공식 예제처럼
    # =============================
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        # trust_remote_code=False 가 기본값
        # use_fast=True 가 기본값
    )

    # 혹시 정말로 이상한 객체가 들어왔을 경우 (환경 꼬임 방지용)
    if not hasattr(tokenizer, "pad_token"):
        raise TypeError(
            f"AutoTokenizer.from_pretrained('{base_model_name}') 가 예상과 다르게 "
            f"{type(tokenizer)} 를 반환했습니다. transformers 버전/환경을 확인해주세요."
        )

    # =============================
    # 5) padding / eos 설정
    # =============================
    if tokenizer.pad_token is None:
        # LLaMA 계열은 pad_token 이 없는 경우가 많아서 eos_token 으로 맞춰주는 패턴이 일반적
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    tokenizer.padding_side = "left"

    model.eval()
    print("[INFO] Pakchanho model + tokenizer loaded.")
    return model, tokenizer


# =========================================
# 3. scoreboard 요약
# =========================================

def summarize_scoreboard(sb: Optional[Dict[str, Any]]) -> str:
    if not sb or not isinstance(sb, dict):
        return "스코어보드에 표시된 정보가 없습니다."

    # 모든 값이 None이면 정보 없음
    vals = []
    for v in sb.values():
        if isinstance(v, dict):
            vals.extend(list(v.values()))
        else:
            vals.append(v)
    if all(v is None for v in vals):
        return "스코어보드에 표시된 정보가 없습니다."

    lines: List[str] = []

    away = sb.get("원정팀")
    home = sb.get("홈팀")
    a_score = sb.get("원정팀 점수")
    h_score = sb.get("홈팀 점수")

    if away or home:
        parts = []
        if away:
            parts.append(f"{away} {a_score}점" if a_score is not None else str(away))
        if home:
            parts.append(f"{home} {h_score}점" if h_score is not None else str(home))
        if parts:
            lines.append(" vs ".join(parts))

    inning = sb.get("이닝")
    half = sb.get("이닝 상황")
    if inning is not None or half is not None:
        inning_str = f"{inning}회" if inning is not None else "이닝 정보 없음"
        if half in ("초", "말"):
            lines.append(f"{inning_str} {half} 상황")
        else:
            lines.append(inning_str)

    balls = sb.get("볼")
    strikes = sb.get("스트라이크")
    outs = sb.get("아웃")
    bso_parts = []
    if balls is not None:
        bso_parts.append(f"{balls}볼")
    if strikes is not None:
        bso_parts.append(f"{strikes}스트라이크")
    if outs is not None:
        bso_parts.append(f"{outs}아웃")
    if bso_parts:
        lines.append("볼카운트: " + ", ".join(bso_parts))

    bases = sb.get("주자")
    if isinstance(bases, dict):
        runners = []
        for base_k in ("1루", "2루", "3루"):
            v = bases.get(base_k)
            if v:
                runners.append(base_k)
        if runners:
            lines.append("주자: " + ", ".join(runners) + " 주자")
        else:
            lines.append("주자: 없음")

    pitcher = sb.get("투수 이름")
    pitch_count = sb.get("투구 수")
    batter = sb.get("타자 이름")
    bat_order = sb.get("타자 타순")
    bat_record = sb.get("타자 경기 기록")

    if pitcher is not None or pitch_count is not None:
        if pitcher and pitch_count is not None:
            lines.append(f"투수 {pitcher}, 투구 수 {pitch_count}개")
        elif pitcher:
            lines.append(f"투수 {pitcher}")

    if batter is not None:
        info = f"타자 {batter}"
        extras = []
        if bat_order is not None:
            extras.append(f"{bat_order}번")
        if bat_record is not None:
            extras.append(bat_record)
        if extras:
            info += " (" + ", ".join(extras) + ")"
        lines.append(info)

    if not lines:
        return "스코어보드에 표시된 정보가 없습니다."

    return "\n".join(lines)


# =========================================
# 4. 길이 / 시간 제약 계산 + 트리밍
# =========================================

def compute_length_limits(
    current_set: SetItem,
    next_set: Optional[SetItem],
) -> Tuple[int, int, int]:
    """
    기존 analyst_text 길이와 다음 세트와의 gap을 고려해
    (orig_len, extra_chars, max_chars)를 계산한다.
    """
    if not current_set.analyst_text:
        return 0, 0, 0

    orig_len = len(current_set.analyst_text)
    gap = 0.0
    if next_set is not None:
        gap = max(0.0, float(next_set.set_start_sec) - float(current_set.set_end_sec))

    extra_chars = 0
    if gap > 2.0:
        extra_chars = int((gap - 2.0) * 4.0)

    max_chars = orig_len + extra_chars
    return orig_len, extra_chars, max_chars


def trim_to_max_chars(text: str, max_chars: int) -> str:
    """
    생성된 문장이 max_chars를 넘으면,
    한국어 문장 종료 패턴(다., 요., 니다., !, ?)을 기준으로 최대한 자연스럽게 자른다.
    """
    text = text.strip()
    if max_chars <= 0 or len(text) <= max_chars:
        return text

    candidate = text[: max_chars]

    # 문장 끝 패턴들
    patterns = ["다.", "요.", "니다.", "!", "?", "."]
    last_pos = -1
    for p in patterns:
        idx = candidate.rfind(p)
        if idx > last_pos:
            last_pos = idx + len(p)

    if last_pos > 0:
        return candidate[:last_pos].strip()
    else:
        return candidate.rstrip() + "..."


# =========================================
# 5. 세트별 프롬프트 구성 + LLM 호출
# =========================================

def build_messages_for_set(
    s: SetItem,
    next_s: Optional[SetItem],
    game_title: str,
) -> Tuple[List[Dict[str, str]], int]:
    """
    한 세트에 대한 system/user 메시지와, max_chars(길이 제한)를 반환.
    analyst_text가 없는 세트는 이 함수를 호출하지 말 것.
    """
    sb_summary = summarize_scoreboard(s.scoreboard)
    orig_analyst = s.analyst_text or ""

    orig_len, extra_chars, max_chars = compute_length_limits(s, next_s)

    # 혹시 너무 작은 값일 때를 대비해 최소/최대 클램핑
    if max_chars <= 0:
        max_chars = max(32, len(orig_analyst) + 10)
    max_chars = min((max_chars + 10), 400)  # 안전상 너무 긴 멘트 방지

    system_content = (
        f"경기: {game_title}\n"
        "당신은 박찬호 해설 위원입니다. 캐스터의 말을 입력 받아, 실제 중계처럼 자연스럽게, "
        "과거 LA, 메이저리그 경험을 적절히 섞어 설명합니다. "
        "스코어보드에 주어진 정보 이외의 구체적인 추가 사실(득점, 주자 이동 등)은 지어내지 말고, "
        "주어진 정보와 일반적인 경험, 느낌 위주로 해설해 주세요."
    )

    user_content = (
        "다음은 KBO 포스트시즌 야구 하이라이트의 한 장면입니다.\n\n"
        "[캐스터 멘트]\n"
        f"{s.caster_text}\n\n"
        "[현재 스코어보드 상황]\n"
        f"{sb_summary}\n\n"
        "[기존 해설 멘트 (내용만 참고용)]\n"
        f"{orig_analyst}\n\n"
        "위 정보를 참고해서, 캐스터 멘트 바로 뒤에 이어질 박찬호 해설 멘트를 한 번만 새로 작성해 주세요.\n\n"
        "요구 조건:\n"
        f"- 기존 해설 멘트의 정보와 의미는 최대한 유지하세요.\n"
        "- 실제 방송 중계처럼 자연스럽고, 박찬호 해설 위원의 말투와 표현 습관을 살려 주세요.\n"
        f"- 길이는 대략 {orig_len}자를 넘기도록 하되, 다음 세트까지 여유 시간을 고려해도 {max_chars}자를 넘기지 마세요.\n"
        "- 문장은 중간에 끊기지 말고, 하나의 발화(한 번의 멘트)로 자연스럽게 끝나게 써 주세요.\n"
        "- ...으로 문장을 중간에 끊지 마세요.\n"
        "- 불필요한 반복이나 군더더기 표현은 줄이세요.\n"
        "- 출력은 해설 멘트 문장만 작성하고, 따옴표나 설명 문구는 넣지 마세요.\n"
        "- 신성진, 심삼진 같은 문자는 스윙 삼진 이에요.\n"
        "- stt가 정확하지 못한 것을 인지하고 이상한 문자가 있으면 비슷한 야구 용어 발음으로 해석하세요.\n"
        "- 기존 해설이 상황을 중계하고 있으면 해당 텍스트가 나타난 텍스트의 순서에 같이 상황을 출력해야 한다. 예를 들면 스윙, 삼진, 안타, 홈으로, 볼넷, 땅볼, 아웃, 선취점, 홈런 등\n"
    )

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]
    return messages, max_chars


def generate_commentary_for_set(
    s: SetItem,
    next_s: Optional[SetItem],
    model,
    tokenizer,
    game_title: str,
    temperature: float = 0.7,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
    no_repeat_ngram_size: int = 3,
    base_max_new_tokens: int = 512,
) -> str:
    """
    analyst_text가 있는 세트에 대해, LLM으로 새 analyst_text를 생성.
    """
    messages, max_chars = build_messages_for_set(s, next_s, game_title=game_title)

    # chat 템플릿 → 프롬프트
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # 문자 수 제한에 따라 max_new_tokens 대략 조정
    max_new_tokens = min(
        base_max_new_tokens,
        max(32, int(max_chars * 1.2)),
    )

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    out_ids = generated[0, inputs["input_ids"].shape[1] :]
    out_text = tokenizer.decode(out_ids, skip_special_tokens=True)
    out_text = out_text.strip()

    # 길이 최종 체크 & 트리밍
    out_text = trim_to_max_chars(out_text, max_chars=max_chars)
    return out_text


# =========================================
# 6. 메인 파이프라인
# =========================================

def load_sets_from_json(path: Path) -> List[SetItem]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    sets: List[SetItem] = []
    for row in data:
        sets.append(
            SetItem(
                set_id=row["set_id"],
                video_id=row["video_id"],
                set_start_sec=float(row["set_start_sec"]),
                set_end_sec=float(row["set_end_sec"]),
                caster_text=row["caster_text"],
                analyst_text=row.get("analyst_text"),
                scoreboard=row.get("scoreboard") or {},
            )
        )
    return sets


def generate_analyst_for_all_sets(
    json_in_path: Path,
    json_out_path: Path,
    model,
    tokenizer,
    game_title: str,
    temperature: float = 0.7,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
    no_repeat_ngram_size: int = 3,
    base_max_new_tokens: int = 512,
) -> List[Dict[str, Any]]:
    """
    - 입력: scoreboard까지 붙어있는 최종 STT 세트 JSON
    - 출력: set_id, video_id, set_start_sec, set_end_sec, caster_text, analyst_text 만 가진 JSON

    규칙:
    - analyst_text == null 이면 LLM 호출 없이 그대로 null 유지
    - analyst_text != null 이면 LLM으로 재작성하여 교체
    """
    json_in_path = Path(json_in_path)
    json_out_path = Path(json_out_path)

    sets = load_sets_from_json(json_in_path)
    print(f"[INFO] 세트 개수: {len(sets)}")

    result: List[Dict[str, Any]] = []

    for i, s in enumerate(sets):
        next_s = sets[i + 1] if i + 1 < len(sets) else None

        if not s.analyst_text:  # null or 빈 문자열
            # 그대로 통과 (analyst_text는 None 또는 "" 유지)
            new_analyst = None
            print(f"[SKIP] {s.set_id}: analyst_text 없음 → LLM 미호출")
        else:
            print(f"[LLM] {s.set_id}: analyst_text 재생성 중...")
            new_analyst = generate_commentary_for_set(
                s,
                next_s,
                model=model,
                tokenizer=tokenizer,
                game_title=game_title,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                base_max_new_tokens=base_max_new_tokens,
            )

        result.append(
            {
                "set_id": s.set_id,
                "video_id": s.video_id,
                "set_start_sec": s.set_start_sec,
                "set_end_sec": s.set_end_sec,
                "caster_text": s.caster_text,
                "analyst_text": new_analyst,
            }
        )

    json_out_path.parent.mkdir(parents=True, exist_ok=True)
    with json_out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"[DONE] Pakchanho analyst_text 생성 완료 → {json_out_path.resolve()}")
    return result
