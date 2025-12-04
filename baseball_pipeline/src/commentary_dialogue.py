# src/commentary_dialogue.py

from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import timedelta
import re

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


# =========================
#  모델 로더
# =========================

BASE_MODEL_ID = "kakaocorp/kanana-1.5-8b-instruct-2505"
LORA_MODEL_ID = "SeHee8546/kanana-1.5-8b-pakchanho-lora"

_TOKENIZER: Optional[AutoTokenizer] = None
_MODEL: Optional[torch.nn.Module] = None


def get_commentary_model():
    """박찬호 모델 로드"""
    global _TOKENIZER, _MODEL
    if _MODEL is not None and _TOKENIZER is not None:
        return _TOKENIZER, _MODEL

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[DIALOGUE] loading base model on {device} ...")

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

    print("[DIALOGUE] loading LoRA:", LORA_MODEL_ID)
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
    print("[DIALOGUE] model ready.")
    return _TOKENIZER, _MODEL


# =========================
#  경기 상황 분석
# =========================

@dataclass
class GameMoment:
    """특정 시점의 경기 상황"""
    timestamp: float
    utter_id: int
    duration: float
    
    # 스코어보드 정보
    away_team: Optional[str]
    home_team: Optional[str]
    away_score: Optional[int]
    home_score: Optional[int]
    inning: Optional[int]
    inning_state: Optional[str]
    balls: Optional[int]
    strikes: Optional[int]
    outs: Optional[int]
    runner_1b: Optional[bool]
    runner_2b: Optional[bool]
    runner_3b: Optional[bool]
    pitcher: Optional[str]
    pitch_count: Optional[int]
    batter: Optional[str]
    batting_order: Optional[int]
    batting_record: Optional[str]
    
    # 원본 발화
    original_text: Optional[str]
    speaker_role: Optional[str]
    
    def __post_init__(self):
        """시간 형식 변환"""
        self.time_str = str(timedelta(seconds=int(self.timestamp))).split('.')[0]
    
    def format_scoreboard(self) -> str:
        """스코어보드를 텍스트로"""
        parts = []
        
        if self.away_team and self.home_team:
            score = f"{self.away_team}"
            if self.away_score is not None:
                score += f" {self.away_score}"
            score += f" vs {self.home_team}"
            if self.home_score is not None:
                score += f" {self.home_score}"
            parts.append(score)
        
        if self.inning is not None:
            inning_text = f"{self.inning}회"
            if self.inning_state:
                inning_text += f" {self.inning_state}"
            parts.append(inning_text)
        
        count_parts = []
        if self.balls is not None:
            count_parts.append(f"{self.balls}B")
        if self.strikes is not None:
            count_parts.append(f"{self.strikes}S")
        if self.outs is not None:
            count_parts.append(f"{self.outs}O")
        if count_parts:
            parts.append(" ".join(count_parts))
        
        runner_parts = []
        if self.runner_1b:
            runner_parts.append("1루")
        if self.runner_2b:
            runner_parts.append("2루")
        if self.runner_3b:
            runner_parts.append("3루")
        if runner_parts:
            parts.append(f"{', '.join(runner_parts)} 주자")
        
        if self.pitcher:
            pitcher_text = f"투수 {self.pitcher}"
            if self.pitch_count:
                pitcher_text += f"({self.pitch_count}구)"
            parts.append(pitcher_text)
        
        if self.batter:
            batter_text = f"타자 {self.batter}"
            if self.batting_order:
                batter_text += f"({self.batting_order}번)"
            parts.append(batter_text)
        
        return " | ".join(parts) if parts else ""
    
    def has_meaningful_scoreboard(self) -> bool:
        """의미있는 스코어보드 데이터가 있는지"""
        info_count = sum([
            self.inning is not None,
            self.pitcher is not None,
            self.batter is not None,
            self.runner_1b or self.runner_2b or self.runner_3b,
            self.balls is not None or self.strikes is not None,
        ])
        return info_count >= 2
    
    def should_analyst_speak(self) -> bool:
        """해설위원이 말해야 할 상황인지 판단"""
        # 1. 원본이 analyst 발화였으면 해설위원이 말한다
        if self.speaker_role == "analyst":
            return True
        
        # 2. 구간이 충분히 길고 (5초+) 스코어보드 정보가 있으면 해설위원이 말한다
        if self.duration >= 5.0 and self.has_meaningful_scoreboard():
            return True
        
        # 3. 구간이 매우 길면 (8초+) 무조건 말한다
        if self.duration >= 8.0:
            return True
        
        return False
    
    def get_analyst_token_budget(self) -> int:
        """해설위원이 사용할 수 있는 토큰 수 (구간 길이 기반)"""
        # 캐스터가 원본 텍스트 말하는데 걸리는 시간 추정
        caster_chars = len(self.original_text) if self.original_text else 20
        caster_time = caster_chars / 4.5  # 1초당 4.5글자
        
        # 남은 시간을 해설위원이 사용
        available_time = max(0, self.duration - caster_time)
        tokens = int(available_time * 18)  # 1초당 18토큰
        return max(50, min(tokens, 350))  # 최소 50, 최대 350


def detect_game_events(moments: List[GameMoment]) -> List[Tuple[int, str, str]]:
    """경기 이벤트 감지 (득점, 이닝 변화 등)"""
    events = []
    
    for i in range(1, len(moments)):
        prev = moments[i-1]
        curr = moments[i]
        
        # 득점 이벤트
        if (prev.away_score is not None and curr.away_score is not None and
            prev.away_score != curr.away_score):
            diff = curr.away_score - prev.away_score
            events.append((i, "score_change", f"{curr.away_team} {diff}점 득점"))
        
        if (prev.home_score is not None and curr.home_score is not None and
            prev.home_score != curr.home_score):
            diff = curr.home_score - prev.home_score
            events.append((i, "score_change", f"{curr.home_team} {diff}점 득점"))
        
        # 이닝 변화
        if (prev.inning is not None and curr.inning is not None and
            prev.inning != curr.inning):
            events.append((i, "inning_change", f"{curr.inning}회 {curr.inning_state or ''}"))
        
        # 주자 변화
        prev_runners = sum([prev.runner_1b or False, prev.runner_2b or False, prev.runner_3b or False])
        curr_runners = sum([curr.runner_1b or False, curr.runner_2b or False, curr.runner_3b or False])
        if curr_runners > prev_runners:
            events.append((i, "scoring_chance", "득점 찬스"))
        
        # 투수 교체
        if (prev.pitcher and curr.pitcher and prev.pitcher != curr.pitcher):
            events.append((i, "pitcher_change", f"투수 교체: {curr.pitcher}"))
    
    return events


# =========================
#  발화 생성
# =========================

def get_caster_line(moment: GameMoment) -> Tuple[str, float]:
    """캐스터 멘트: 원본 텍스트 그대로 사용"""
    if moment.original_text and moment.original_text.strip():
        text = moment.original_text.strip()
        duration = len(text) / 4.5  # 1초당 4.5글자
        return text, duration
    
    # 기본 멘트
    return "경기가 진행됩니다.", 2.0


def generate_analyst_commentary(
    caster_line: str,
    moment: GameMoment,
    conversation_history: List[Dict[str, str]],
    event_type: Optional[str] = None,
    event_desc: Optional[str] = None,
    game_context: Optional[str] = None,
) -> str:
    """해설위원(박찬호) 멘트 생성 - VLM 스코어보드 상황 기반"""
    
    tokenizer, model = get_commentary_model()
    
    # 토큰 예산
    max_tokens = moment.get_analyst_token_budget()
    
    # 시스템 프롬프트
    system_prompt = (
        "당신은 프로야구 해설위원입니다. "
        "캐스터가 상황을 말한 직후, 전문적인 분석과 해설을 제공합니다. "
        "박찬호 해설위원의 차분하고 전문적인 말투로 이야기하세요. "
        "~같아요, ~하죠, ~네요 같은 자연스러운 종결어미를 사용하세요. "
        "절대로 자신의 이름(박찬호)이나 1인칭 표현('저', '나')을 사용하지 마세요."
    )
    
    # 컨텍스트 구성
    context_parts = []
    
    if game_context:
        context_parts.append(f"경기: {game_context}")
    
    scoreboard = moment.format_scoreboard()
    if scoreboard:
        context_parts.append(f"현재 상황: {scoreboard}")
    
    # 이벤트 유형에 따른 지시
    if event_type == "score_change":
        context_parts.append(
            "득점 상황입니다. "
            "이 득점의 배경, 투수와 타자의 대결 과정, "
            "전략적 의미를 상세히 분석하세요."
        )
    elif event_type == "inning_change":
        context_parts.append(
            "이닝이 바뀌었습니다. "
            "지금까지의 경기 흐름과 앞으로의 전개를 예측하세요."
        )
    elif event_type == "pitcher_change":
        context_parts.append(
            "투수가 교체되었습니다. "
            "교체 배경, 새 투수의 특징, 예상되는 투구 전략을 설명하세요."
        )
    elif event_type == "scoring_chance":
        context_parts.append(
            "득점 찬스입니다. "
            "주자 상황에서의 공격 전략, 수비의 대응, 타자의 역할을 분석하세요."
        )
    else:
        # 일반 상황
        if moment.duration >= 10:
            context_parts.append(
                "시간이 충분하니 현재 상황을 여러 각도에서 분석하세요. "
                "투수와 타자의 특징, 주자 상황, 팀 전략 등을 "
                "여러 문장에 걸쳐 상세히 설명하세요."
            )
        elif moment.duration >= 7:
            context_parts.append(
                "현재 상황을 전문적으로 분석하세요. "
                "투수와 타자의 특징, 경기 상황의 중요성을 설명하세요."
            )
        else:
            context_parts.append(
                "현재 상황에 대한 간단한 전문가 의견을 말하세요."
            )
    
    context_text = "\n".join(context_parts)
    
    # 대화 이력 (최근 12턴)
    messages = [{"role": "system", "content": system_prompt}]
    
    recent_history = conversation_history[-12:] if len(conversation_history) > 12 else conversation_history
    for msg in recent_history:
        messages.append(msg)
    
    # 현재 캐스터 멘트
    user_content = (
        f"{context_text}\n\n"
        f"캐스터: \"{caster_line}\"\n\n"
        f"위 상황에서 박찬호 해설위원답게 전문적인 해설을 제공하세요. "
        f"약 {moment.duration:.1f}초 분량입니다."
    )
    messages.append({"role": "user", "content": user_content})
    
    # 생성
    try:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        inputs = tokenizer(prompt, return_tensors="pt")
        device = model.device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        pad_id = tokenizer.eos_token_id or tokenizer.pad_token_id or 0
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.75,
                top_p=0.92,
                repetition_penalty=1.2,
                pad_token_id=pad_id,
            )
        
        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        
        # 메모리 정리
        del inputs, outputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 후처리
        text = _postprocess_analyst_text(text)
        
        return text
    
    except Exception as e:
        print(f"[DIALOGUE] ERROR 해설 생성 실패: {e}")
        import traceback
        traceback.print_exc()
        return ""


def _postprocess_analyst_text(text: str) -> str:
    """해설 텍스트 후처리"""
    if not text:
        return ""
    
    # 반복 제거
    text = re.sub(r'\b(\w+)(?:,\s*\1)+\b', r'\1', text)
    text = re.sub(r'\b(\w+)(?:\s+\1){2,}\b', r'\1', text)
    
    # 자기 지칭 제거
    text = re.sub(r'박찬호\s*(?:입니다|위원|해설위원)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'(?:^|\.\s+)(?:저는|제가|나는)\s+', ' ', text)
    
    # 불필요한 메타 표현 제거
    bad_phrases = ["작성하세요", "말하세요", "설명하세요", "분석하세요", "해설을", "코멘트", "의견을"]
    for phrase in bad_phrases:
        text = text.replace(phrase, '')
    
    # 공백 정리
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'\s+([,.!?])', r'\1', text)
    
    return text


# =========================
#  메인 파이프라인
# =========================

def create_dialogue_commentary(
    vlm_csv_path: Path | str,
    output_csv_path: Path | str,
    game_desc: Optional[str] = None,
    min_duration_for_analyst: float = 5.0,
    max_moments: Optional[int] = None,
) -> Path:
    """
    VLM 스코어보드 CSV에서 캐스터-해설 대화 생성
    
    Args:
        vlm_csv_path: VLM 스코어보드 CSV
        output_csv_path: 출력 CSV
        game_desc: 경기 설명
        min_duration_for_analyst: 해설위원이 말하기 위한 최소 구간 길이
        max_moments: 최대 처리 개수 (테스트용)
    """
    vlm_csv_path = Path(vlm_csv_path)
    output_csv_path = Path(output_csv_path)
    
    print("="*80)
    print("[DIALOGUE] 캐스터-해설 대화 생성")
    print("="*80)
    print(f"입력: {vlm_csv_path}")
    print(f"출력: {output_csv_path}")
    print(f"해설 최소 구간: {min_duration_for_analyst}초")
    print("="*80)
    
    # CSV 로드
    df = pd.read_csv(vlm_csv_path)
    df = df.sort_values('start_sec').reset_index(drop=True)
    
    # GameMoment 객체 생성
    moments: List[GameMoment] = []
    for _, row in df.iterrows():
        duration = row.get('duration', 0)
        if duration <= 0:
            duration = 3.0
        
        moment = GameMoment(
            timestamp=row.get('start_sec', 0),
            utter_id=row.get('utter_id', 0),
            duration=duration,
            away_team=row.get('sb_원정팀'),
            home_team=row.get('sb_홈팀'),
            away_score=row.get('sb_원정팀 점수'),
            home_score=row.get('sb_홈팀 점수'),
            inning=row.get('sb_이닝'),
            inning_state=row.get('sb_이닝 상황'),
            balls=row.get('sb_볼'),
            strikes=row.get('sb_스트라이크'),
            outs=row.get('sb_아웃'),
            runner_1b=row.get('sb_주자_1루'),
            runner_2b=row.get('sb_주자_2루'),
            runner_3b=row.get('sb_주자_3루'),
            pitcher=row.get('sb_투수 이름'),
            pitch_count=row.get('sb_투구 수'),
            batter=row.get('sb_타자 이름'),
            batting_order=row.get('sb_타자 타순'),
            batting_record=row.get('sb_타자 경기 기록'),
            original_text=row.get('text'),
            speaker_role=row.get('role'),
        )
        moments.append(moment)
    
    if max_moments:
        moments = moments[:max_moments]
    
    print(f"[DIALOGUE] 총 {len(moments)}개 moment 로드")
    
    # 이벤트 감지
    events = detect_game_events(moments)
    event_dict = {idx: (etype, edesc) for idx, etype, edesc in events}
    print(f"[DIALOGUE] {len(events)}개 주요 이벤트 감지")
    
    # 모델 로드
    _ = get_commentary_model()
    
    # 대화 생성
    dialogues = []
    conversation_history = []
    
    total_processed = 0
    analyst_spoke_count = 0
    
    for idx, moment in enumerate(moments):
        event_info = event_dict.get(idx, (None, None))
        event_type, event_desc = event_info
        
        # 해설위원이 말할지 결정
        analyst_speaks = (
            moment.duration >= min_duration_for_analyst or
            moment.should_analyst_speak() or
            event_type is not None
        )
        
        total_processed += 1
        
        print(f"\n[{total_processed}/{len(moments)}] t={moment.time_str} (dur={moment.duration:.1f}s, utter_id={moment.utter_id})")
        
        # 캐스터 멘트
        caster_line, caster_duration = get_caster_line(moment)
        print(f"  캐스터 ({caster_duration:.1f}s): {caster_line[:70]}...")
        
        # 해설위원 멘트
        analyst_line = ""
        analyst_duration = 0.0
        
        if analyst_speaks:
            analyst_spoke_count += 1
            analyst_duration = moment.duration - caster_duration
            
            analyst_line = generate_analyst_commentary(
                caster_line=caster_line,
                moment=moment,
                conversation_history=conversation_history,
                event_type=event_type,
                event_desc=event_desc,
                game_context=game_desc,
            )
            
            if analyst_line:
                print(f"  해설   ({analyst_duration:.1f}s): {analyst_line[:70]}...")
            else:
                print(f"  해설   (생성 실패)")
        else:
            print(f"  (해설 없음 - 구간이 짧음)")
        
        # 대화 이력 업데이트
        conversation_history.append({"role": "user", "content": f"캐스터: {caster_line}"})
        if analyst_line:
            conversation_history.append({"role": "assistant", "content": analyst_line})
        
        # 결과 저장
        dialogues.append({
            "timestamp": moment.timestamp,
            "time_str": moment.time_str,
            "utter_id": moment.utter_id,
            "duration": moment.duration,
            "event_type": event_type or "",
            "event_desc": event_desc or "",
            "scoreboard": moment.format_scoreboard(),
            "caster": caster_line,
            "caster_duration": caster_duration,
            "analyst": analyst_line,
            "analyst_duration": analyst_duration if analyst_line else 0,
            "original_text": moment.original_text or "",
            "original_role": moment.speaker_role or "",
        })
        
        # 10개마다 중간 저장
        if total_processed % 10 == 0:
            temp_df = pd.DataFrame(dialogues)
            temp_path = output_csv_path.with_name(output_csv_path.stem + ".tmp.csv")
            temp_df.to_csv(temp_path, index=False, encoding="utf-8-sig")
            print(f"  → 중간 저장: {total_processed}개 처리")
    
    # 최종 저장
    result_df = pd.DataFrame(dialogues)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")
    
    print("\n" + "="*80)
    print("[DIALOGUE] 완료!")
    print("="*80)
    print(f"  - 총 구간: {total_processed}개")
    print(f"  - 해설위원 발화: {analyst_spoke_count}개")
    print(f"  - 캐스터만: {total_processed - analyst_spoke_count}개")
    print(f"  - 저장 위치: {output_csv_path}")
    print("="*80)
    
    return output_csv_path
