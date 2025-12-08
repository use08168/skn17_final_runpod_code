from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any, Optional

import cv2
import torch
from transformers import (
    Qwen3VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
)

# =========================================================
# 0. 기본 설정
# =========================================================

MODEL_ID = "SeHee8546/qwen3-vl-8b-kbo-scoreboard-qlora-final-V2"

SYSTEM_PROMPT = """당신은 야구 중계 화면 분석 전문 AI입니다.
제공된 야구 중계 이미지에서 '스코어보드(Scoreboard)' 영역을 찾아 현재 경기 상황을 정확한 JSON 형식으로 추출하십시오.
1. 스코어보드에서 다음 정보를 읽어 JSON 형태로 출력한다.
2. 스코어보드 위치: 주로 화면의 좌측 상단, 우측 하단, 좌측 하단에 위치한 직사각형 모양의 그래픽 오버레이입니다.
3. 팀 및 점수 배치: 원정팀(Away)은 왼쪽 혹은 위, 홈팀(Home)은 오른쪽 혹은 아래에 위치하고 점수는 팀 이름과 가까운 곳에 위치한다.
4. 이닝(Inning): 숫자와 함께 화살표(▲/△: 초, ▼/▽: 말) 혹은 텍스트(초/말)로 표시됩니다.
5. 주자(Bases): 다이아몬드(◇) 형태의 그래픽 3개로 오른쪽부터 반시계 방향으로 1, 2, 3루이다. 오른쪽 다이아몬드(◇)는 1루, 중앙 다이아몬드(◇)는 2루, 왼쪽는 다이아몬드(◇) 3루를 나타냅니다. 색상(주로 노란색)이 채워져 있는 루(base)가 주자가 있는 루입니다. 
주자 예시 (대부분 노란색으로 표시)
- ◇◇◇ {1루: false, 2루: false, 3루: false}
- ◇◇◆ {1루: true, 2루: false, 3루: false}
- ◇◆◆ {1루: true, 2루: true, 3루: false}
- ◆◆◆ {1루: true, 2루: true, 3루: true}
- ◇◆◇ {1루: false, 2루: true, 3루: false}
- ◆◆◇ {1루: false, 2루: true, 3루: true}
- ◆◇◆ {1루: true, 2루: false, 3루: true}
- ◆◇◇ {1루: false, 2루: false, 3루: true}
6. 볼카운트(B-S-O): B(Ball), S(Strike)은 숫자-숫자 또는 동그라미 모양(B는 ○ 3개, S는 ○ 2개)으로 카운트가 올라갈 때 B는 초록색(green) S는 노란색(yellow)으로 동그라미에 색이 하나씩 채워지고, O(out)은 일반적으로 동그라미 모양(○) 2개로 나타내고 카운트가 올라갈 때 빨간색(red)으로 색을 채운다.
볼카운트 예시
- 볼 : 대부분 초록색으로 ○○○ (0볼) ●○○ (1볼), ●●○ (2볼), ●●● (3 볼)
- 스트라이크 : 대부분 노란색으로 ○○ (0 스트라이크), ●○ (1 스트라이크), ●● (2 스트라이크)
- 아웃 : 대부분 빨간색으로○○ (0 아웃), ●○ (1 아웃), ●● (2 아웃), 숫자-숫자로 표시된 경우 앞에 있는 숫자는 볼, 뒤에 있는 숫자는 스트라이크.
7. 투수(P)와 타자(B) 정보는 스코어보드 상단 또는 하단 또는 측면에 작은 텍스트로 부착되어 있다.
8. 투구 수는 일반적으로 작은 글씨로 P 옆 또는 야구공 이미지 옆에 숫자로 표시되어있다. km/h로 투구 속도만 나타난 경우 null을 출력한다.
9. 타자 타순은 일반적으로 타자 이름 왼쪽에 숫자로 표시된다.
10. 타자 경기 기록은 일반적으로 타자이름 오른쪽에 1/2 또는 2타수 1안타의 형식으로 나타난다.(1/2의 경우 2타수 1안타로 출력) 또한 타율(소수점 세 째 자리)가 표시되는 경우가 있다. 타율은 출력하지 않는다. 타율만 있다면 null을 출력하라.
11. 온전한 json 형태로
- 원정팀
- 홈팀
- 원정팀 점수
- 홈팀 점수
- 이닝
- 이닝 상황 ("초" or "말")
- 볼 (0~3)
- 스트라이크 (0~2)
- 아웃 (0~2)
- 주자: { "1루": true/false, "2루": true/false, "3루": true/false }
- 투수 이름
- 투구 수
- 타자 이름
- 타자 타순
- 타자 경기 기록
JSON만 출력하고 다른 설명은 쓰지 마라.
정보를 읽을 수 없는 항목은 null로 출력한다.
"""

USER_PROMPT = "이 사진은 야구 하이라이트의 한 장면이다. 사진을 보고 스코어보드가 있으면 스코어보드로 현재 상황을 분석하라. 스코어 보드가 없다면 json에 null을 채워 출력하라."

SCOREBOARD_KEYS = [
    "원정팀",
    "홈팀",
    "원정팀 점수",
    "홈팀 점수",
    "이닝",
    "이닝 상황",
    "볼",
    "스트라이크",
    "아웃",
    "주자",
    "투수 이름",
    "투구 수",
    "타자 이름",
    "타자 타순",
    "타자 경기 기록",
]


# =========================================================
# 1. scoreboard 유틸
# =========================================================

def build_null_scoreboard() -> Dict[str, Any]:
    sb = {k: None for k in SCOREBOARD_KEYS}
    sb["주자"] = None
    return sb


def normalize_scoreboard_dict(raw: Any) -> Dict[str, Any]:
    """
    모델이 출력한 dict를 우리가 원하는 형태로 정규화:
    - 모든 키가 존재하도록 채우고
    - '주자'는 dict 또는 None 으로 정리
    """
    sb = build_null_scoreboard()
    if not isinstance(raw, dict):
        return sb

    for k in SCOREBOARD_KEYS:
        if k not in raw:
            continue
        if k == "주자":
            val = raw[k]
            if val is None:
                sb["주자"] = None
            elif isinstance(val, dict):
                bases = {"1루": None, "2루": None, "3루": None}
                for base in bases.keys():
                    if base in val:
                        bases[base] = val[base]
                sb["주자"] = bases
            else:
                sb["주자"] = None
        else:
            sb[k] = raw[k]
    return sb


def is_all_null_scoreboard(sb: Dict[str, Any]) -> bool:
    """
    scoreboard의 모든 필드가 null인지 체크.
    (주자가 dict인 경우 안의 값도 모두 None이면 null 취급)
    """
    if sb is None:
        return True

    for k, v in sb.items():
        if k == "주자":
            if v is None:
                continue
            if isinstance(v, dict):
                # 하나라도 값이 있으면 null 아님
                if any(val is not None for val in v.values()):
                    return False
            else:
                # dict도 None도 아닌데 값이 있으면 null 아님
                return False
        else:
            if v is not None:
                return False
    return True


def extract_json_from_text(text: str) -> Dict[str, Any]:
    """
    모델 출력 텍스트에서 JSON 부분만 안전하게 파싱.
    문제가 있으면 null scoreboard 반환.
    """
    text = text.strip()
    # 1차 시도: 전체를 JSON으로 파싱
    try:
        return json.loads(text)
    except Exception:
        pass

    # 2차 시도: 가장 바깥 {} 구간만 잘라서 시도
    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last != -1 and last > first:
        candidate = text[first:last+1]
        try:
            return json.loads(candidate)
        except Exception:
            pass

    return build_null_scoreboard()


# =========================================================
# 2. Qwen3-VL 모델 / 프로세서 로드
# =========================================================

def load_scoreboard_model_and_processor(
    model_id: str = MODEL_ID,
    min_pixels: int = 256 * 28 * 28,
    max_pixels: int = 1920 * 1080,
):
    """
    4bit QLoRA 양자화된 Qwen3-VL 기반 scoreboard 모델 로드.
    (파인튜닝 때 사용한 설정을 최대한 맞춤)
    """
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        if major >= 8:
            compute_dtype = torch.bfloat16
        else:
            compute_dtype = torch.float16
    else:
        compute_dtype = torch.float32

    print(f"compute_dtype = {compute_dtype}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
    )

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        dtype=compute_dtype,
        device_map="auto",
        attn_implementation="sdpa",
    )

    processor = AutoProcessor.from_pretrained(
        model_id,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )

    # 패딩 설정
    if hasattr(processor, "tokenizer"):
        tokenizer = processor.tokenizer
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

    # inference이므로 gradient checkpointing / require_grads는 사용 X
    if hasattr(model, "config"):
        model.config.use_cache = True

    print("모델 / 프로세서 로드 완료")
    return model, processor


# =========================================================
# 3. 단일 이미지 → scoreboard 추론
# =========================================================

def run_vlm_on_image(
    model,
    processor,
    image_path: Path,
    max_new_tokens: int = 256,
) -> Dict[str, Any]:
    """
    하나의 이미지에 대해 scoreboard JSON을 추출한다.
    """
    image_path = Path(image_path)
    if not image_path.exists():
        print(f"[WARN] 이미지 없음: {image_path}")
        return build_null_scoreboard()

    # Qwen3-VL 추천 포맷: messages + apply_chat_template
    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": SYSTEM_PROMPT},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": USER_PROMPT},
            ],
        },
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,   # scoreboard는 deterministic하게 뽑는 것이 좋음
            temperature=0.0,
        )

    # 프롬프트 부분 제거
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_texts = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    output_text = output_texts[0] if output_texts else ""

    raw = extract_json_from_text(output_text)
    sb = normalize_scoreboard_dict(raw)
    return sb


# =========================================================
# 4. 비디오에서 프레임 캡처 유틸
# =========================================================

def _open_video(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"비디오를 열 수 없습니다: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0:
        cap.release()
        raise RuntimeError(f"FPS를 가져올 수 없습니다. fps={fps}")

    duration = frame_count / fps
    print(f"[INFO] 파일: {video_path}")
    print(f"[INFO] FPS = {fps}, 총 프레임 수 = {frame_count}, 길이 ≈ {duration:.3f}초")

    return cap, fps, frame_count, duration


def _capture_frame_to_file(
    cap,
    fps: float,
    frame_count: int,
    timestamp_sec: float,
    out_path: Path,
) -> bool:
    """
    이미 열려 있는 cap을 사용하여 특정 초 위치의 프레임을 캡처해
    out_path에 저장한다.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ts = float(timestamp_sec)
    if ts < 0:
        ts = 0.0

    frame_idx = int(round(ts * fps))
    frame_idx = max(0, min(frame_idx, frame_count - 1))

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    print(f"[DEBUG] capture ts={ts:.3f}s, frame_idx={frame_idx}, ret={ret}")

    if not ret or frame is None:
        print(f"[WARN] {ts:.3f}s 근처 프레임을 읽지 못했습니다.")
        return False

    saved = cv2.imwrite(str(out_path), frame)
    if saved:
        print(f"[INFO] 프레임 저장: {out_path}")
        return True
    else:
        print(f"[ERROR] 프레임 저장 실패: {out_path}")
        return False


# =========================================================
# 5. 메인 파이프라인:
#    세트 JSON에 scoreboard 키 추가
# =========================================================

def attach_scoreboard_to_sets(
    json_after_split_path: Path,
    output_json_path: Path,
    frames_root: Path,
    video_path: Path,
    model,
    processor,
    retry_if_all_null: bool = True,
    retry_offset_sec: float = 2.0,
) -> List[Dict[str, Any]]:
    """
    - 세트 분리 후 JSON 파일을 로드
    - 각 세트의 set_id 기준 이미지를 FRAMES_ROOT에서 찾음
      (없으면 비디오에서 캡처)
    - VLM 통과 결과를 scoreboard 키로 추가
    - 결과가 all-null이면 set_start_sec + retry_offset_sec 에서 한 번 더 캡처+추론
    - 최종 결과를 output_json_path에 저장

    ➜ analyst_text가 null/빈 문자열인 세트는 VLM 추론도 건너뛰고,
       scoreboard는 all-null로 채운다.
    """
    json_after_split_path = Path(json_after_split_path)
    output_json_path = Path(output_json_path)
    frames_root = Path(frames_root)
    video_path = Path(video_path)

    frames_root.mkdir(parents=True, exist_ok=True)

    # 세트 로드
    with json_after_split_path.open("r", encoding="utf-8") as f:
        sets: List[Dict[str, Any]] = json.load(f)

    print(f"[INFO] 세트 개수: {len(sets)}")

    # 비디오 오픈
    cap, fps, frame_count, duration = _open_video(video_path)

    try:
        for s in sets:
            set_id = s.get("set_id")
            set_start_sec = float(s.get("set_start_sec", 0.0))

            # ★ analyst_text가 null/빈 문자열이면 스코어보드도 건너뛰기 ★
            analyst = s.get("analyst_text", None)
            if analyst is None or (isinstance(analyst, str) and not analyst.strip()):
                print(f"[SKIP] set_id={set_id}: analyst_text가 없으므로 VLM 스코어보드 추출 건너뜀")
                s["scoreboard"] = build_null_scoreboard()
                continue

            print(f"\n[SET] set_id={set_id}, set_start_sec={set_start_sec:.3f}")

            # 1) 1차 이미지 경로 (set_start_sec 시점)
            img_path = frames_root / f"{set_id}.jpg"

            # 이미지가 없다면 비디오에서 캡처
            if not img_path.exists():
                print(f"[INFO] 1차 이미지 없음, 비디오에서 캡처: {img_path}")
                ok = _capture_frame_to_file(cap, fps, frame_count, set_start_sec, img_path)
                if not ok:
                    # 프레임을 못 뽑으면 scoreboard는 null로
                    s["scoreboard"] = build_null_scoreboard()
                    continue

            # 2) 1차 VLM 추론
            sb1 = run_vlm_on_image(model, processor, img_path)
            print(f"[INFO] 1차 scoreboard: {sb1}")

            # 3) all-null 체크
            if retry_if_all_null and is_all_null_scoreboard(sb1):
                print("[INFO] 1차 결과가 all-null → +2초 위치 재시도")
                retry_ts = set_start_sec + retry_offset_sec
                # 영상 길이 밖으로 나가지 않게 클램핑
                if duration > 0:
                    retry_ts = min(retry_ts, max(duration - 0.1, 0.0))

                retry_img_path = frames_root / f"{set_id}_retry.jpg"
                ok2 = _capture_frame_to_file(cap, fps, frame_count, retry_ts, retry_img_path)
                if ok2:
                    sb2 = run_vlm_on_image(model, processor, retry_img_path)
                    print(f"[INFO] 2차 scoreboard: {sb2}")
                    # 2차 결과는 all-null이든 말든 무조건 사용
                    s["scoreboard"] = sb2
                else:
                    # 2차 캡처도 실패하면 1차 결과라도 유지
                    print("[WARN] 2차 캡처 실패 → 1차 결과 유지")
                    s["scoreboard"] = sb1
            else:
                # 1차 결과 사용
                s["scoreboard"] = sb1

    finally:
        cap.release()
        print("[INFO] 비디오 리소스 해제")

    # 결과 저장
    with output_json_path.open("w", encoding="utf-8") as f:
        json.dump(sets, f, ensure_ascii=False, indent=2)

    print(f"[DONE] scoreboard 추가 완료 → {output_json_path.resolve()}")
    return sets