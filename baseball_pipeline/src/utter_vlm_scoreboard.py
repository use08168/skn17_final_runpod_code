# src/utter_vlm_scoreboard.py

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import cv2
import pandas as pd
import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


# ========================================
# 로깅 설정
# ========================================

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# ========================================
# VLM 모델 로더
# ========================================

_VLM_MODEL: Optional[Qwen2VLForConditionalGeneration] = None
_VLM_PROCESSOR: Optional[AutoProcessor] = None


def get_vlm_model():
    """VLM 모델 싱글톤 로더"""
    global _VLM_MODEL, _VLM_PROCESSOR
    
    if _VLM_MODEL is not None and _VLM_PROCESSOR is not None:
        return _VLM_MODEL, _VLM_PROCESSOR
    
    model_name = "Qwen/Qwen2-VL-7B-Instruct"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    logger.info(f"VLM 모델 로딩: {model_name} on {device}")
    
    # 프로세서 로드
    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=True,
        min_pixels=256*28*28,
        max_pixels=1024*28*28,
    )
    
    # 모델 로드
    if device == "cuda":
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )
    
    model.eval()
    
    _VLM_MODEL = model
    _VLM_PROCESSOR = processor
    
    logger.info("VLM 모델 로딩 완료")
    return _VLM_MODEL, _VLM_PROCESSOR


# ========================================
# 프레임 캡처 (최적화 버전)
# ========================================

def capture_frames_optimized(
    video_path: Path,
    utterances_df: pd.DataFrame,
    frames_dir: Path,
    duration_threshold: float = 5.0,
) -> Dict[int, List[Path]]:
    """
    최적화된 프레임 캡처
    
    - 5초 미만: 중간 지점 1장만
    - 5초 이상: 기존 방식 (시작/중간/끝)
    
    Args:
        video_path: 비디오 경로
        utterances_df: utterances CSV 데이터
        frames_dir: 프레임 저장 디렉토리
        duration_threshold: 단일 프레임 캡처 기준 (초)
    
    Returns:
        {utter_id: [frame_path1, frame_path2, ...]}
    """
    frames_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"비디오 열기 실패: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    logger.info(f"비디오 정보: FPS={fps:.2f}, 총 프레임={total_frames_count}")
    
    frame_map = {}
    total_captured = 0
    
    for idx, row in utterances_df.iterrows():
        utter_id = row['utter_id']
        start_sec = row['start_sec']
        end_sec = row['end_sec']
        duration = end_sec - start_sec
        
        # 기존 프레임 확인
        existing_frames = sorted(frames_dir.glob(f"utter_{utter_id}_*.jpg"))
        if existing_frames:
            logger.info(f"[{idx+1}/{len(utterances_df)}] utter_{utter_id}: 기존 프레임 재사용 ({len(existing_frames)}장)")
            frame_map[utter_id] = existing_frames
            total_captured += len(existing_frames)
            continue
        
        # 캡처할 시점 결정
        if duration < duration_threshold:
            # 5초 미만: 중간 지점 1장만
            timestamps = [start_sec + duration / 2]
            logger.info(f"[{idx+1}/{len(utterances_df)}] utter_{utter_id}: dur={duration:.1f}s (짧음) → 중간 1장")
        else:
            # 5초 이상: 시작/중간/끝
            timestamps = [
                start_sec + 0.5,
                start_sec + duration / 2,
                end_sec - 0.5,
            ]
            logger.info(f"[{idx+1}/{len(utterances_df)}] utter_{utter_id}: dur={duration:.1f}s (길음) → 3장")
        
        captured_frames = []
        
        for ts_idx, ts in enumerate(timestamps):
            frame_num = int(ts * fps)
            frame_num = max(0, min(frame_num, total_frames_count - 1))
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            
            if not ret:
                logger.warning(f"  프레임 {frame_num} 읽기 실패")
                continue
            
            frame_filename = f"utter_{utter_id}_{ts_idx:02d}.jpg"
            frame_path = frames_dir / frame_filename
            cv2.imwrite(str(frame_path), frame)
            captured_frames.append(frame_path)
            total_captured += 1
        
        frame_map[utter_id] = captured_frames
    
    cap.release()
    
    logger.info(f"프레임 캡처 완료: 총 {total_captured}장")
    return frame_map


# ========================================
# VLM 스코어보드 추출
# ========================================

SCOREBOARD_SCHEMA = {
    "원정팀": "str",
    "홈팀": "str",
    "원정팀 점수": "int",
    "홈팀 점수": "int",
    "이닝": "int",
    "이닝 상황": "str",
    "볼": "int",
    "스트라이크": "int",
    "아웃": "int",
    "주자_1루": "bool",
    "주자_2루": "bool",
    "주자_3루": "bool",
    "투수 이름": "str",
    "투구 수": "int",
    "타자 이름": "str",
    "타자 타순": "int",
    "타자 경기 기록": "str",
}


def extract_scoreboard_from_frame(
    frame_path: Path,
    model,
    processor,
) -> Dict[str, Any]:
    """단일 프레임에서 스코어보드 추출"""
    
    schema_text = json.dumps(SCOREBOARD_SCHEMA, ensure_ascii=False, indent=2)
    
    prompt = f"""이 야구 경기 영상의 스코어보드를 분석하여 아래 JSON 스키마에 맞춰 정보를 추출하세요.

<schema>
{schema_text}
</schema>

**중요 규칙:**
1. 스코어보드가 없거나 정보를 확실히 알 수 없으면 해당 필드는 null로 반환
2. 반드시 JSON 형식으로만 응답 (다른 텍스트 포함 금지)
3. 주자 정보는 true/false로 명확히 표시
4. 숫자는 정수로, 문자열은 따옴표로 감싸기

JSON 응답:"""

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{frame_path.absolute()}"},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    
    device = model.device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
        )
    
    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    response = processor.decode(generated_ids, skip_special_tokens=True).strip()
    
    # 메모리 정리
    del inputs, outputs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # JSON 파싱
    try:
        # JSON 블록 추출
        if "```json" in response:
            json_start = response.index("```json") + 7
            json_end = response.index("```", json_start)
            json_str = response[json_start:json_end].strip()
        elif "```" in response:
            json_start = response.index("```") + 3
            json_end = response.index("```", json_start)
            json_str = response[json_start:json_end].strip()
        else:
            json_str = response.strip()
        
        data = json.loads(json_str)
        return data
    
    except Exception as e:
        logger.warning(f"JSON 파싱 실패: {e}")
        logger.warning(f"응답: {response[:200]}...")
        return {}


def merge_scoreboard_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """여러 프레임의 결과를 병합 (가장 완전한 정보 선택)"""
    if not results:
        return {}
    
    if len(results) == 1:
        return results[0]
    
    merged = {}
    
    for key in SCOREBOARD_SCHEMA.keys():
        # 각 결과에서 해당 키의 값을 수집
        values = [r.get(key) for r in results if r.get(key) is not None]
        
        if not values:
            merged[key] = None
        elif len(values) == 1:
            merged[key] = values[0]
        else:
            # 가장 많이 등장한 값 선택
            from collections import Counter
            counter = Counter(values)
            merged[key] = counter.most_common(1)[0][0]
    
    return merged


# ========================================
# 메인 파이프라인
# ========================================

def build_vlm_scoreboard_for_utterances(
    utter_csv_path: Path | str,
    video_path: Path | str,
    frames_root_dir: Path | str,
    output_csv_path: Path | str,
    duration_threshold: float = 5.0,
    resume: bool = True,
) -> Path:
    """
    utterances CSV에서 VLM 스코어보드 추출
    
    Args:
        utter_csv_path: utterances CSV
        video_path: 원본 비디오
        frames_root_dir: 프레임 저장 루트
        output_csv_path: 출력 CSV
        duration_threshold: 단일 프레임 캡처 기준 (초)
        resume: 중단된 작업 재개 여부
    """
    utter_csv_path = Path(utter_csv_path)
    video_path = Path(video_path)
    frames_root_dir = Path(frames_root_dir)
    output_csv_path = Path(output_csv_path)
    
    logger.info("="*80)
    logger.info("VLM 스코어보드 추출 시작")
    logger.info("="*80)
    logger.info(f"입력 CSV: {utter_csv_path}")
    logger.info(f"비디오: {video_path}")
    logger.info(f"프레임 디렉토리: {frames_root_dir}")
    logger.info(f"출력 CSV: {output_csv_path}")
    logger.info(f"단일 프레임 기준: {duration_threshold}초")
    logger.info("="*80)
    
    # utterances 로드
    df = pd.read_csv(utter_csv_path)
    df = df.sort_values('start_sec').reset_index(drop=True)
    
    logger.info(f"총 {len(df)}개 utterance")
    
    # 프레임 디렉토리 생성
    video_stem = video_path.stem
    frames_dir = frames_root_dir / video_stem
    
    # 기존 결과 로드 (resume)
    existing_results = {}
    if resume and output_csv_path.exists():
        existing_df = pd.read_csv(output_csv_path)
        for _, row in existing_df.iterrows():
            utter_id = row['utter_id']
            result = {k.replace('sb_', ''): row.get(f'sb_{k}') 
                     for k in SCOREBOARD_SCHEMA.keys() 
                     if f'sb_{k}' in row}
            existing_results[utter_id] = result
        logger.info(f"기존 결과 {len(existing_results)}개 로드 (resume)")
    
    # 프레임 캡처 (최적화)
    logger.info("프레임 캡처 시작...")
    frame_map = capture_frames_optimized(
        video_path=video_path,
        utterances_df=df,
        frames_dir=frames_dir,
        duration_threshold=duration_threshold,
    )
    
    # VLM 모델 로드
    model, processor = get_vlm_model()
    
    # VLM 추론
    results = []
    processed_count = 0
    skipped_count = 0
    
    for idx, row in df.iterrows():
        utter_id = row['utter_id']
        
        # 기존 결과 확인
        if utter_id in existing_results:
            logger.info(f"[{idx+1}/{len(df)}] utter_{utter_id}: 기존 결과 재사용")
            scoreboard = existing_results[utter_id]
            skipped_count += 1
        else:
            # VLM 추론
            frames = frame_map.get(utter_id, [])
            if not frames:
                logger.warning(f"[{idx+1}/{len(df)}] utter_{utter_id}: 프레임 없음")
                scoreboard = {}
            else:
                logger.info(f"[{idx+1}/{len(df)}] utter_{utter_id}: VLM 추론 ({len(frames)}장)")
                
                frame_results = []
                for frame_path in frames:
                    try:
                        result = extract_scoreboard_from_frame(frame_path, model, processor)
                        frame_results.append(result)
                    except Exception as e:
                        logger.error(f"  프레임 {frame_path.name} 추론 실패: {e}")
                
                scoreboard = merge_scoreboard_results(frame_results)
                processed_count += 1
        
        # 결과 병합
        record = {
            'utter_id': utter_id,
            'video_id': row.get('video_id', video_stem),
            'role': row.get('role'),
            'text': row.get('text'),
            'start_sec': row['start_sec'],
            'end_sec': row['end_sec'],
            'duration': row['end_sec'] - row['start_sec'],
        }
        
        for key in SCOREBOARD_SCHEMA.keys():
            record[f'sb_{key}'] = scoreboard.get(key)
        
        results.append(record)
        
        # 50개마다 중간 저장
        if (processed_count + skipped_count) % 50 == 0:
            temp_df = pd.DataFrame(results)
            temp_path = output_csv_path.with_name(output_csv_path.stem + ".tmp.csv")
            temp_df.to_csv(temp_path, index=False, encoding='utf-8-sig')
            logger.info(f"  → 중간 저장: {processed_count + skipped_count}개 처리")
    
    # 최종 저장
    result_df = pd.DataFrame(results)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
    
    logger.info("="*80)
    logger.info("VLM 스코어보드 추출 완료")
    logger.info("="*80)
    logger.info(f"총 처리: {len(df)}개")
    logger.info(f"  - 새로 추론: {processed_count}개")
    logger.info(f"  - 재사용: {skipped_count}개")
    logger.info(f"출력: {output_csv_path}")
    logger.info("="*80)
    
    return output_csv_path
