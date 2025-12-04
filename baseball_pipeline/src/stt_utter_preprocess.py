# src/stt_utter_preprocess.py

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd


def build_utterances_from_clova_raw(
    raw_json_path: Path | str,
    out_csv_path: Path | str,
    pause_threshold_ms: int = 600,
) -> Path:
    """
    Clova STT raw JSON을 utterances.csv로 변환
    
    Args:
        raw_json_path: Clova raw JSON 경로
        out_csv_path: 출력 CSV 경로
        pause_threshold_ms: pause 기준 (밀리초)
    
    Returns:
        출력 CSV 경로
    """
    raw_json_path = Path(raw_json_path)
    out_csv_path = Path(out_csv_path)
    
    print(f"[PREPROCESS] STT 전처리 시작")
    print(f"  입력: {raw_json_path}")
    print(f"  출력: {out_csv_path}")
    
    # JSON 로드
    with open(raw_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    segments = data.get('segments', [])
    
    if not segments:
        raise ValueError(f"segments가 비어있습니다: {raw_json_path}")
    
    print(f"  - 총 segments: {len(segments)}개")
    
    # Speaker 통계 (가장 많이 말한 사람 = 캐스터)
    speaker_stats = {}
    for seg in segments:
        speaker_label = seg.get('speaker', {}).get('label', 'unknown')
        speaker_stats[speaker_label] = speaker_stats.get(speaker_label, 0) + 1
    
    # 캐스터 = 가장 많이 말한 사람
    if speaker_stats:
        caster_label = max(speaker_stats, key=speaker_stats.get)
        print(f"  - Speaker 통계: {speaker_stats}")
        print(f"  - 캐스터 자동 할당: {caster_label}")
    else:
        caster_label = None
    
    # Utterance 생성
    utterances = []
    utter_id = 0
    
    current_utterance = None
    
    for seg_idx, seg in enumerate(segments):
        start_sec = seg.get('start', 0) / 1000.0  # ms → sec
        end_sec = seg.get('end', 0) / 1000.0
        text = seg.get('text', '').strip()
        confidence = seg.get('confidence', 0.0)
        speaker_label = seg.get('speaker', {}).get('label', 'unknown')
        
        if not text:
            continue
        
        # 역할 할당
        if speaker_label == caster_label:
            role = 'caster'
        else:
            role = 'analyst'
        
        # 새 utterance 시작 조건
        # 1) 첫 segment
        # 2) speaker 변경
        # 3) 긴 pause
        start_new = False
        
        if current_utterance is None:
            start_new = True
        else:
            # speaker 변경
            if current_utterance['role'] != role:
                start_new = True
            # 긴 pause (이전 끝 ~ 현재 시작)
            elif (start_sec - current_utterance['end_sec']) * 1000 > pause_threshold_ms:
                start_new = True
        
        if start_new:
            # 이전 utterance 저장
            if current_utterance is not None:
                utterances.append(current_utterance)
                utter_id += 1
            
            # 새 utterance 시작
            current_utterance = {
                'utter_id': utter_id,
                'video_id': raw_json_path.stem.replace('.clova_raw', ''),
                'role': role,
                'speaker_label': speaker_label,
                'text': text,
                'start_sec': start_sec,
                'end_sec': end_sec,
                'confidence': confidence,
            }
        else:
            # 기존 utterance에 병합
            current_utterance['text'] += ' ' + text
            current_utterance['end_sec'] = end_sec
            # confidence는 평균
            current_utterance['confidence'] = (
                current_utterance['confidence'] + confidence
            ) / 2.0
    
    # 마지막 utterance 저장
    if current_utterance is not None:
        utterances.append(current_utterance)
    
    # DataFrame 생성
    df = pd.DataFrame(utterances)
    
    # duration 계산
    df['duration'] = df['end_sec'] - df['start_sec']
    
    # 컬럼 순서 정리
    columns = [
        'utter_id', 'video_id', 'role', 'speaker_label',
        'text', 'start_sec', 'end_sec', 'duration', 'confidence'
    ]
    df = df[columns]
    
    # 저장
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv_path, index=False, encoding='utf-8-sig')
    
    print(f"[PREPROCESS] 완료!")
    print(f"  - 총 utterances: {len(df)}개")
    print(f"  - 캐스터: {(df['role'] == 'caster').sum()}개")
    print(f"  - 해설: {(df['role'] == 'analyst').sum()}개")
    print(f"  - 출력: {out_csv_path}")
    
    return out_csv_path


def load_utterances_csv(csv_path: Path | str) -> pd.DataFrame:
    """utterances CSV 로드"""
    csv_path = Path(csv_path)
    
    if not csv_path.exists():
        raise FileNotFoundError(f"utterances CSV를 찾을 수 없습니다: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    required_cols = ['utter_id', 'text', 'start_sec', 'end_sec']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"필수 컬럼 누락: {col}")
    
    return df


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python stt_utter_preprocess.py <raw_json_path> <out_csv_path>")
        sys.exit(1)
    
    raw_json = sys.argv[1]
    out_csv = sys.argv[2]
    
    build_utterances_from_clova_raw(raw_json, out_csv)
