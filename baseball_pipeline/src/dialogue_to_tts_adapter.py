# src/dialogue_to_tts_adapter.py

from __future__ import annotations

from pathlib import Path
import pandas as pd


def convert_dialogue_to_tts_format(
    dialogue_csv_path: Path | str,
    output_csv_path: Path | str,
    video_stem: str = None,
) -> Path:
    """
    commentary_dialogue.csv를 TTS/align 코드들이 기대하는 형식으로 변환
    
    입력 (dialogue_commentary.csv):
        timestamp, utter_id, duration, caster, caster_duration, analyst, analyst_duration, ...
    
    출력 (tts_phrases 형식):
        utterance_id, source_video, role, speaker_id, speaker_name, 
        start_sec, end_sec, text, confidence, orig_text, llm_text, ...
    
    Args:
        dialogue_csv_path: 입력 dialogue CSV
        output_csv_path: 출력 tts_phrases CSV
        video_stem: 비디오 파일명 (확장자 제외)
    """
    dialogue_csv_path = Path(dialogue_csv_path)
    output_csv_path = Path(output_csv_path)
    
    df = pd.read_csv(dialogue_csv_path)
    
    # video_stem 결정
    if video_stem is None:
        video_stem = dialogue_csv_path.stem.replace(".dialogue_commentary", "")
    
    print(f"[ADAPTER] 데이터 형식 변환 시작")
    print(f"  입력: {dialogue_csv_path}")
    print(f"  video_stem: {video_stem}")
    
    records = []
    
    for idx, row in df.iterrows():
        timestamp = row['timestamp']
        utter_id = row['utter_id']
        caster_text = str(row['caster']).strip()
        analyst_text = str(row['analyst']).strip()
        caster_dur = row['caster_duration']
        analyst_dur = row['analyst_duration']
        
        # 캐스터 발화 추가
        if caster_text and caster_text.lower() not in ['nan', '', 'none']:
            records.append({
                'utterance_id': f"{utter_id}_caster",
                'source_video': video_stem + ".mp4",
                'role': 'caster',
                'speaker_id': '1',
                'speaker_name': 'A',
                'start_sec': timestamp,
                'end_sec': timestamp + caster_dur,
                'text': caster_text,
                'confidence': 1.0,
                'orig_text': caster_text,
                'llm_text': caster_text,  # 캐스터는 원본 그대로
                'source_utter_id': utter_id,
            })
        
        # 해설위원 발화 추가
        if analyst_text and analyst_text.lower() not in ['nan', '', 'none'] and analyst_dur > 0:
            records.append({
                'utterance_id': f"{utter_id}_analyst",
                'source_video': video_stem + ".mp4",
                'role': 'analyst',
                'speaker_id': '2',
                'speaker_name': 'B',
                'start_sec': timestamp + caster_dur,
                'end_sec': timestamp + caster_dur + analyst_dur,
                'text': analyst_text,
                'confidence': 1.0,
                'orig_text': analyst_text,
                'llm_text': analyst_text,  # LLM 생성 텍스트
                'source_utter_id': utter_id,
            })
    
    if not records:
        raise ValueError("[ADAPTER] 변환할 데이터가 없습니다.")
    
    result_df = pd.DataFrame(records)
    
    # 시간순 정렬
    result_df = result_df.sort_values('start_sec').reset_index(drop=True)
    
    # 저장
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
    
    print(f"[ADAPTER] 변환 완료:")
    print(f"  출력: {output_csv_path}")
    print(f"  총 moments: {len(df)}개")
    print(f"  총 utterances: {len(result_df)}개")
    print(f"    - 캐스터: {(result_df['role'] == 'caster').sum()}개")
    print(f"    - 해설: {(result_df['role'] == 'analyst').sum()}개")
    
    return output_csv_path
