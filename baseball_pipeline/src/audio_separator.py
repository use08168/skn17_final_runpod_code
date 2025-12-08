from __future__ import annotations

import os
import logging
from pathlib import Path
from audio_separator.separator import Separator


def separate_audio_sota(video_path, output_dir, device="cuda"):
    """
    SOTA(State-of-the-Art) 모델인 BS-RoFormer-ViperX-1297을 사용하여
    비디오 파일에서 'vocals'(해설)과 'instrumental'(현장음)을 최고 성능으로 분리합니다.
    
    Args:
        video_path (str): 입력 비디오 파일 경로.
        output_dir (str): 분리된 오디오 파일을 저장할 디렉토리.
        device (str): 연산 장치 ('cuda' 권장, 없을 경우 'cpu').
    
    Returns:
        dict: 분리된 파일들의 경로 사전 {'vocals': path, 'no_vocals': path}.
    """
    
    # 1. 출력 디렉토리 생성
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 모델 파일명 지정 (ViperX 1297 모델 - 현재 보컬 분리 성능 1위)
    # audio-separator 라이브러리가 최초 실행 시 자동으로 다운로드합니다.
    model_filename = 'model_bs_roformer_ep_317_sdr_12.9755.ckpt'
    
    print(f"\n 오디오 분리 시작: {video_path}")
    print(f"  사용 모델: {model_filename} (BS-RoFormer-ViperX)")
    print(f"  실행 장치: {device}")

    # 2. Separator 인스턴스 초기화 및 고품질 파라미터 설정
    # mdx_params 내의 옵션들은 모델 아키텍처에 따라 자동 조정되지만,
    # 명시적인 고품질 설정을 위해 일부 값을 지정합니다.
    separator = Separator(
        log_level=logging.INFO,               # 로그 레벨 설정
        model_file_dir=os.path.join(output_dir, "models"), # 모델 저장 경로 캐싱
        output_dir=output_dir,                # 결과물 저장 경로
        output_format="wav",                  # 고음질 유지를 위한 WAV 포맷
        normalization_threshold=0.9,          # 클리핑 방지를 위한 노말라이즈 (0.9 권장)
        output_single_stem=None,              # 두 트랙(Vocals, Inst) 모두 출력
        
        # MDX/RoFormer 관련 파라미터 (고품질 설정)
        mdx_params={
            "hop_length": 1024,               # 주파수 해상도 관련 (기본값)
            "segment_size": 256,              # 청크 사이즈 (VRAM에 따라 조절 가능)
            "overlap": 0.25,                  # 청크 간 오버랩 비율 (연결 부위 자연스럽게)
            "batch_size": 1,                  # 안정성을 위해 배치 사이즈 1 설정
            "enable_denoise": True            # 모델 자체 노이즈 캔슬링 활성화
        }
    )

    # 3. 모델 로드
    print(f"  >> 모델 로딩 중... (최초 실행 시 다운로드에 시간이 소요됩니다)")
    separator.load_model(model_filename=model_filename)

    # 4. 분리 실행
    # audio-separator는 ffmpeg를 내장하여 비디오 파일에서 오디오를 직접 추출/변환합니다.
    print(f"  >> 분리 작업 수행 중...")
    output_files = separator.separate(video_path)

    # 5. 결과 파일 매핑 및 리네이밍
    # 라이브러리는 출력 파일명에 모델 이름 등을 포함하므로, 사용자가 원하는 이름으로 변경합니다.
    vocals_path = None
    no_vocals_path = None

    print(f"  >> 파일 정리 및 이름 변경 중...")
    for file in output_files:
        original_file_path = output_path / file
        
        # BS-RoFormer 모델의 출력 스템 이름 확인 (Vocals / Instrumental)
        if "Vocals" in file:
            target_path = output_path / "vocals.wav"
            if target_path.exists(): target_path.unlink() # 기존 파일 있으면 삭제
            os.rename(original_file_path, target_path)
            vocals_path = target_path
            
        elif "Instrumental" in file:
            target_path = output_path / "no_vocals.wav"
            if target_path.exists(): target_path.unlink()
            os.rename(original_file_path, target_path)
            no_vocals_path = target_path

    return {
        "vocals": vocals_path,
        "no_vocals": no_vocals_path
    }
