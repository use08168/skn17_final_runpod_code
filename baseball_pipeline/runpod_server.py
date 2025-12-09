# runpod_server.py
from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from pathlib import Path
import shutil
import os
import json
import uvicorn
from typing import Optional
import traceback
import uuid

app = FastAPI(title="Baseball Commentary Pipeline with Voice Selection")

# ==========================================
# 디렉토리 설정
# ==========================================
PROJECT_ROOT = Path("/workspace/baseball_pipeline")
DATA_DIR = PROJECT_ROOT / "data"
INPUT_VIDEO_DIR = DATA_DIR / "input_videos"
OUTPUT_VIDEO_DIR = DATA_DIR / "output_videos"
AUDIO_ROOT = DATA_DIR / "audio_separator"
TTS_REFS_DIR = DATA_DIR / "tts_refs"

for d in [DATA_DIR, INPUT_VIDEO_DIR, OUTPUT_VIDEO_DIR, AUDIO_ROOT, TTS_REFS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ==========================================
# 해설위원 매핑 (notebook 코드와 동일)
# ==========================================
ANALYST_MAPPING = {
    1: "park",   # 박찬호
    2: "lee",    # 이대호
    3: "kim",    # 김광현
}

ANALYST_NAMES = {
    1: "박찬호",
    2: "이대호",
    3: "김광현",
}

# LLM 모델 매핑 (추후 각 해설위원별 파인튜닝 모델로 교체)
ANALYST_LLM_MODELS = {
    1: "SeHee8546/kanana-1.5-8b-pakchanho-lora-v2",      # 박찬호
    2: "SeHee8546/kanana-1.5-8b-daeholeee-lora-v1",      # 이대호 (예정)
    3: "SeHee8546/kanana-1.5-8b-kimgh-lora-v1",          # 김광현 (예정)
}

# 작업 상태 관리
job_status = {}

# ==========================================
# 파이프라인 실행 함수
# ==========================================
def run_full_pipeline(
    video_path: Path,
    analyst_select: int,
    job_id: str
):
    """전체 파이프라인 실행"""
    try:
        # 초기 상태 설정
        selected_analyst = ANALYST_MAPPING[analyst_select]
        selected_analyst_name = ANALYST_NAMES[analyst_select]
        selected_llm_model = ANALYST_LLM_MODELS[analyst_select]
        
        job_status[job_id] = {
            "status": "processing",
            "step": "초기화",
            "progress": 0,
            "analyst": selected_analyst_name,
            "analyst_code": selected_analyst,
        }
        
        print(f"\n{'='*60}")
        print(f"🎙️ 선택된 해설위원: {selected_analyst_name} ({selected_analyst})")
        print(f"📦 LLM 모델: {selected_llm_model}")
        print(f"{'='*60}\n")
        
        # 환경 설정
        import sys
        sys.path.append(str(PROJECT_ROOT / "src"))
        
        from src import DATA_DIR as SRC_DATA_DIR
        from src.audio_separator import separate_audio_sota
        from src.stt_pipeline import run_stt_pipeline
        from src.stt_event_splitter import stt_json_to_event_sets
        from src.image_extraction import capture_frames_for_sets
        from src.vlm_scoreboard import (
            load_scoreboard_model_and_processor,
            attach_scoreboard_to_sets,
        )
        from src.llm_generator import (
            load_pakchanho_model,
            generate_analyst_for_all_sets,
        )
        from src.json_tts_pipeline import run_full_tts_pipeline_from_json
        
        video_stem = video_path.stem
        
        # 디렉토리 설정
        STT_RAW_DIR = DATA_DIR / "stt_raw"
        STT_SEG_DIR = DATA_DIR / "stt_segments"
        LLM_OUT_DIR = DATA_DIR / "llm_outputs"
        TTS_AUDIO_DIR = DATA_DIR / "tts_audio"
        FRAMES_ROOT = DATA_DIR / "frames"
        
        for d in [STT_RAW_DIR, STT_SEG_DIR, LLM_OUT_DIR, TTS_AUDIO_DIR, FRAMES_ROOT]:
            d.mkdir(parents=True, exist_ok=True)
        
        # ==========================================
        # Step 1: Audio Separation
        # ==========================================
        job_status[job_id]["step"] = "음성 분리"
        job_status[job_id]["progress"] = 10
        print(f"\n[{job_id}] Step 1: Audio Separation")
        
        track_dict = separate_audio_sota(
            video_path=str(video_path),
            output_dir=str(AUDIO_ROOT),
            device="cuda"
        )
        vocals_path = track_dict["vocals"]
        no_vocals_path = track_dict["no_vocals"]
        
        # ==========================================
        # Step 2: STT
        # ==========================================
        job_status[job_id]["step"] = "음성 인식"
        job_status[job_id]["progress"] = 20
        print(f"\n[{job_id}] Step 2: STT")
        
        CLOVA_INVOKE_URL = os.environ.get("CLOVA_INVOKE_URL", "")
        CLOVA_SECRET_KEY = os.environ.get("CLOVA_SECRET_KEY", "")
        
        timeline_json_path = run_stt_pipeline(
            audio_path=vocals_path,
            invoke_url=CLOVA_INVOKE_URL,
            secret_key=CLOVA_SECRET_KEY,
            stt_raw_dir=STT_RAW_DIR,
            stt_seg_dir=STT_SEG_DIR,
            xlsx_keywords_path=None,
            use_domain_boostings=True,
            speaker_count_min=2,
            speaker_count_max=3,
            save_raw_json=True,
            pause_thresh_ms=50000,
        )
        
        # ==========================================
        # Step 3: Event Split
        # ==========================================
        job_status[job_id]["step"] = "이벤트 분할"
        job_status[job_id]["progress"] = 30
        print(f"\n[{job_id}] Step 3: Event Split")
        
        with timeline_json_path.open("r", encoding="utf-8") as f:
            stt_json = json.load(f)
        
        event_sets = stt_json_to_event_sets(
            stt_json,
            caster_gap=10.0,
            silence_gap=2.0,
        )
        
        json_after_split_path = timeline_json_path.with_name(
            f"{timeline_json_path.stem}_set_split.json"
        )
        with json_after_split_path.open("w", encoding="utf-8") as f:
            json.dump(event_sets, f, ensure_ascii=False, indent=2)
        
        # ==========================================
        # Step 4: Frame Extraction
        # ==========================================
        job_status[job_id]["step"] = "프레임 추출"
        job_status[job_id]["progress"] = 40
        print(f"\n[{job_id}] Step 4: Frame Extraction")
        
        capture_frames_for_sets(
            video_path=str(video_path),
            sets=event_sets,
            output_dir=str(FRAMES_ROOT)
        )
        
        # ==========================================
        # Step 5: VLM Scoreboard
        # ==========================================
        job_status[job_id]["step"] = "스코어보드 추출"
        job_status[job_id]["progress"] = 50
        print(f"\n[{job_id}] Step 5: VLM Scoreboard")
        
        vlm_model, vlm_processor = load_scoreboard_model_and_processor()
        
        scoreboard_json_path = json_after_split_path.with_name(
            f"{json_after_split_path.stem}_scoreboard.json"
        )
        
        updated_sets = attach_scoreboard_to_sets(
            json_after_split_path=json_after_split_path,
            output_json_path=scoreboard_json_path,
            frames_root=FRAMES_ROOT,
            video_path=video_path,
            model=vlm_model,
            processor=vlm_processor,
            retry_if_all_null=False,
            retry_offset_sec=2.0,
        )
        
        # ==========================================
        # Step 6: LLM Commentary Generation
        # ==========================================
        job_status[job_id]["step"] = f"해설 생성 ({selected_analyst_name})"
        job_status[job_id]["progress"] = 60
        print(f"\n[{job_id}] Step 6: LLM Generation - {selected_analyst_name}")
        
        json_llm_output_path = LLM_OUT_DIR / f"{video_stem}_{selected_analyst}.json"
        
        # 🔥 선택된 해설위원의 LLM 모델 로드
        model, tokenizer = load_pakchanho_model(
            base_model_name="kakaocorp/kanana-1.5-8b-instruct-2505",
            lora_model_id=selected_llm_model,
            load_in_4bit=True,
        )
        
        result_sets = generate_analyst_for_all_sets(
            json_in_path=scoreboard_json_path,
            json_out_path=json_llm_output_path,
            model=model,
            tokenizer=tokenizer,
            game_title="2025 KBO 경기",
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            no_repeat_ngram_size=3,
            base_max_new_tokens=512,
        )
        
        # ==========================================
        # Step 7: TTS Pipeline
        # ==========================================
        job_status[job_id]["step"] = f"음성 합성 ({selected_analyst_name})"
        job_status[job_id]["progress"] = 70
        print(f"\n[{job_id}] Step 7: TTS - {selected_analyst_name}")
        
        FISH_API_URL = "http://127.0.0.1:8080/v1/tts"
        
        # 🔥 notebook 코드와 동일한 참조 파일 구조
        CASTER_REF_WAVS = [TTS_REFS_DIR / "caster_jung.wav"]
        ANALYST_REF_WAVS = [TTS_REFS_DIR / f"analyst_{selected_analyst}.wav"]
        
        # 파일 존재 확인
        print("\n📁 참조 음성 파일 확인:")
        for ref_path in CASTER_REF_WAVS:
            if ref_path.exists():
                print(f"✅ 캐스터: {ref_path.name}")
            else:
                raise FileNotFoundError(f"캐스터 참조 없음: {ref_path}")
        
        for ref_path in ANALYST_REF_WAVS:
            if ref_path.exists():
                print(f"✅ 해설 ({selected_analyst_name}): {ref_path.name}")
            else:
                raise FileNotFoundError(f"해설 참조 없음: {ref_path}")
        
        final_tts_wav, aligned_csv, tts_csv_with_paths = run_full_tts_pipeline_from_json(
            json_sets_path=json_llm_output_path,
            video_path=video_path,
            caster_ref_wavs=CASTER_REF_WAVS,
            analyst_ref_wavs=ANALYST_REF_WAVS,
            fish_api_url=FISH_API_URL,
            min_text_chars=2,
            merge_same_role=True,
            merge_gap_thresh_sec=0.25,
            merge_short_thresh_sec=1.0,
            min_gap_sec=0.02,
            caster_extra_ratio=0.2,
            analyst_extra_ratio=2.0,
            max_analyst_expand_sec=7.0,
            analyst_priority_min_overlap_sec=0.5,
            min_gap_ms=60,
            tail_margin_ms=80,
            caster_max_speedup=1.3,
            analyst_max_speedup=1.8,
        )
        
        # ==========================================
        # Step 8: Final Video Encoding
        # ==========================================
        job_status[job_id]["step"] = "최종 영상 생성"
        job_status[job_id]["progress"] = 90
        print(f"\n[{job_id}] Step 8: Video Encoding")
        
        import subprocess
        
        final_video_path = OUTPUT_VIDEO_DIR / f"{video_stem}_{selected_analyst}.final.mp4"
        
        # 임시 오디오 믹싱
        temp_mixed_audio = OUTPUT_VIDEO_DIR / f"temp_mixed_{video_stem}.m4a"
        
        audio_cmd = [
            'ffmpeg', '-y',
            '-i', str(final_tts_wav),
            '-i', str(no_vocals_path),
            '-filter_complex',
            f'[0:a]volume=1.0[a1];[1:a]volume=0.7[a2];[a1][a2]amix=inputs=2:duration=first:dropout_transition=0[aout]',
            '-map', '[aout]',
            '-c:a', 'aac',
            '-b:a', '192k',
            str(temp_mixed_audio)
        ]
        subprocess.run(audio_cmd, check=True, capture_output=True)
        
        # 비디오 + 오디오 결합
        video_cmd = [
            'ffmpeg', '-y',
            '-i', str(video_path),
            '-i', str(temp_mixed_audio),
            '-map', '0:v:0',
            '-map', '1:a:0',
            '-c:v', 'copy',
            '-c:a', 'copy',
            '-shortest',
            str(final_video_path)
        ]
        subprocess.run(video_cmd, check=True, capture_output=True)
        
        # 임시 파일 삭제
        temp_mixed_audio.unlink(missing_ok=True)
        
        # ==========================================
        # 완료
        # ==========================================
        job_status[job_id] = {
            "status": "completed",
            "step": "완료",
            "progress": 100,
            "output_file": final_video_path.name,
            "analyst": selected_analyst_name,
            "analyst_code": selected_analyst,
        }
        print(f"\n[{job_id}] Pipeline Complete!")
        
    except Exception as e:
        error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        job_status[job_id] = {
            "status": "failed",
            "step": "오류 발생",
            "progress": 0,
            "error": str(e)
        }

# ==========================================
# API 엔드포인트
# ==========================================

@app.get("/")
async def root():
    return {
        "message": "Baseball Commentary Pipeline Server",
        "status": "running",
        "available_analysts": ANALYST_NAMES
    }

@app.get("/analysts")
async def list_analysts():
    """사용 가능한 해설위원 목록"""
    return {
        "analysts": [
            {
                "id": analyst_id,
                "name": ANALYST_NAMES[analyst_id],
                "code": ANALYST_MAPPING[analyst_id],
                "llm_model": ANALYST_LLM_MODELS[analyst_id],
                "ref_wav": f"analyst_{ANALYST_MAPPING[analyst_id]}.wav"
            }
            for analyst_id in sorted(ANALYST_NAMES.keys())
        ]
    }

@app.post("/process_video")
async def process_video(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    analyst_select: int = Form(1)  # 1, 2, 3 중 선택
):
    """
    영상 처리 요청
    
    Args:
        video: 업로드할 영상 파일
        analyst_select: 해설위원 선택 (1: 박찬호, 2: 이대호, 3: 김광현)
    """
    
    # 유효성 검사
    if analyst_select not in ANALYST_NAMES:
        return JSONResponse(
            status_code=400,
            content={
                "error": f"Invalid analyst_select: {analyst_select}",
                "valid_options": list(ANALYST_NAMES.keys())
            }
        )
    
    # 업로드된 영상 저장
    input_path = INPUT_VIDEO_DIR / video.filename
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)
    
    # Job ID 생성
    job_id = str(uuid.uuid4())
    
    # 백그라운드에서 파이프라인 실행
    background_tasks.add_task(
        run_full_pipeline,
        video_path=input_path,
        analyst_select=analyst_select,
        job_id=job_id
    )
    
    selected_analyst_name = ANALYST_NAMES[analyst_select]
    
    return {
        "status": "accepted",
        "job_id": job_id,
        "analyst": selected_analyst_name,
        "analyst_id": analyst_select,
        "message": f"파이프라인이 시작되었습니다. (해설: {selected_analyst_name})"
    }

@app.get("/status/{job_id}")
async def get_status(job_id: str):
    """작업 상태 확인"""
    if job_id not in job_status:
        return JSONResponse(
            status_code=404,
            content={"error": "Job not found"}
        )
    return job_status[job_id]

@app.get("/download/{filename}")
async def download_video(filename: str):
    """처리된 영상 다운로드"""
    file_path = OUTPUT_VIDEO_DIR / filename
    if not file_path.exists():
        return JSONResponse(
            status_code=404,
            content={"error": "File not found"}
        )
    
    return FileResponse(
        path=str(file_path),
        filename=filename,
        media_type='video/mp4'
    )

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "gpu": "available" if os.path.exists("/dev/nvidia0") else "unavailable"
    }

if __name__ == "__main__":
    # 환경 변수 로드
    HF_TOKEN = os.environ.get("HF_TOKEN", "")
    if HF_TOKEN:
        os.environ["HUGGINGFACE_HUB_TOKEN"] = HF_TOKEN
    
    print("\n" + "="*60)
    print("🎙️ Baseball Commentary Pipeline Server")
    print("="*60)
    print("\n사용 가능한 해설위원:")
    for analyst_id, name in ANALYST_NAMES.items():
        code = ANALYST_MAPPING[analyst_id]
        model = ANALYST_LLM_MODELS[analyst_id]
        print(f"  {analyst_id}. {name} ({code})")
        print(f"     모델: {model}")
        print(f"     참조: analyst_{code}.wav")
    print("\n" + "="*60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)