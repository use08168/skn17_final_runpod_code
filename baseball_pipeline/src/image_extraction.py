from __future__ import annotations

import cv2
from pathlib import Path
from typing import List, Dict, Any, Optional


def _open_video(video_path: str):
    """
    내부용: 비디오를 열고 (cap, fps, frame_count, duration)를 반환.
    """
    video_path = Path(video_path)
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


def capture_frames_near_timestamps(
    video_path: str,
    timestamps_sec: List[float],
    output_dir: str = "frames",
    filename_prefix: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    (기존 일반 버전) 주어진 동영상에서 타임스탬프(초 단위) 근처 프레임을 캡처해서
    output_dir에 이미지 파일(jpg)로 저장한다.
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap, fps, frame_count, duration = _open_video(str(video_path))

    base_name = filename_prefix if filename_prefix is not None else video_path.stem
    results = []

    for idx, ts in enumerate(timestamps_sec):
        ts = float(ts)
        if ts < 0 or ts > duration:
            print(f"[WARN] {ts:.3f}초는 영상 길이(0~{duration:.3f}) 밖입니다. 스킵.")
            results.append(
                {"timestamp": ts, "out_path": None, "success": False, "reason": "out_of_range"}
            )
            continue

        frame_idx = int(round(ts * fps))
        frame_idx = max(0, min(frame_idx, frame_count - 1))

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        print(f"[DEBUG] ts={ts:.3f}s, frame_idx={frame_idx}, ret={ret}")

        if not ret or frame is None:
            print(f"[WARN] {ts:.3f}s 근처 프레임을 읽지 못했습니다.")
            results.append(
                {"timestamp": ts, "out_path": None, "success": False, "reason": "read_fail"}
            )
            continue

        safe_ts = str(round(ts, 3)).replace(".", "_")
        filename = f"{base_name}_{safe_ts}.jpg"
        out_path = output_dir / filename

        saved = cv2.imwrite(str(out_path), frame)
        if saved:
            print(f"[INFO] {ts:.3f}s 근처 프레임 저장: {out_path}")
            results.append(
                {"timestamp": ts, "out_path": str(out_path), "success": True, "reason": None}
            )
        else:
            print(f"[ERROR] {out_path} 저장 실패!")
            results.append(
                {"timestamp": ts, "out_path": str(out_path), "success": False, "reason": "write_fail"}
            )

    cap.release()
    print("[DONE] 프레임 캡처 완료.")
    return results


def capture_frames_for_sets(
    video_path: str,
    sets: List[Dict[str, Any]],
    output_dir: str = "frames",
    time_key: str = "set_start_sec",
    id_key: str = "set_id",
) -> List[Dict[str, Any]]:
    """
    최종 세트 JSON(list of dict)을 입력으로 받아,
    각 세트의 time_key(기본: set_start_sec) 위치에서 프레임을 캡처하고
    파일명을 set_id.jpg로 저장한다.

    ➜ analyst_text가 null인 세트는 이미지 추출을 건너뛴다.

    Parameters
    ----------
    video_path : str
        동영상 파일 경로
    sets : List[dict]
        각 세트 dict. 예) {
            "set_id": "vocals-1",
            "set_start_sec": 10.29,
            "analyst_text": "...",  # 없거나 null일 수 있음
            ...
        }
    output_dir : str
        출력 이미지 폴더
    time_key : str
        세트 dict에서 타임스탬프를 읽어올 키 이름 (기본: "set_start_sec")
    id_key : str
        세트 dict에서 이미지 파일명으로 사용할 set_id 키 이름 (기본: "set_id")

    Returns
    -------
    List[dict]
        각 세트별 캡처 결과 요약 리스트
        예) { "set_id": "vocals-1", "timestamp": 10.29, "out_path": ".../vocals-1.jpg", "success": True }
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap, fps, frame_count, duration = _open_video(str(video_path))

    results = []

    for s in sets:
        # ★ analyst_text가 null(또는 비어 있음)이면 스킵 ★
        if "analyst_text" in s:
            if s["analyst_text"] is None or (isinstance(s["analyst_text"], str) and not s["analyst_text"].strip()):
                set_id_for_log = s.get(id_key)
                print(f"[SKIP] 세트 {set_id_for_log}: analyst_text가 null/빈 문자열 → 이미지 추출 건너뜀")
                results.append(
                    {
                        "set_id": set_id_for_log,
                        "timestamp": None,
                        "out_path": None,
                        "success": False,
                        "reason": "analyst_null",
                    }
                )
                continue

        # 필수 필드 체크
        if time_key not in s or id_key not in s:
            print(f"[WARN] 세트에 {time_key} 또는 {id_key}가 없습니다. 세트 스킵: {s}")
            results.append(
                {
                    "set_id": s.get(id_key),
                    "timestamp": None,
                    "out_path": None,
                    "success": False,
                    "reason": "missing_keys",
                }
            )
            continue

        try:
            ts = float(s[time_key])
        except (TypeError, ValueError):
            print(f"[WARN] 세트 {s.get(id_key)}의 {time_key} 값을 float로 변환할 수 없습니다. 값={s[time_key]!r}")
            results.append(
                {
                    "set_id": s.get(id_key),
                    "timestamp": s.get(time_key),
                    "out_path": None,
                    "success": False,
                    "reason": "invalid_timestamp",
                }
            )
            continue

        set_id = str(s[id_key])

        if ts < 0 or ts > duration:
            print(
                f"[WARN] 세트 {set_id}: {ts:.3f}초는 영상 길이(0~{duration:.3f}) 밖입니다. 스킵."
            )
            results.append(
                {
                    "set_id": set_id,
                    "timestamp": ts,
                    "out_path": None,
                    "success": False,
                    "reason": "out_of_range",
                }
            )
            continue

        frame_idx = int(round(ts * fps))
        frame_idx = max(0, min(frame_idx, frame_count - 1))

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        print(f"[DEBUG] set_id={set_id}, ts={ts:.3f}s, frame_idx={frame_idx}, ret={ret}")

        if not ret or frame is None:
            print(f"[WARN] 세트 {set_id}: {ts:.3f}s 근처 프레임을 읽지 못했습니다.")
            results.append(
                {
                    "set_id": set_id,
                    "timestamp": ts,
                    "out_path": None,
                    "success": False,
                    "reason": "read_fail",
                }
            )
            continue

        # ★ 파일명 = set_id.jpg ★
        filename = f"{set_id}.jpg"
        out_path = output_dir / filename

        saved = cv2.imwrite(str(out_path), frame)
        if saved:
            print(f"[INFO] 세트 {set_id}: {ts:.3f}s 프레임 저장 → {out_path}")
            results.append(
                {
                    "set_id": set_id,
                    "timestamp": ts,
                    "out_path": str(out_path),
                    "success": True,
                    "reason": None,
                }
            )
        else:
            print(f"[ERROR] 세트 {set_id}: {out_path} 저장 실패!")
            results.append(
                {
                    "set_id": set_id,
                    "timestamp": ts,
                    "out_path": str(out_path),
                    "success": False,
                    "reason": "write_fail",
                }
            )

    cap.release()
    print("[DONE] 세트 기반 프레임 캡처 완료.")
    return results