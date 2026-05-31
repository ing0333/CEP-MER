"""
meld_vm_v2.py
"""

import os, subprocess, shutil
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from ultralytics import YOLO

VIDEO_ROOT  = "/workspace/KangIngyeong_for_everything/MM_rebutal/MELD/test"
OUTPUT_ROOT = "/workspace/KangIngyeong_for_everything/MM_rebutal/MELD_intervention/C1_social/test"
CHECKPOINT  = "/workspace/KangIngyeong_for_everything/MM_rebutal/MELD_intervention/C1_social/progress.txt"
TMP_DIR     = Path("/workspace/KangIngyeong_for_everything/MM_rebutal/tmp_social")

os.makedirs(OUTPUT_ROOT, exist_ok=True)
os.makedirs(TMP_DIR, exist_ok=True)

# 체크포인트 로드
processed = set()
if os.path.exists(CHECKPOINT):
    with open(CHECKPOINT) as f:
        processed = set(f.read().splitlines())
print(f"이미 처리된 파일: {len(processed)}개")

# YOLOv8 세그멘테이션 모델 로드 (최초 1회만)
print("YOLOv8 모델 로드 중...")
model = YOLO("yolov8s-seg.pt")   # nano: 가장 가벼움 / x-seg.pt: 정확도 높음
print("모델 로드 완료")


def remove_social_background(video_path: Path, output_path: Path) -> bool:
    """
    배경만 블랙아웃 — 모든 사람(person) 영역은 원본 유지
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [오류] 열기 실패: {video_path.name}")
        return False

    fps    = cap.get(cv2.CAP_PROP_FPS) or 25
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 임시 무음 영상 저장
    tmp_video = TMP_DIR / f"{video_path.stem}_social.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(tmp_video), fourcc, fps, (width, height))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # YOLOv8 추론 (person 클래스 = 0)
        results = model(frame, classes=[0], verbose=False)

        # 사람 마스크 합산
        person_mask = np.zeros((height, width), dtype=np.uint8)

        if results[0].masks is not None:
            for mask in results[0].masks.data:
                mask_np = mask.cpu().numpy()
                mask_resized = cv2.resize(
                    mask_np, (width, height),
                    interpolation=cv2.INTER_NEAREST
                )
                person_mask = np.maximum(person_mask, (mask_resized > 0.5).astype(np.uint8))

        # 배경 블랙아웃: 사람 영역만 원본 유지
        masked_frame = np.zeros_like(frame)
        masked_frame[person_mask == 1] = frame[person_mask == 1]

        writer.write(masked_frame)
        frame_count += 1

    cap.release()
    writer.release()

    if frame_count == 0:
        print(f"  [오류] 프레임 없음: {video_path.name}")
        return False

    # 원본 오디오 + 처리된 영상 합치기
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run([
        "ffmpeg", "-y",
        "-i", str(tmp_video),      # 처리된 영상 (무음)
        "-i", str(video_path),     # 원본 (오디오 소스)
        "-c:v", "libx264",
        "-c:a", "aac",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-shortest",
        str(output_path)
    ], capture_output=True, text=True)

    tmp_video.unlink(missing_ok=True)

    if result.returncode != 0:
        print(f"  [ffmpeg 오류] {video_path.name}: {result.stderr[-200:]}")
        return False

    return True


# ── 메인 처리 ──
split   = "test"
mp4_files = sorted(f for f in Path(VIDEO_ROOT).glob("*.mp4") if not f.name.startswith("._"))
mp4_files = [f for f in mp4_files if f.name not in processed]
out_dir   = Path(OUTPUT_ROOT)

print(f"\n[ {split} ] 미처리 {len(mp4_files)}개 처리 시작")
success = 0

for video_path in mp4_files:
    output_path = out_dir / video_path.name

    ok = remove_social_background(video_path, output_path)

    if ok:
        success += 1
        with open(CHECKPOINT, "a") as f:
            f.write(video_path.name + "\n")
        processed.add(video_path.name)

    if success % 50 == 0 and success > 0:
        print(f"  {success}개 완료")

print(f"\n[ {split} ] 완료: {success}개")