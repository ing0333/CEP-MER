#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
meld_vm_v1.py
MELD active-speaker foreground-preserving intervention pipeline (single-file version)

Changes from original:
- resume: 이미 output mp4가 있으면 스킵
- _work/, _debug/ 저장 없음: 중간 파일은 처리 후 즉시 삭제
- --shutdown_when_done: 완료 후 서버 자동 종료
"""

from __future__ import annotations

import argparse
import copy
import csv
import difflib
import glob
import json
import math
import os
import pickle
import re
import shutil
import subprocess
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "bool"):
    np.bool = bool

import pandas as pd
import python_speech_features
import torch
import tqdm
from scipy import signal
from scipy.interpolate import interp1d
from scipy.io import wavfile

from scenedetect.video_manager import VideoManager
from scenedetect.scene_manager import SceneManager
from scenedetect.stats_manager import StatsManager
from scenedetect.detectors import ContentDetector

warnings.filterwarnings("ignore")


# =========================================================
# config dataclasses
# =========================================================
@dataclass
class PipelineConfig:
    # paths
    input_dir: str
    output_dir: str
    talknet_repo: str
    meld_csv_glob: str = ""
    pretrain_model: str = "pretrain_TalkSet.model"

    # device / runtime
    device: str = "cuda"
    num_workers: int = 8

    # face detection / tracking
    facedet_scale: float = 0.25
    face_conf: float = 0.90
    min_track: int = 10
    num_failed_det: int = 10
    min_face_size: int = 1
    crop_scale: float = 0.40

    # fusion
    fps: int = 25
    talknet_smooth: int = 5
    switch_penalty: float = 1.25
    null_bias: float = 0.25
    speech_bonus: float = 0.75
    silence_penalty: float = 1.5
    transcript_match_bonus: float = 0.20

    # whisperx
    use_whisperx: bool = True
    whisper_model: str = "small"
    whisper_batch_size: int = 8
    whisper_compute_type: str = "float16"
    hf_token: str = ""
    use_diarization: bool = True

    # person segmentation / rendering
    person_seg_model: str = "yolo11n-seg.pt"
    person_seg_conf: float = 0.25
    person_seg_iou: float = 0.50
    person_seg_imgsz: int = 640
    person_seg_stride: int = 1
    mask_dilate_px: int = 7
    body_fallback_expand_x: float = 2.2
    body_fallback_expand_top: float = 1.6
    body_fallback_expand_bottom: float = 5.2

    # misc
    reset_output: bool = False
    limit: int = 0


@dataclass
class MeldClipMeta:
    split: str
    dialogue_id: int
    utterance_id: int
    speaker: str
    utterance: str
    emotion: str = ""
    sentiment: str = ""


# =========================================================
# environment / repo loading
# =========================================================
def ensure_talknet_imports(repo_path: str):
    repo = Path(repo_path).resolve()
    if not repo.exists():
        raise FileNotFoundError(f"TalkNet repo가 없습니다: {repo}")
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    os.chdir(str(repo))
    try:
        from model.faceDetector.s3fd import S3FD  # type: ignore
        from talkNet import talkNet  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "TalkNet repo import 실패. --talknet_repo 경로가 맞는지 확인하세요."
        ) from e
    return S3FD, talkNet


def ensure_ultralytics_imports():
    try:
        from ultralytics import YOLO  # type: ignore
    except Exception as e:
        raise RuntimeError("Ultralytics import 실패. `pip install ultralytics` 후 다시 실행하세요.") from e
    return YOLO


def ensure_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], check=True, capture_output=True)
    except Exception as e:
        raise RuntimeError("ffmpeg가 PATH에 없습니다.") from e


def maybe_download_talkset_checkpoint(checkpoint_path: Path):
    if checkpoint_path.exists():
        return
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "gdown",
        "--id", "1AbN9fCf9IexMxEKXLQY2KYBlb-IhSEea",
        "-O", str(checkpoint_path),
    ]
    print(f"[download] TalkNet TalkSet checkpoint -> {checkpoint_path}")
    subprocess.run(cmd, check=True)


# =========================================================
# MELD metadata
# =========================================================
def parse_meld_filename(stem: str) -> Tuple[Optional[int], Optional[int]]:
    m = re.search(r"dia(\d+)_utt(\d+)", stem, flags=re.IGNORECASE)
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


def load_meld_metadata(meld_csv_glob: str) -> Dict[Tuple[int, int], MeldClipMeta]:
    meta: Dict[Tuple[int, int], MeldClipMeta] = {}
    if not meld_csv_glob:
        return meta
    for csv_path in glob.glob(meld_csv_glob):
        split = Path(csv_path).stem.lower()
        df = pd.read_csv(csv_path)
        cols = {c.lower(): c for c in df.columns}
        did_col = cols.get("dialogue_id") or cols.get("old_dialogue_id")
        uid_col = cols.get("utterance_id") or cols.get("old_utterance_id")
        spk_col = cols.get("speaker")
        utt_col = cols.get("utterance")
        emo_col = cols.get("emotion")
        sen_col = cols.get("sentiment")
        if did_col is None or uid_col is None:
            continue
        for _, row in df.iterrows():
            key = (int(row[did_col]), int(row[uid_col]))
            meta[key] = MeldClipMeta(
                split=split,
                dialogue_id=int(row[did_col]),
                utterance_id=int(row[uid_col]),
                speaker=str(row[spk_col]) if spk_col else "",
                utterance=str(row[utt_col]) if utt_col else "",
                emotion=str(row[emo_col]) if emo_col else "",
                sentiment=str(row[sen_col]) if sen_col else "",
            )
    return meta


# =========================================================
# helper utils
# =========================================================
def run_cmd(cmd: str):
    subprocess.run(cmd, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def normalize_text(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9\s']+", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s


def text_similarity(a: str, b: str) -> float:
    a_n = normalize_text(a)
    b_n = normalize_text(b)
    if not a_n and not b_n:
        return 1.0
    return difflib.SequenceMatcher(None, a_n, b_n).ratio()


def moving_average(arr: np.ndarray, k: int) -> np.ndarray:
    if len(arr) == 0:
        return arr
    if k <= 1:
        return arr.copy()
    k = max(1, int(k))
    pad = k // 2
    x = np.pad(arr, (pad, pad), mode="edge")
    kernel = np.ones(k, dtype=np.float32) / float(k)
    return np.convolve(x, kernel, mode="valid")


def expand_bbox(x1, y1, x2, y2, w, h, ratio: float):
    bw = x2 - x1
    bh = y2 - y1
    ex = bw * ratio
    ey = bh * ratio
    nx1 = max(0, int(x1 - ex))
    ny1 = max(0, int(y1 - ey))
    nx2 = min(w - 1, int(x2 + ex))
    ny2 = min(h - 1, int(y2 + ey))
    return nx1, ny1, nx2, ny2


def bb_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter_area = max(0, xB - xA) * max(0, yB - yA)
    boxA_area = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    boxB_area = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])
    denom = boxA_area + boxB_area - inter_area + 1e-6
    return inter_area / denom


def dilate_binary_mask(mask: np.ndarray, px: int) -> np.ndarray:
    if px <= 0:
        return mask
    k = max(1, int(px))
    kernel = np.ones((k, k), np.uint8)
    out = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
    return out.astype(bool)


def resolve_logs_dir(output_dir: Path) -> Path:
    if output_dir.name.lower() in {"train", "dev", "test"}:
        return output_dir.parent / "logs"
    return output_dir / "logs"


def infer_split_name(output_dir: Path) -> str:
    if output_dir.name.lower() in {"train", "dev", "test"}:
        return output_dir.name.lower()
    return "all"


# =========================================================
# scene detection / face detection / tracking
# =========================================================
def scene_detect(video_file: Path, pywork_path: Path):
    video_manager = VideoManager([str(video_file)])
    stats_manager = StatsManager()
    scene_manager = SceneManager(stats_manager)
    scene_manager.add_detector(ContentDetector())
    base_timecode = video_manager.get_base_timecode()
    video_manager.set_downscale_factor()
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)
    scene_list = scene_manager.get_scene_list(base_timecode)
    if scene_list == []:
        scene_list = [(video_manager.get_base_timecode(), video_manager.get_current_timecode())]
    save_path = pywork_path / "scene.pckl"
    with open(save_path, "wb") as f:
        pickle.dump(scene_list, f)
    return scene_list


def inference_video_s3fd(pyframes_path: Path, pywork_path: Path, S3FD_cls, cfg: PipelineConfig):
    det = S3FD_cls(device=cfg.device)
    flist = sorted(glob.glob(str(pyframes_path / "*.jpg")))
    dets = []
    for fidx, fname in enumerate(tqdm.tqdm(flist, desc="face detect", leave=False)):
        image = cv2.imread(fname)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        bboxes = det.detect_faces(image_rgb, conf_th=cfg.face_conf, scales=[cfg.facedet_scale])
        dets.append([])
        for bbox in bboxes:
            dets[-1].append({
                "frame": fidx,
                "bbox": (bbox[:-1]).tolist(),
                "conf": float(bbox[-1]),
            })
    save_path = pywork_path / "faces.pckl"
    with open(save_path, "wb") as f:
        pickle.dump(dets, f)
    return dets


def track_shot(scene_faces, cfg: PipelineConfig):
    iou_thres = 0.5
    tracks = []
    while True:
        track = []
        for frame_faces in scene_faces:
            for face in list(frame_faces):
                if track == []:
                    track.append(face)
                    frame_faces.remove(face)
                elif face["frame"] - track[-1]["frame"] <= cfg.num_failed_det:
                    iou = bb_iou(face["bbox"], track[-1]["bbox"])
                    if iou > iou_thres:
                        track.append(face)
                        frame_faces.remove(face)
                        continue
                else:
                    break
        if track == []:
            break
        elif len(track) > cfg.min_track:
            frame_num = np.array([f["frame"] for f in track])
            bboxes = np.array([np.array(f["bbox"]) for f in track])
            frame_i = np.arange(frame_num[0], frame_num[-1] + 1)
            bboxes_i = []
            for ij in range(4):
                interp_fn = interp1d(frame_num, bboxes[:, ij])
                bboxes_i.append(interp_fn(frame_i))
            bboxes_i = np.stack(bboxes_i, axis=1)
            face_size = max(np.mean(bboxes_i[:, 2] - bboxes_i[:, 0]), np.mean(bboxes_i[:, 3] - bboxes_i[:, 1]))
            if face_size > cfg.min_face_size:
                tracks.append({"frame": frame_i, "bbox": bboxes_i})
    return tracks


def crop_video(cfg: PipelineConfig, track: dict, crop_file_stem: Path, pyframes_path: Path, audio_file_path: Path):
    flist = sorted(glob.glob(str(pyframes_path / "*.jpg")))
    v_out = cv2.VideoWriter(str(crop_file_stem) + "t.avi", cv2.VideoWriter_fourcc(*"XVID"), cfg.fps, (224, 224))

    dets = {"x": [], "y": [], "s": []}
    for det in track["bbox"]:
        dets["s"].append(max((det[3] - det[1]), (det[2] - det[0])) / 2)
        dets["y"].append((det[1] + det[3]) / 2)
        dets["x"].append((det[0] + det[2]) / 2)

    dets["s"] = signal.medfilt(dets["s"], kernel_size=13)
    dets["x"] = signal.medfilt(dets["x"], kernel_size=13)
    dets["y"] = signal.medfilt(dets["y"], kernel_size=13)

    for fidx, frame in enumerate(track["frame"]):
        cs = cfg.crop_scale
        bs = dets["s"][fidx]
        bsi = int(bs * (1 + 2 * cs))
        image = cv2.imread(flist[frame])
        frame_pad = np.pad(image, ((bsi, bsi), (bsi, bsi), (0, 0)), "constant", constant_values=(110, 110))
        my = dets["y"][fidx] + bsi
        mx = dets["x"][fidx] + bsi
        face = frame_pad[int(my - bs): int(my + bs * (1 + 2 * cs)), int(mx - bs * (1 + cs)): int(mx + bs * (1 + cs))]
        v_out.write(cv2.resize(face, (224, 224)))
    v_out.release()

    audio_tmp = str(crop_file_stem) + ".wav"
    audio_start = track["frame"][0] / cfg.fps
    audio_end = (track["frame"][-1] + 1) / cfg.fps
    run_cmd(
        f"ffmpeg -y -i {audio_file_path} -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 "
        f"-threads {cfg.num_workers} -ss {audio_start:.3f} -to {audio_end:.3f} {audio_tmp} -loglevel panic"
    )
    run_cmd(
        f"ffmpeg -y -i {str(crop_file_stem)}t.avi -i {audio_tmp} -threads {cfg.num_workers} "
        f"-c:v copy -c:a copy {str(crop_file_stem)}.avi -loglevel panic"
    )
    os.remove(str(crop_file_stem) + "t.avi")
    return {"track": track, "proc_track": dets}


# =========================================================
# TalkNet ASD scoring
# =========================================================
def evaluate_network_talknet(files: List[str], talkNet_cls, checkpoint_path: str):
    s = talkNet_cls()
    s.loadParameters(checkpoint_path)
    s.eval()

    all_scores = []
    duration_set = {1, 1, 1, 2, 2, 2, 3, 3, 4, 5, 6}

    for file in tqdm.tqdm(files, desc="talknet asd", leave=False):
        wav_path = os.path.splitext(file)[0] + ".wav"
        _, audio = wavfile.read(wav_path)
        audio_feature = python_speech_features.mfcc(audio, 16000, numcep=13, winlen=0.025, winstep=0.010)

        video = cv2.VideoCapture(file)
        video_feature = []
        while video.isOpened():
            ret, frames = video.read()
            if not ret:
                break
            face = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
            face = cv2.resize(face, (224, 224))
            face = face[int(112 - (112 / 2)): int(112 + (112 / 2)), int(112 - (112 / 2)): int(112 + (112 / 2))]
            video_feature.append(face)
        video.release()
        video_feature = np.array(video_feature)

        length = min((audio_feature.shape[0] - audio_feature.shape[0] % 4) / 100, video_feature.shape[0] / 25)
        audio_feature = audio_feature[: int(round(length * 100)), :]
        video_feature = video_feature[: int(round(length * 25)), :, :]

        score_ensembles = []
        for duration in duration_set:
            batch_size = int(math.ceil(length / duration))
            scores = []
            with torch.no_grad():
                for i in range(batch_size):
                    inputA = torch.FloatTensor(audio_feature[i * duration * 100:(i + 1) * duration * 100, :]).unsqueeze(0).cuda()
                    inputV = torch.FloatTensor(video_feature[i * duration * 25:(i + 1) * duration * 25, :, :]).unsqueeze(0).cuda()
                    embedA = s.model.forward_audio_frontend(inputA)
                    embedV = s.model.forward_visual_frontend(inputV)
                    embedA, embedV = s.model.forward_cross_attention(embedA, embedV)
                    out = s.model.forward_audio_visual_backend(embedA, embedV)
                    score = s.lossAV.forward(out, labels=None)
                    scores.extend(score)
            score_ensembles.append(scores)
        final_score = np.round(np.mean(np.array(score_ensembles), axis=0), 1).astype(float)
        all_scores.append(final_score)
    return all_scores


# =========================================================
# WhisperX
# =========================================================
def run_whisperx(audio_path: Path, cfg: PipelineConfig) -> dict:
    if not cfg.use_whisperx:
        return {"text": "", "segments": [], "words": [], "diarization": []}

    try:
        import whisperx  # type: ignore
    except Exception as e:
        raise RuntimeError(f"whisperx import 실패: {type(e).__name__}: {e}") from e

    device = cfg.device if torch.cuda.is_available() and cfg.device.startswith("cuda") else "cpu"
    compute_type = cfg.whisper_compute_type if device == "cuda" else "int8"

    model = whisperx.load_model(cfg.whisper_model, device, compute_type=compute_type)
    audio = whisperx.load_audio(str(audio_path))
    result = model.transcribe(audio, batch_size=cfg.whisper_batch_size)

    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

    diarization_segments = []
    if cfg.use_diarization and cfg.hf_token:
        from whisperx.diarize import DiarizationPipeline
        diarize_model = DiarizationPipeline(use_auth_token=cfg.hf_token, device=device)
        diarization_segments = diarize_model(audio)
        result = whisperx.assign_word_speakers(diarization_segments, result)

    full_text = " ".join([seg.get("text", "") for seg in result.get("segments", [])]).strip()
    words = []
    for seg in result.get("segments", []):
        for w in seg.get("words", []):
            if "start" in w and "end" in w:
                words.append(w)

    del model
    try:
        del model_a
    except Exception:
        pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "text": full_text,
        "segments": result.get("segments", []),
        "words": words,
        "diarization": diarization_segments,
        "language": result.get("language", "en"),
    }


def build_speech_mask(n_frames: int, fps: int, asr_result: dict) -> np.ndarray:
    speech = np.zeros(n_frames, dtype=np.float32)
    words = asr_result.get("words", []) or []
    if words:
        for w in words:
            s = max(0, int(float(w["start"]) * fps))
            e = min(n_frames, int(math.ceil(float(w["end"]) * fps)))
            speech[s:e] = 1.0
    else:
        for seg in asr_result.get("segments", []) or []:
            if "start" in seg and "end" in seg:
                s = max(0, int(float(seg["start"]) * fps))
                e = min(n_frames, int(math.ceil(float(seg["end"]) * fps)))
                speech[s:e] = 1.0
    if speech.sum() > 0:
        speech = moving_average(speech, max(3, int(0.12 * fps)))
        speech = (speech > 0.15).astype(np.float32)
    return speech


# =========================================================
# fusion + Viterbi decoding
# =========================================================
def build_track_table(tracks: List[dict], scores: List[np.ndarray], n_frames: int, smooth_k: int) -> pd.DataFrame:
    rows = []
    for tidx, track in enumerate(tracks):
        score = np.asarray(scores[tidx], dtype=np.float32)
        score = moving_average(score, smooth_k)
        frames = track["track"]["frame"].tolist()
        for fidx, frame in enumerate(frames):
            if fidx >= len(score):
                continue
            rows.append({
                "frame": int(frame),
                "track_id": tidx,
                "score": float(score[fidx]),
                "s": float(track["proc_track"]["s"][fidx]),
                "x": float(track["proc_track"]["x"][fidx]),
                "y": float(track["proc_track"]["y"][fidx]),
            })
    if not rows:
        return pd.DataFrame(columns=["frame", "track_id", "score", "s", "x", "y"])
    df = pd.DataFrame(rows)
    df = df[df["frame"] < n_frames].copy()
    return df


def decode_active_speaker(
    track_df: pd.DataFrame,
    n_tracks: int,
    n_frames: int,
    speech_mask: np.ndarray,
    transcript_similarity_score: float,
    cfg: PipelineConfig,
):
    null_state = n_tracks
    n_states = n_tracks + 1

    emission = np.full((n_frames, n_states), -10.0, dtype=np.float32)
    emission[:, null_state] = cfg.null_bias

    for _, row in track_df.iterrows():
        f = int(row["frame"])
        t = int(row["track_id"])
        s = float(row["score"])
        s = s + cfg.transcript_match_bonus * transcript_similarity_score
        if speech_mask[f] > 0:
            s = s + cfg.speech_bonus
        else:
            s = s - cfg.silence_penalty
        emission[f, t] = max(emission[f, t], s)

    for f in range(n_frames):
        if speech_mask[f] > 0:
            emission[f, null_state] -= 1.0
        else:
            emission[f, null_state] += 0.25

    dp = np.full((n_frames, n_states), -1e9, dtype=np.float32)
    back = np.full((n_frames, n_states), -1, dtype=np.int32)

    dp[0] = emission[0]
    for f in range(1, n_frames):
        for cur in range(n_states):
            best_score = -1e9
            best_prev = -1
            for prev in range(n_states):
                trans = 0.0 if prev == cur else -cfg.switch_penalty
                if prev == null_state or cur == null_state:
                    trans += 0.25
                cand = dp[f - 1, prev] + trans + emission[f, cur]
                if cand > best_score:
                    best_score = cand
                    best_prev = prev
            dp[f, cur] = best_score
            back[f, cur] = best_prev

    states = [int(np.argmax(dp[-1]))]
    for f in range(n_frames - 1, 0, -1):
        states.append(int(back[f, states[-1]]))
    states = states[::-1]
    states = np.asarray(states, dtype=np.int32)
    return states, emission


# =========================================================
# person segmentation + render
# =========================================================
def face_row_to_box(row: pd.Series, width: int, height: int, expand_ratio: float = 0.10) -> Tuple[int, int, int, int]:
    s = float(row["s"])
    x = float(row["x"])
    y = float(row["y"])
    x1 = int(x - s)
    y1 = int(y - s)
    x2 = int(x + s)
    y2 = int(y + s)
    return expand_bbox(x1, y1, x2, y2, width, height, expand_ratio)


def fallback_body_mask_from_face(row: pd.Series, width: int, height: int, cfg: PipelineConfig) -> np.ndarray:
    s = float(row["s"])
    x = float(row["x"])
    y = float(row["y"])
    x1 = max(0, int(x - cfg.body_fallback_expand_x * s))
    x2 = min(width, int(x + cfg.body_fallback_expand_x * s))
    y1 = max(0, int(y - cfg.body_fallback_expand_top * s))
    y2 = min(height, int(y + cfg.body_fallback_expand_bottom * s))
    mask = np.zeros((height, width), dtype=bool)
    mask[y1:y2, x1:x2] = True
    return mask


def choose_active_person_mask(frame: np.ndarray, row: pd.Series, seg_result, cfg: PipelineConfig):
    h, w = frame.shape[:2]
    face_box = face_row_to_box(row, w, h, expand_ratio=0.10)
    fx1, fy1, fx2, fy2 = face_box
    cx = int((fx1 + fx2) / 2)
    cy = int((fy1 + fy2) / 2)

    if seg_result is None or getattr(seg_result, "masks", None) is None or seg_result.masks is None:
        return fallback_body_mask_from_face(row, w, h, cfg), None, True

    masks_data = seg_result.masks.data
    if masks_data is None or len(masks_data) == 0:
        return fallback_body_mask_from_face(row, w, h, cfg), None, True

    masks = masks_data.cpu().numpy() > 0.5
    boxes_xyxy = seg_result.boxes.xyxy.cpu().numpy() if seg_result.boxes is not None else np.zeros((0, 4), dtype=np.float32)
    confs = seg_result.boxes.conf.cpu().numpy() if seg_result.boxes is not None else np.zeros((0,), dtype=np.float32)

    best_idx = -1
    best_score = -1e9
    for idx in range(len(masks)):
        mask = masks[idx]
        bx1, by1, bx2, by2 = boxes_xyxy[idx]
        inside = 0.0
        if 0 <= cy < mask.shape[0] and 0 <= cx < mask.shape[1] and mask[cy, cx]:
            inside = 1.0
        face_iou = bb_iou(face_box, (bx1, by1, bx2, by2))
        bcx = 0.5 * (bx1 + bx2)
        bcy = 0.5 * (by1 + by2)
        dist = math.hypot(bcx - cx, bcy - cy) / max(1.0, math.hypot(w, h))
        box_area = max(0.0, (bx2 - bx1) * (by2 - by1)) / max(1.0, float(w * h))
        score = 4.0 * inside + 2.5 * face_iou + 0.20 * float(confs[idx]) + 0.25 * min(box_area * 5.0, 1.0) - 1.50 * dist
        if score > best_score:
            best_score = score
            best_idx = idx

    if best_idx < 0:
        return fallback_body_mask_from_face(row, w, h, cfg), None, True

    chosen_mask = masks[best_idx]
    chosen_mask = dilate_binary_mask(chosen_mask, cfg.mask_dilate_px)
    chosen_box = boxes_xyxy[best_idx]
    return chosen_mask, chosen_box, False


def compose_foreground_black(frame: np.ndarray, person_mask: np.ndarray) -> np.ndarray:
    out = np.zeros_like(frame)
    out[person_mask] = frame[person_mask]
    return out


def render_outputs(
    render_video: Path,
    audio_source: Path,
    output_video: Path,
    track_df: pd.DataFrame,
    states: np.ndarray,
    cfg: PipelineConfig,
    person_seg_model,
):
    cap = cv2.VideoCapture(str(render_video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(round(cap.get(cv2.CAP_PROP_FPS))) or cfg.fps

    output_video.parent.mkdir(parents=True, exist_ok=True)
    tmp_video_only = output_video.with_suffix(".video_only.avi")
    writer = cv2.VideoWriter(str(tmp_video_only), cv2.VideoWriter_fourcc(*"XVID"), fps, (width, height))

    per_frame: Dict[int, List[pd.Series]] = {}
    for _, row in track_df.iterrows():
        f = int(row["frame"])
        per_frame.setdefault(f, []).append(row)

    n_tracks = int(track_df["track_id"].max()) + 1 if len(track_df) > 0 else 0

    last_seg_state = None
    last_seg_mask = None
    last_seg_box = None
    last_seg_frame_idx = -999999

    frames_rendered = 0
    blackout_frames = 0
    valid_active_speaker_visible_frames = 0
    fallback_body_visible_frames = 0

    for fidx in tqdm.tqdm(range(total), desc="render", leave=False):
        ret, frame = cap.read()
        if not ret:
            break

        frames_rendered += 1
        chosen_state = int(states[fidx]) if fidx < len(states) else -1

        rows = per_frame.get(fidx, [])
        chosen_row = None
        for row in rows:
            if int(row["track_id"]) == chosen_state:
                chosen_row = row
                break

        if chosen_row is None or chosen_state < 0 or chosen_state >= n_tracks:
            blackout_frames += 1
            masked_frame = np.zeros_like(frame)
        else:
            need_new_seg = True
            if (
                cfg.person_seg_stride > 1
                and last_seg_state == chosen_state
                and (fidx - last_seg_frame_idx) < cfg.person_seg_stride
                and last_seg_mask is not None
            ):
                need_new_seg = False

            if need_new_seg:
                seg_result = person_seg_model.predict(
                    source=frame,
                    verbose=False,
                    device=cfg.device,
                    classes=[0],
                    conf=cfg.person_seg_conf,
                    iou=cfg.person_seg_iou,
                    imgsz=cfg.person_seg_imgsz,
                    retina_masks=True,
                )[0]
                person_mask, chosen_person_box, used_fallback = choose_active_person_mask(frame, chosen_row, seg_result, cfg)
                last_seg_mask = person_mask
                last_seg_box = chosen_person_box
                last_seg_state = chosen_state
                last_seg_frame_idx = fidx
            else:
                person_mask = last_seg_mask
                used_fallback = False

            masked_frame = compose_foreground_black(frame, person_mask)
            valid_active_speaker_visible_frames += 1
            if used_fallback:
                fallback_body_visible_frames += 1

        writer.write(masked_frame)

    cap.release()
    writer.release()

    run_cmd(
        f"ffmpeg -y -i {tmp_video_only} -i {audio_source} -map 0:v:0 -map 1:a:0? -c:v copy -c:a aac {output_video} -loglevel panic"
    )
    os.remove(tmp_video_only)

    denom = float(frames_rendered) if frames_rendered > 0 else 1.0
    return {
        "frames_rendered": frames_rendered,
        "blackout_frames": blackout_frames,
        "blackout_ratio": blackout_frames / denom,
        "valid_active_speaker_visible_frames": valid_active_speaker_visible_frames,
        "valid_active_speaker_visible_ratio": valid_active_speaker_visible_frames / denom,
        "fallback_body_visible_frames": fallback_body_visible_frames,
        "fallback_body_visible_ratio": fallback_body_visible_frames / denom,
    }


# =========================================================
# per-video pipeline
# =========================================================
def prepare_workdirs(base_out: Path, video_stem: str, reset: bool):
    """임시 작업 디렉토리. 처리 완료 후 삭제됨."""
    import tempfile
    work_root = Path(tempfile.mkdtemp(prefix=f"meld_{video_stem}_"))
    pyavi = work_root / "pyavi"
    pyframes = work_root / "pyframes"
    pywork = work_root / "pywork"
    pycrop = work_root / "pycrop"
    pyavi.mkdir(parents=True, exist_ok=True)
    pyframes.mkdir(parents=True, exist_ok=True)
    pywork.mkdir(parents=True, exist_ok=True)
    pycrop.mkdir(parents=True, exist_ok=True)
    return work_root, pyavi, pyframes, pywork, pycrop


def extract_video_audio_frames(input_video: Path, pyavi: Path, pyframes: Path, cfg: PipelineConfig):
    video_file = pyavi / "video.avi"
    audio_file = pyavi / "audio.wav"
    run_cmd(
        f"ffmpeg -y -i {input_video} -qscale:v 2 -threads {cfg.num_workers} -async 1 -r {cfg.fps} {video_file} -loglevel panic"
    )
    run_cmd(
        f"ffmpeg -y -i {video_file} -qscale:v 2 -threads {cfg.num_workers} {pyframes / '%06d.jpg'} -loglevel panic"
    )
    run_cmd(
        f"ffmpeg -y -i {video_file} -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 {audio_file} -loglevel panic"
    )
    return video_file, audio_file


def classify_status(num_tracks: int, selected_ratio: float, mean_speech: float, mean_track_score: float) -> str:
    if num_tracks == 0:
        return "no_speaker_track"
    if selected_ratio < 0.15:
        return "low_selected_ratio"
    if mean_speech > 0.10 and selected_ratio < 0.30:
        return "ambiguous_speaker"
    if selected_ratio < 0.40:
        return "suspicious_ok"
    if mean_track_score < -0.5:
        return "suspicious_ok"
    return "ok"


def process_one_video(
    video_path: Path,
    out_root: Path,
    cfg: PipelineConfig,
    meld_meta: Dict[Tuple[int, int], MeldClipMeta],
    S3FD_cls,
    talkNet_cls,
    checkpoint_path: Path,
    person_seg_model,
) -> dict:
    stem = video_path.stem
    dia_id, utt_id = parse_meld_filename(stem)
    clip_meta = meld_meta.get((dia_id, utt_id)) if dia_id is not None and utt_id is not None else None

    # 임시 작업 디렉토리 (처리 후 자동 삭제)
    work_root, pyavi, pyframes, pywork, pycrop = prepare_workdirs(out_root, stem, cfg.reset_output)

    try:
        video_file, audio_file = extract_video_audio_frames(video_path, pyavi, pyframes, cfg)

        scene_list = scene_detect(video_file, pywork)
        dets = inference_video_s3fd(pyframes, pywork, S3FD_cls, cfg)

        all_tracks = []
        crop_files = []
        for shot in scene_list:
            start, end = shot[0].frame_num, shot[1].frame_num
            scene_faces = copy.deepcopy(dets[start:end])
            shot_tracks = track_shot(scene_faces, cfg)
            for tr in shot_tracks:
                crop_stem = pycrop / ("%05d" % len(all_tracks))
                out_track = crop_video(cfg, tr, crop_stem, pyframes, audio_file)
                all_tracks.append(out_track)
                crop_files.append(str(crop_stem) + ".avi")

        # pyframes는 crop 끝나면 더 이상 불필요 -> 즉시 삭제
        shutil.rmtree(pyframes, ignore_errors=True)

        talk_scores = evaluate_network_talknet(crop_files, talkNet_cls, str(checkpoint_path)) if crop_files else []

        cap = cv2.VideoCapture(str(video_file))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(round(cap.get(cv2.CAP_PROP_FPS))) or cfg.fps
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        asr = run_whisperx(audio_file, cfg) if cfg.use_whisperx else {"text": "", "segments": [], "words": [], "diarization": []}
        speech_mask = build_speech_mask(n_frames, fps, asr)

        meta_text = clip_meta.utterance if clip_meta else ""
        whisper_text = asr.get("text", "")
        transcript_sim = text_similarity(meta_text, whisper_text) if meta_text else 0.0

        track_df = build_track_table(all_tracks, talk_scores, n_frames, cfg.talknet_smooth)
        states, emission = decode_active_speaker(
            track_df=track_df,
            n_tracks=len(all_tracks),
            n_frames=n_frames,
            speech_mask=speech_mask,
            transcript_similarity_score=transcript_sim,
            cfg=cfg,
        )

        output_video = out_root / video_path.name

        render_stats = render_outputs(
            render_video=video_file,
            audio_source=video_path,
            output_video=output_video,
            track_df=track_df,
            states=states,
            cfg=cfg,
            person_seg_model=person_seg_model,
        )

        selected_ratio = float(np.mean(states < len(all_tracks))) if len(states) > 0 else 0.0
        mean_speech = float(np.mean(speech_mask)) if len(speech_mask) > 0 else 0.0
        mean_track_score = float(track_df["score"].mean()) if len(track_df) > 0 else 0.0
        status = classify_status(len(all_tracks), selected_ratio, mean_speech, mean_track_score)

        log = {
            "file": video_path.name,
            "input_path": str(video_path),
            "output_path": str(output_video),
            "status": status,
            "dialogue_id": dia_id,
            "utterance_id": utt_id,
            "meld_speaker": clip_meta.speaker if clip_meta else "",
            "meld_utterance": clip_meta.utterance if clip_meta else "",
            "whisperx_text": whisper_text,
            "transcript_similarity": transcript_sim,
            "num_tracks_total": len(all_tracks),
            "frames_total": n_frames,
            "frames_rendered": render_stats["frames_rendered"],
            "fps": fps,
            "width": width,
            "height": height,
            "speech_active_ratio": mean_speech,
            "selected_frame_ratio": selected_ratio,
            "mean_track_score": mean_track_score,
            "blackout_frames": render_stats["blackout_frames"],
            "blackout_ratio": render_stats["blackout_ratio"],
            "valid_active_speaker_visible_frames": render_stats["valid_active_speaker_visible_frames"],
            "valid_active_speaker_visible_ratio": render_stats["valid_active_speaker_visible_ratio"],
            "fallback_body_visible_frames": render_stats["fallback_body_visible_frames"],
            "fallback_body_visible_ratio": render_stats["fallback_body_visible_ratio"],
        }

    finally:
        # 성공/실패 관계없이 임시 작업 디렉토리 삭제
        shutil.rmtree(work_root, ignore_errors=True)

    return log


# =========================================================
# batch driver
# =========================================================
def find_videos(input_dir: Path) -> List[Path]:
    videos = []
    for ext in ("*.mp4", "*.avi", "*.mov", "*.mkv"):
        videos.extend(f for f in sorted(input_dir.glob(ext)) if not f.name.startswith("._"))
    return sorted(videos)


def write_logs_csv(logs: List[dict], csv_path: Path):
    if not logs:
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    cols = []
    for row in logs:
        for k in row.keys():
            if k not in cols:
                cols.append(k)
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for row in logs:
            writer.writerow(row)


def main():
    ap = argparse.ArgumentParser(description="MELD active-speaker foreground-preserving person masking")
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--talknet_repo", required=True)
    ap.add_argument("--pretrain_model", default="pretrain_TalkSet.model")
    ap.add_argument("--meld_csv_glob", default="")
    ap.add_argument("--hf_token", default="")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--reset_output", action="store_true")
    ap.add_argument("--disable_whisperx", action="store_true")
    ap.add_argument("--disable_diarization", action="store_true")
    ap.add_argument("--person_seg_model", default="yolo11n-seg.pt")
    ap.add_argument("--person_seg_conf", type=float, default=0.25)
    ap.add_argument("--person_seg_iou", type=float, default=0.50)
    ap.add_argument("--person_seg_imgsz", type=int, default=640)
    ap.add_argument("--person_seg_stride", type=int, default=1)
    ap.add_argument("--shutdown_when_done", action="store_true",
                    help="모든 처리 완료 후 서버 shutdown (sudo 권한 필요)")
    args = ap.parse_args()

    cfg = PipelineConfig(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        talknet_repo=args.talknet_repo,
        pretrain_model=args.pretrain_model,
        meld_csv_glob=args.meld_csv_glob,
        hf_token=args.hf_token,
        device=args.device,
        limit=args.limit,
        reset_output=args.reset_output,
        use_whisperx=not args.disable_whisperx,
        use_diarization=not args.disable_diarization,
        person_seg_model=args.person_seg_model,
        person_seg_conf=args.person_seg_conf,
        person_seg_iou=args.person_seg_iou,
        person_seg_imgsz=args.person_seg_imgsz,
        person_seg_stride=max(1, int(args.person_seg_stride)),
    )

    ensure_ffmpeg()
    checkpoint_path = Path(cfg.pretrain_model)
    if not checkpoint_path.is_absolute():
        checkpoint_path = Path(cfg.talknet_repo).resolve() / checkpoint_path
    maybe_download_talkset_checkpoint(checkpoint_path)

    S3FD_cls, talkNet_cls = ensure_talknet_imports(cfg.talknet_repo)
    YOLO = ensure_ultralytics_imports()
    person_seg_model = YOLO(cfg.person_seg_model)

    input_dir = Path(cfg.input_dir)
    out_root = Path(cfg.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    meld_meta = load_meld_metadata(cfg.meld_csv_glob)
    videos = find_videos(input_dir)
    if cfg.limit and cfg.limit > 0:
        videos = videos[:cfg.limit]

    print(f"[videos] {len(videos)} clips found")
    print(f"[metadata] {len(meld_meta)} MELD metadata rows loaded")

    logs = []
    skipped = 0
    for idx, video in enumerate(videos, start=1):
        # ── resume: 이미 output이 있으면 스킵 ──
        output_video = out_root / video.name
        if output_video.exists():
            skipped += 1
            print(f"[{idx:04d}/{len(videos):04d}] skip (done): {video.name}")
            continue

        print(f"\n[{idx:04d}/{len(videos):04d}] {video.name}")
        try:
            log = process_one_video(
                video_path=video,
                out_root=out_root,
                cfg=cfg,
                meld_meta=meld_meta,
                S3FD_cls=S3FD_cls,
                talkNet_cls=talkNet_cls,
                checkpoint_path=checkpoint_path,
                person_seg_model=person_seg_model,
            )
            logs.append(log)
            print(
                f"  -> status={log['status']} | tracks={log['num_tracks_total']} | "
                f"selected_ratio={log['selected_frame_ratio']:.3f} | "
                f"blackout={log['blackout_ratio']:.2f} | output={Path(log['output_path']).name}"
            )
        except Exception as e:
            err = {
                "file": video.name,
                "input_path": str(video),
                "output_path": "",
                "status": "exception",
                "error": str(e),
            }
            logs.append(err)
            print(f"  !! exception: {e}")

    logs_dir = resolve_logs_dir(out_root)
    split_name = infer_split_name(out_root)
    csv_path = logs_dir / f"{split_name}_pipeline_log.csv"
    write_logs_csv(logs, csv_path)
    print(f"\n[done] processed={len(logs)} skipped={skipped} | log -> {csv_path}")

    if args.shutdown_when_done:
        print("[shutdown] 5초 후 서버를 종료합니다...")
        subprocess.run("sleep 5 && sudo shutdown -h now", shell=True)


if __name__ == "__main__":
    main()