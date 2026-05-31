#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
iemocap_vm.py

IEMOCAP active-speaker foreground-preserving intervention pipeline
MELD 버전에서 IEMOCAP용으로 수정
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
    input_dir: str
    output_dir: str
    talknet_repo: str
    iemocap_session_dir: str = ""       # Session5/ 경로 (메타데이터 로딩용)
    pretrain_model: str = "pretrain_TalkSet.model"

    device: str = "cuda"
    num_workers: int = 8

    facedet_scale: float = 0.25
    face_conf: float = 0.90
    min_track: int = 10
    num_failed_det: int = 10
    min_face_size: int = 1
    crop_scale: float = 0.40

    fps: int = 25
    talknet_smooth: int = 5
    switch_penalty: float = 1.25
    null_bias: float = 0.25
    speech_bonus: float = 0.75
    silence_penalty: float = 1.5
    transcript_match_bonus: float = 0.20

    use_whisperx: bool = True
    whisper_model: str = "small"
    whisper_batch_size: int = 8
    whisper_compute_type: str = "float16"
    hf_token: str = ""
    use_diarization: bool = True

    person_seg_model: str = "yolo11n-seg.pt"
    person_seg_conf: float = 0.25
    person_seg_iou: float = 0.50
    person_seg_imgsz: int = 640
    person_seg_stride: int = 1
    mask_dilate_px: int = 7
    body_fallback_expand_x: float = 2.2
    body_fallback_expand_top: float = 1.6
    body_fallback_expand_bottom: float = 5.2

    reset_output: bool = False
    limit: int = 0


# =========================================================
# ★ IEMOCAP 전용: 파일명 파서
# =========================================================
@dataclass
class IemocapClipMeta:
    utt_name: str       # e.g. Ses05F_impro01_F000
    dialog_name: str    # e.g. Ses05F_impro01
    speaker: str        # F or M
    utterance: str      # transcription 텍스트
    emotion: str        # categorical emotion


def parse_iemocap_filename(stem: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Ses05F_impro01_F000 → (dialog_name, speaker, utt_name)
    FXX 계열은 None 반환 → 스킵
    """
    m = re.match(r"(Ses\d+[FM]_(?:impro|script)\w+)_([FM])(XX\d+|\d+)$", stem)
    if not m:
        return None, None, None
    dialog_name = m.group(1)
    speaker     = m.group(2)
    utt_idx     = m.group(3)
    if utt_idx.startswith("XX"):
        return None, None, None   # FXX, MXX → 불명확 발화, 스킵
    return dialog_name, speaker, stem


# =========================================================
# ★ IEMOCAP 전용: 메타데이터 로더
# =========================================================
def load_iemocap_metadata(session_dir: str) -> Dict[str, IemocapClipMeta]:
    """
    EmoEvaluation + transcriptions 에서 utterance별 메타데이터 로드
    """
    meta: Dict[str, IemocapClipMeta] = {}
    if not session_dir:
        return meta

    session_path = Path(session_dir)

    # ── 1. 감정 레이블: EmoEvaluation/Categorical/ ──
    emo_map: Dict[str, str] = {}
    emo_dir = session_path / "dialog" / "EmoEvaluation" / "Categorical"
    if not emo_dir.exists():
        # fallback: EmoEvaluation/ 바로 아래 txt
        emo_dir = session_path / "dialog" / "EmoEvaluation"

    for emo_file in sorted(emo_dir.glob("*.txt")):
        if emo_file.name.startswith("._"):
            continue
        with open(emo_file, encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                # 예: [6.29 - 8.24]  Ses05F_impro01_F000  neu  [2.5, 2.5, 2.5]
                m = re.match(r"\[.*?\]\s+(Ses\w+)\s+(\w+)", line)
                if m:
                    utt_name = m.group(1)
                    emotion  = m.group(2)
                    if emotion != "xxx":
                        emo_map[utt_name] = emotion

    # ── 2. 전사 텍스트: dialog/transcriptions/ ──
    trans_map: Dict[str, str] = {}
    trans_dir = session_path / "dialog" / "transcriptions"
    for trans_file in sorted(trans_dir.glob("*.txt")):
        if trans_file.name.startswith("._"):
            continue
        with open(trans_file, encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                # 예: Ses05F_impro01_F000 [6.29-8.24]: I don't know what to say.
                m = re.match(r"(Ses\w+)\s+\[.*?\]:\s*(.*)", line)
                if m:
                    utt_name = m.group(1)
                    text     = m.group(2).strip()
                    trans_map[utt_name] = text

    # ── 3. 합치기 ──
    all_utts = set(emo_map.keys()) | set(trans_map.keys())
    for utt_name in all_utts:
        dm, spk, _ = parse_iemocap_filename(utt_name)
        if dm is None:
            continue
        meta[utt_name] = IemocapClipMeta(
            utt_name    = utt_name,
            dialog_name = dm,
            speaker     = spk or "",
            utterance   = trans_map.get(utt_name, ""),
            emotion     = emo_map.get(utt_name, ""),
        )

    print(f"[metadata] IEMOCAP 메타데이터 {len(meta)}개 로드 완료")
    return meta


# =========================================================
# environment / repo loading  (MELD 버전과 동일)
# =========================================================
def ensure_talknet_imports(repo_path: str):
    repo = Path(repo_path).resolve()
    if not repo.exists():
        raise FileNotFoundError(f"TalkNet repo가 없습니다: {repo}")
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    os.chdir(str(repo))
    try:
        from model.faceDetector.s3fd import S3FD
        from talkNet import talkNet
    except Exception as e:
        raise RuntimeError("TalkNet repo import 실패.") from e
    return S3FD, talkNet


def ensure_ultralytics_imports():
    try:
        from ultralytics import YOLO
    except Exception as e:
        raise RuntimeError("Ultralytics import 실패.") from e
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
    cmd = [sys.executable, "-m", "gdown", "--id", "1AbN9fCf9IexMxEKXLQY2KYBlb-IhSEea", "-O", str(checkpoint_path)]
    print(f"[download] TalkNet checkpoint -> {checkpoint_path}")
    subprocess.run(cmd, check=True)


# =========================================================
# helper utils  (MELD 버전과 동일)
# =========================================================
def run_cmd(cmd: str):
    subprocess.run(cmd, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def normalize_text(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9\s']+", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s

def text_similarity(a: str, b: str) -> float:
    a_n, b_n = normalize_text(a), normalize_text(b)
    if not a_n and not b_n:
        return 1.0
    return difflib.SequenceMatcher(None, a_n, b_n).ratio()

def moving_average(arr: np.ndarray, k: int) -> np.ndarray:
    if len(arr) == 0: return arr
    if k <= 1: return arr.copy()
    k = max(1, int(k))
    pad = k // 2
    x = np.pad(arr, (pad, pad), mode="edge")
    kernel = np.ones(k, dtype=np.float32) / float(k)
    return np.convolve(x, kernel, mode="valid")

def expand_bbox(x1, y1, x2, y2, w, h, ratio: float):
    bw, bh = x2 - x1, y2 - y1
    ex, ey = bw * ratio, bh * ratio
    return max(0, int(x1-ex)), max(0, int(y1-ey)), min(w-1, int(x2+ex)), min(h-1, int(y2+ey))

def bb_iou(boxA, boxB):
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    inter = max(0, xB-xA) * max(0, yB-yA)
    aA = max(0, boxA[2]-boxA[0]) * max(0, boxA[3]-boxA[1])
    aB = max(0, boxB[2]-boxB[0]) * max(0, boxB[3]-boxB[1])
    return inter / (aA + aB - inter + 1e-6)

def dilate_binary_mask(mask: np.ndarray, px: int) -> np.ndarray:
    if px <= 0: return mask
    kernel = np.ones((max(1,int(px)), max(1,int(px))), np.uint8)
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)


# =========================================================
# ★ IEMOCAP 전용: 비디오 탐색 (하위 디렉토리 재귀)
# =========================================================
def find_videos(input_dir: Path) -> List[Path]:
    """
    sentences/avi/Ses05F_impro01/*.avi 처럼 하위 디렉토리에 있는 avi 전부 수집
    FXX/MXX 파일은 스킵
    """
    videos = []
    for avi in sorted(input_dir.rglob("*.avi")):
        if avi.name.startswith("._"):
            continue
        dm, spk, utt = parse_iemocap_filename(avi.stem)
        if dm is None:  # FXX/MXX 등 불명확 발화
            continue
        videos.append(avi)
    return videos


# =========================================================
# scene detection / face detection / tracking  (동일)
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
    if not scene_list:
        scene_list = [(video_manager.get_base_timecode(), video_manager.get_current_timecode())]
    with open(pywork_path / "scene.pckl", "wb") as f:
        pickle.dump(scene_list, f)
    return scene_list


def inference_video_s3fd(pyframes_path, pywork_path, S3FD_cls, cfg):
    det = S3FD_cls(device=cfg.device)
    flist = sorted(glob.glob(str(pyframes_path / "*.jpg")))
    dets = []
    for fidx, fname in enumerate(tqdm.tqdm(flist, desc="face detect", leave=False)):
        image = cv2.imread(fname)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        bboxes = det.detect_faces(image_rgb, conf_th=cfg.face_conf, scales=[cfg.facedet_scale])
        dets.append([])
        for bbox in bboxes:
            dets[-1].append({"frame": fidx, "bbox": (bbox[:-1]).tolist(), "conf": float(bbox[-1])})
    with open(pywork_path / "faces.pckl", "wb") as f:
        pickle.dump(dets, f)
    return dets


def track_shot(scene_faces, cfg):
    iou_thres = 0.5
    tracks = []
    while True:
        track = []
        for frame_faces in scene_faces:
            for face in list(frame_faces):
                if not track:
                    track.append(face); frame_faces.remove(face)
                elif face["frame"] - track[-1]["frame"] <= cfg.num_failed_det:
                    if bb_iou(face["bbox"], track[-1]["bbox"]) > iou_thres:
                        track.append(face); frame_faces.remove(face); continue
                else:
                    break
        if not track:
            break
        elif len(track) > cfg.min_track:
            frame_num = np.array([f["frame"] for f in track])
            bboxes    = np.array([np.array(f["bbox"]) for f in track])
            frame_i   = np.arange(frame_num[0], frame_num[-1] + 1)
            bboxes_i  = np.stack([interp1d(frame_num, bboxes[:, ij])(frame_i) for ij in range(4)], axis=1)
            face_size = max(np.mean(bboxes_i[:,2]-bboxes_i[:,0]), np.mean(bboxes_i[:,3]-bboxes_i[:,1]))
            if face_size > cfg.min_face_size:
                tracks.append({"frame": frame_i, "bbox": bboxes_i})
    return tracks


def crop_video(cfg, track, crop_file_stem, pyframes_path, audio_file_path):
    flist = sorted(glob.glob(str(pyframes_path / "*.jpg")))
    v_out = cv2.VideoWriter(str(crop_file_stem)+"t.avi", cv2.VideoWriter_fourcc(*"XVID"), cfg.fps, (224,224))
    dets = {"x":[], "y":[], "s":[]}
    for det in track["bbox"]:
        dets["s"].append(max((det[3]-det[1]),(det[2]-det[0]))/2)
        dets["y"].append((det[1]+det[3])/2)
        dets["x"].append((det[0]+det[2])/2)
    dets["s"] = signal.medfilt(dets["s"], kernel_size=13)
    dets["x"] = signal.medfilt(dets["x"], kernel_size=13)
    dets["y"] = signal.medfilt(dets["y"], kernel_size=13)
    for fidx, frame in enumerate(track["frame"]):
        cs, bs = cfg.crop_scale, dets["s"][fidx]
        bsi   = int(bs*(1+2*cs))
        image = cv2.imread(flist[frame])
        frame_pad = np.pad(image, ((bsi,bsi),(bsi,bsi),(0,0)), "constant", constant_values=(110,110))
        my, mx = dets["y"][fidx]+bsi, dets["x"][fidx]+bsi
        face = frame_pad[int(my-bs):int(my+bs*(1+2*cs)), int(mx-bs*(1+cs)):int(mx+bs*(1+cs))]
        v_out.write(cv2.resize(face, (224,224)))
    v_out.release()
    audio_tmp   = str(crop_file_stem)+".wav"
    audio_start = track["frame"][0]/cfg.fps
    audio_end   = (track["frame"][-1]+1)/cfg.fps
    run_cmd(f"ffmpeg -y -i {audio_file_path} -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 -threads {cfg.num_workers} -ss {audio_start:.3f} -to {audio_end:.3f} {audio_tmp} -loglevel panic")
    run_cmd(f"ffmpeg -y -i {str(crop_file_stem)}t.avi -i {audio_tmp} -threads {cfg.num_workers} -c:v copy -c:a copy {str(crop_file_stem)}.avi -loglevel panic")
    os.remove(str(crop_file_stem)+"t.avi")
    return {"track": track, "proc_track": dets}


# =========================================================
# TalkNet / WhisperX / fusion  (동일)
# =========================================================
def evaluate_network_talknet(files, talkNet_cls, checkpoint_path):
    s = talkNet_cls(); s.loadParameters(checkpoint_path); s.eval()
    all_scores = []
    duration_set = {1,1,1,2,2,2,3,3,4,5,6}
    for file in tqdm.tqdm(files, desc="talknet asd", leave=False):
        wav_path = os.path.splitext(file)[0]+".wav"
        _, audio  = wavfile.read(wav_path)
        audio_feature = python_speech_features.mfcc(audio, 16000, numcep=13, winlen=0.025, winstep=0.010)
        video = cv2.VideoCapture(file); video_feature = []
        while video.isOpened():
            ret, frames = video.read()
            if not ret: break
            face = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
            face = cv2.resize(face, (224,224))
            face = face[56:168, 56:168]
            video_feature.append(face)
        video.release()
        video_feature = np.array(video_feature)
        length = min((audio_feature.shape[0]-audio_feature.shape[0]%4)/100, video_feature.shape[0]/25)
        audio_feature = audio_feature[:int(round(length*100)),:]
        video_feature = video_feature[:int(round(length*25)),:,:]
        score_ensembles = []
        for duration in duration_set:
            batch_size = int(math.ceil(length/duration)); scores = []
            with torch.no_grad():
                for i in range(batch_size):
                    inputA = torch.FloatTensor(audio_feature[i*duration*100:(i+1)*duration*100,:]).unsqueeze(0).cuda()
                    inputV = torch.FloatTensor(video_feature[i*duration*25:(i+1)*duration*25,:,:]).unsqueeze(0).cuda()
                    embedA = s.model.forward_audio_frontend(inputA)
                    embedV = s.model.forward_visual_frontend(inputV)
                    embedA, embedV = s.model.forward_cross_attention(embedA, embedV)
                    out   = s.model.forward_audio_visual_backend(embedA, embedV)
                    score = s.lossAV.forward(out, labels=None)
                    scores.extend(score)
            score_ensembles.append(scores)
        all_scores.append(np.round(np.mean(np.array(score_ensembles), axis=0), 1).astype(float))
    return all_scores


def run_whisperx(audio_path: Path, cfg: PipelineConfig) -> dict:
    if not cfg.use_whisperx:
        return {"text":"","segments":[],"words":[],"diarization":[]}
    try:
        import whisperx
    except Exception as e:
        raise RuntimeError(f"whisperx import 실패: {e}") from e
    device = cfg.device if torch.cuda.is_available() else "cpu"
    compute_type = cfg.whisper_compute_type if device=="cuda" else "int8"
    model = whisperx.load_model(cfg.whisper_model, device, compute_type=compute_type)
    audio = whisperx.load_audio(str(audio_path))
    result = model.transcribe(audio, batch_size=cfg.whisper_batch_size)
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
    full_text = " ".join([seg.get("text","") for seg in result.get("segments",[])]).strip()
    words = [w for seg in result.get("segments",[]) for w in seg.get("words",[]) if "start" in w and "end" in w]
    del model
    try: del model_a
    except: pass
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return {"text":full_text,"segments":result.get("segments",[]),"words":words,"diarization":[],"language":result.get("language","en")}


def build_speech_mask(n_frames, fps, asr_result):
    speech = np.zeros(n_frames, dtype=np.float32)
    words = asr_result.get("words",[]) or []
    if words:
        for w in words:
            s = max(0, int(float(w["start"])*fps))
            e = min(n_frames, int(math.ceil(float(w["end"])*fps)))
            speech[s:e] = 1.0
    else:
        for seg in asr_result.get("segments",[]) or []:
            if "start" in seg and "end" in seg:
                s = max(0, int(float(seg["start"])*fps))
                e = min(n_frames, int(math.ceil(float(seg["end"])*fps)))
                speech[s:e] = 1.0
    if speech.sum() > 0:
        speech = moving_average(speech, max(3, int(0.12*fps)))
        speech = (speech > 0.15).astype(np.float32)
    return speech


def build_track_table(tracks, scores, n_frames, smooth_k):
    rows = []
    for tidx, track in enumerate(tracks):
        score = moving_average(np.asarray(scores[tidx], dtype=np.float32), smooth_k)
        for fidx, frame in enumerate(track["track"]["frame"].tolist()):
            if fidx >= len(score): continue
            rows.append({"frame":int(frame),"track_id":tidx,"score":float(score[fidx]),
                         "s":float(track["proc_track"]["s"][fidx]),
                         "x":float(track["proc_track"]["x"][fidx]),
                         "y":float(track["proc_track"]["y"][fidx])})
    if not rows:
        return pd.DataFrame(columns=["frame","track_id","score","s","x","y"])
    df = pd.DataFrame(rows)
    return df[df["frame"] < n_frames].copy()


def decode_active_speaker(track_df, n_tracks, n_frames, speech_mask, transcript_sim, cfg):
    null_state = n_tracks
    n_states   = n_tracks + 1
    emission   = np.full((n_frames, n_states), -10.0, dtype=np.float32)
    emission[:, null_state] = cfg.null_bias
    for _, row in track_df.iterrows():
        f, t, s = int(row["frame"]), int(row["track_id"]), float(row["score"])
        s += cfg.transcript_match_bonus * transcript_sim
        s += cfg.speech_bonus if speech_mask[f] > 0 else -cfg.silence_penalty
        emission[f, t] = max(emission[f, t], s)
    for f in range(n_frames):
        if speech_mask[f] > 0: emission[f, null_state] -= 1.0
        else:                   emission[f, null_state] += 0.25
    dp   = np.full((n_frames, n_states), -1e9, dtype=np.float32)
    back = np.full((n_frames, n_states), -1,   dtype=np.int32)
    dp[0] = emission[0]
    for f in range(1, n_frames):
        for cur in range(n_states):
            best_s, best_p = -1e9, -1
            for prev in range(n_states):
                trans = 0.0 if prev==cur else -cfg.switch_penalty
                if prev==null_state or cur==null_state: trans += 0.25
                cand = dp[f-1,prev] + trans + emission[f,cur]
                if cand > best_s: best_s, best_p = cand, prev
            dp[f,cur], back[f,cur] = best_s, best_p
    states = [int(np.argmax(dp[-1]))]
    for f in range(n_frames-1, 0, -1): states.append(int(back[f, states[-1]]))
    return np.asarray(states[::-1], dtype=np.int32), emission


# =========================================================
# person segmentation + render  (동일)
# =========================================================
def face_row_to_box(row, width, height, expand_ratio=0.10):
    s, x, y = float(row["s"]), float(row["x"]), float(row["y"])
    return expand_bbox(int(x-s), int(y-s), int(x+s), int(y+s), width, height, expand_ratio)

def fallback_body_mask_from_face(row, width, height, cfg):
    s, x, y = float(row["s"]), float(row["x"]), float(row["y"])
    mask = np.zeros((height, width), dtype=bool)
    x1 = max(0, int(x - cfg.body_fallback_expand_x*s))
    x2 = min(width, int(x + cfg.body_fallback_expand_x*s))
    y1 = max(0, int(y - cfg.body_fallback_expand_top*s))
    y2 = min(height, int(y + cfg.body_fallback_expand_bottom*s))
    mask[y1:y2, x1:x2] = True
    return mask

def choose_active_person_mask(frame, row, seg_result, cfg):
    h, w = frame.shape[:2]
    face_box = face_row_to_box(row, w, h)
    cx, cy   = int((face_box[0]+face_box[2])/2), int((face_box[1]+face_box[3])/2)
    if seg_result is None or getattr(seg_result,"masks",None) is None or seg_result.masks is None:
        return fallback_body_mask_from_face(row, w, h, cfg), None, True
    masks_data = seg_result.masks.data
    if masks_data is None or len(masks_data)==0:
        return fallback_body_mask_from_face(row, w, h, cfg), None, True
    masks     = masks_data.cpu().numpy() > 0.5
    boxes     = seg_result.boxes.xyxy.cpu().numpy() if seg_result.boxes is not None else np.zeros((0,4))
    confs     = seg_result.boxes.conf.cpu().numpy() if seg_result.boxes is not None else np.zeros((0,))
    best_idx, best_score = -1, -1e9
    for idx in range(len(masks)):
        mask = masks[idx]
        bx1,by1,bx2,by2 = boxes[idx]
        inside   = 1.0 if (0<=cy<mask.shape[0] and 0<=cx<mask.shape[1] and mask[cy,cx]) else 0.0
        face_iou = bb_iou(face_box, (bx1,by1,bx2,by2))
        bcx,bcy  = 0.5*(bx1+bx2), 0.5*(by1+by2)
        dist     = math.hypot(bcx-cx,bcy-cy)/max(1.0,math.hypot(w,h))
        box_area = max(0.0,(bx2-bx1)*(by2-by1))/max(1.0,float(w*h))
        score    = 4.0*inside + 2.5*face_iou + 0.20*float(confs[idx]) + 0.25*min(box_area*5.0,1.0) - 1.50*dist
        if score > best_score: best_score, best_idx = score, idx
    if best_idx < 0:
        return fallback_body_mask_from_face(row, w, h, cfg), None, True
    chosen_mask = dilate_binary_mask(masks[best_idx], cfg.mask_dilate_px)
    return chosen_mask, boxes[best_idx], False

def compose_foreground_black(frame, person_mask):
    out = np.zeros_like(frame)
    out[person_mask] = frame[person_mask]
    return out


def render_outputs(render_video, audio_source, output_video, track_df, states, cfg, person_seg_model):
    cap    = cv2.VideoCapture(str(render_video))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = int(round(cap.get(cv2.CAP_PROP_FPS))) or cfg.fps
    output_video.parent.mkdir(parents=True, exist_ok=True)
    tmp_video_only = output_video.with_suffix(".video_only.avi")
    writer = cv2.VideoWriter(str(tmp_video_only), cv2.VideoWriter_fourcc(*"XVID"), fps, (width,height))
    per_frame: Dict[int, List] = {}
    for _, row in track_df.iterrows():
        per_frame.setdefault(int(row["frame"]), []).append(row)
    n_tracks = int(track_df["track_id"].max())+1 if len(track_df)>0 else 0
    last_seg_state, last_seg_mask, last_seg_frame_idx = None, None, -999999
    frames_rendered = blackout_frames = valid_frames = fallback_frames = 0
    for fidx in tqdm.tqdm(range(total), desc="render", leave=False):
        ret, frame = cap.read()
        if not ret: break
        frames_rendered += 1
        chosen_state = int(states[fidx]) if fidx < len(states) else -1
        chosen_row   = next((r for r in per_frame.get(fidx,[]) if int(r["track_id"])==chosen_state), None)
        if chosen_row is None or chosen_state < 0 or chosen_state >= n_tracks:
            blackout_frames += 1
            masked_frame = np.zeros_like(frame)
        else:
            need_new_seg = not (cfg.person_seg_stride>1 and last_seg_state==chosen_state
                                and (fidx-last_seg_frame_idx)<cfg.person_seg_stride and last_seg_mask is not None)
            if need_new_seg:
                seg_result = person_seg_model.predict(source=frame, verbose=False, device=cfg.device,
                    classes=[0], conf=cfg.person_seg_conf, iou=cfg.person_seg_iou,
                    imgsz=cfg.person_seg_imgsz, retina_masks=True)[0]
                person_mask, _, used_fallback = choose_active_person_mask(frame, chosen_row, seg_result, cfg)
                last_seg_mask, last_seg_state, last_seg_frame_idx = person_mask, chosen_state, fidx
            else:
                person_mask, used_fallback = last_seg_mask, False
            masked_frame = compose_foreground_black(frame, person_mask)
            valid_frames += 1
            if used_fallback: fallback_frames += 1
        writer.write(masked_frame)
    cap.release(); writer.release()
    run_cmd(f"ffmpeg -y -i {tmp_video_only} -i {audio_source} -map 0:v:0 -map 1:a:0? -c:v copy -c:a aac {output_video} -loglevel panic")
    os.remove(tmp_video_only)
    denom = float(frames_rendered) or 1.0
    return {"frames_rendered":frames_rendered,"blackout_frames":blackout_frames,
            "blackout_ratio":blackout_frames/denom,"valid_active_speaker_visible_frames":valid_frames,
            "valid_active_speaker_visible_ratio":valid_frames/denom,
            "fallback_body_visible_frames":fallback_frames,"fallback_body_visible_ratio":fallback_frames/denom}


# =========================================================
# per-video pipeline
# =========================================================
def prepare_workdirs(video_stem):
    import tempfile
    work_root = Path(tempfile.mkdtemp(prefix=f"iemocap_{video_stem}_"))
    pyavi, pyframes, pywork, pycrop = work_root/"pyavi", work_root/"pyframes", work_root/"pywork", work_root/"pycrop"
    for d in [pyavi, pyframes, pywork, pycrop]: d.mkdir(parents=True, exist_ok=True)
    return work_root, pyavi, pyframes, pywork, pycrop


def extract_video_audio_frames(input_video, pyavi, pyframes, cfg):
    video_file = pyavi / "video.avi"
    audio_file = pyavi / "audio.wav"
    run_cmd(f"ffmpeg -y -i {input_video} -qscale:v 2 -threads {cfg.num_workers} -async 1 -r {cfg.fps} {video_file} -loglevel panic")
    run_cmd(f"ffmpeg -y -i {video_file} -qscale:v 2 -threads {cfg.num_workers} {pyframes/'%06d.jpg'} -loglevel panic")
    run_cmd(f"ffmpeg -y -i {video_file} -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 {audio_file} -loglevel panic")
    return video_file, audio_file


def process_one_video(video_path, out_root, cfg, iemocap_meta, S3FD_cls, talkNet_cls, checkpoint_path, person_seg_model):
    stem = video_path.stem
    clip_meta = iemocap_meta.get(stem)

    work_root, pyavi, pyframes, pywork, pycrop = prepare_workdirs(stem)
    try:
        video_file, audio_file = extract_video_audio_frames(video_path, pyavi, pyframes, cfg)
        scene_list = scene_detect(video_file, pywork)
        dets = inference_video_s3fd(pyframes, pywork, S3FD_cls, cfg)

        all_tracks, crop_files = [], []
        for shot in scene_list:
            start, end = shot[0].frame_num, shot[1].frame_num
            scene_faces = copy.deepcopy(dets[start:end])
            for tr in track_shot(scene_faces, cfg):
                crop_stem = pycrop / ("%05d" % len(all_tracks))
                all_tracks.append(crop_video(cfg, tr, crop_stem, pyframes, audio_file))
                crop_files.append(str(crop_stem)+".avi")

        shutil.rmtree(pyframes, ignore_errors=True)

        talk_scores = evaluate_network_talknet(crop_files, talkNet_cls, str(checkpoint_path)) if crop_files else []

        cap = cv2.VideoCapture(str(video_file))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps      = int(round(cap.get(cv2.CAP_PROP_FPS))) or cfg.fps
        width    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        asr          = run_whisperx(audio_file, cfg) if cfg.use_whisperx else {"text":"","segments":[],"words":[],"diarization":[]}
        speech_mask  = build_speech_mask(n_frames, fps, asr)
        meta_text    = clip_meta.utterance if clip_meta else ""
        whisper_text = asr.get("text","")
        trans_sim    = text_similarity(meta_text, whisper_text) if meta_text else 0.0

        track_df     = build_track_table(all_tracks, talk_scores, n_frames, cfg.talknet_smooth)
        states, _    = decode_active_speaker(track_df, len(all_tracks), n_frames, speech_mask, trans_sim, cfg)

        # ★ 출력 경로: out_root/Ses05F_impro01_F000.avi (flat)
        output_video = out_root / f"{stem}.avi"
        render_stats = render_outputs(video_file, video_path, output_video, track_df, states, cfg, person_seg_model)

        selected_ratio   = float(np.mean(states < len(all_tracks))) if len(states)>0 else 0.0
        mean_speech      = float(np.mean(speech_mask)) if len(speech_mask)>0 else 0.0
        mean_track_score = float(track_df["score"].mean()) if len(track_df)>0 else 0.0

        log = {
            "file": f"{stem}.avi",
            "input_path": str(video_path),
            "output_path": str(output_video),
            "status": "ok",
            "utt_name": stem,
            "dialog_name": clip_meta.dialog_name if clip_meta else "",
            "speaker": clip_meta.speaker if clip_meta else "",
            "emotion": clip_meta.emotion if clip_meta else "",
            "iemocap_utterance": meta_text,
            "whisperx_text": whisper_text,
            "transcript_similarity": trans_sim,
            "num_tracks_total": len(all_tracks),
            "frames_total": n_frames, "fps": fps, "width": width, "height": height,
            "speech_active_ratio": mean_speech,
            "selected_frame_ratio": selected_ratio,
            "mean_track_score": mean_track_score,
            **render_stats,
        }
    finally:
        shutil.rmtree(work_root, ignore_errors=True)
    return log


# =========================================================
# batch driver + CSV 저장
# =========================================================
def write_logs_csv(logs, csv_path):
    if not logs: return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    cols = []
    for row in logs:
        for k in row.keys():
            if k not in cols: cols.append(k)
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for row in logs: writer.writerow(row)


def main():
    ap = argparse.ArgumentParser(description="IEMOCAP active-speaker masking pipeline")
    ap.add_argument("--input_dir",   required=True,  help="sentences/avi/ 경로")
    ap.add_argument("--output_dir",  required=True,  help="IEMOCAP_intervention/C1_visual/")
    ap.add_argument("--log_dir",     required=True,  help="IEMOCAP_intervention_log_csv/")
    ap.add_argument("--talknet_repo",required=True)
    ap.add_argument("--iemocap_session_dir", default="", help="Session5/ 경로 (메타데이터용)")
    ap.add_argument("--pretrain_model", default="pretrain_TalkSet.model")
    ap.add_argument("--hf_token",    default="")
    ap.add_argument("--device",      default="cuda")
    ap.add_argument("--limit",       type=int, default=0)
    ap.add_argument("--reset_output",action="store_true")
    ap.add_argument("--disable_whisperx",    action="store_true")
    ap.add_argument("--disable_diarization", action="store_true")
    ap.add_argument("--person_seg_model",    default="yolo11n-seg.pt")
    ap.add_argument("--person_seg_stride",   type=int,   default=1)
    ap.add_argument("--shutdown_when_done",  action="store_true")
    args = ap.parse_args()

    cfg = PipelineConfig(
        input_dir=args.input_dir, output_dir=args.output_dir,
        talknet_repo=args.talknet_repo, iemocap_session_dir=args.iemocap_session_dir,
        pretrain_model=args.pretrain_model, hf_token=args.hf_token,
        device=args.device, limit=args.limit, reset_output=args.reset_output,
        use_whisperx=not args.disable_whisperx, use_diarization=not args.disable_diarization,
        person_seg_model=args.person_seg_model, person_seg_stride=max(1, args.person_seg_stride),
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
    out_root  = Path(cfg.output_dir)
    log_dir   = Path(args.log_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    iemocap_meta = load_iemocap_metadata(cfg.iemocap_session_dir)
    videos = find_videos(input_dir)
    if cfg.limit > 0:
        videos = videos[:cfg.limit]
    print(f"[videos] {len(videos)}개 utterance 영상 발견")

    logs, skipped = [], 0
    for idx, video_path in enumerate(videos, start=1):
        output_video = out_root / f"{video_path.stem}.avi"
        if output_video.exists():
            skipped += 1
            print(f"[{idx:04d}/{len(videos):04d}] skip: {video_path.name}")
            continue
        print(f"\n[{idx:04d}/{len(videos):04d}] {video_path.name}")
        try:
            log = process_one_video(video_path, out_root, cfg, iemocap_meta,
                                    S3FD_cls, talkNet_cls, checkpoint_path, person_seg_model)
            logs.append(log)
            print(f"  -> status={log['status']} | tracks={log['num_tracks_total']} | "
                  f"selected={log['selected_frame_ratio']:.3f} | blackout={log['blackout_ratio']:.2f}")
        except Exception as e:
            logs.append({"file": video_path.name, "status": "exception", "error": str(e)})
            print(f"  !! exception: {e}")

    csv_path = log_dir / "iemocap_session5_C1_visual_log.csv"
    write_logs_csv(logs, csv_path)
    print(f"\n[done] 처리={len(logs)}개 | 스킵={skipped}개 | log -> {csv_path}")

    if args.shutdown_when_done:
        print("[shutdown] 5초 후 종료...")
        subprocess.run("sleep 5 && sudo shutdown -h now", shell=True)


if __name__ == "__main__":
    main()