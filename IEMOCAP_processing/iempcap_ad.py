#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
iemocap_ad.py

IEMOCAP audio intervention — VAD only 버전
Demucs / DeepFilterNet 없음.

파이프라인:
  1. avi에서 오디오 추출 (wav)
  2. Silero VAD — 발화 구간 감지
  3. VAD 구간만 유지, 나머지 무음 처리
  4. 원본 avi + masked audio 합쳐서 avi 출력
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import subprocess
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torchaudio

warnings.filterwarnings("ignore")


# =========================================================
# config
# =========================================================
@dataclass
class Config:
    input_dir: str
    output_dir: str
    log_dir: str
    device: str = "cuda"

    vad_threshold: float = 0.6
    vad_min_speech_ms: int = 80
    vad_min_silence_ms: int = 80
    vad_pad_ms: int = 30

    limit: int = 0
    shutdown_when_done: bool = False


# =========================================================
# utils
# =========================================================
def run_cmd(cmd: str):
    subprocess.run(cmd, shell=True, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def ensure_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], check=True, capture_output=True)
    except Exception:
        raise RuntimeError("ffmpeg가 PATH에 없습니다.")


def find_videos(d: Path) -> List[Path]:
    """하위 디렉토리 재귀 탐색, FXX/MXX 및 macOS 숨김파일 스킵"""
    videos = []
    for avi in sorted(d.rglob("*.avi")):
        if avi.name.startswith("._"):
            continue
        if re.search(r"_[FM]XX\d+\.avi$", avi.name):
            continue
        videos.append(avi)
    return videos


def merge_intervals(intervals: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda x: x[0])
    merged = [list(intervals[0])]
    for s, e in intervals[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(a, b) for a, b in merged]


def apply_audio_mask(audio: np.ndarray, sr: int,
                     keep: List[Tuple[float, float]]) -> np.ndarray:
    out    = np.zeros_like(audio)
    n_samp = audio.shape[-1] if audio.ndim > 1 else len(audio)
    for s_sec, e_sec in keep:
        s = max(0, int(s_sec * sr))
        e = min(n_samp, int(math.ceil(e_sec * sr)))
        if e <= s:
            continue
        if audio.ndim == 1:
            out[s:e]    = audio[s:e]
        else:
            out[:, s:e] = audio[:, s:e]
    return out


# =========================================================
# Silero VAD — 한 번만 로드
# =========================================================
def load_silero_vad():
    print("[init] Silero VAD 로드 중...")
    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        trust_repo=True,
    )
    (get_speech_timestamps, _, read_audio, *_) = utils
    print("[init] Silero VAD 완료")
    return model, get_speech_timestamps, read_audio


def silero_vad_segments(audio_path: Path, vad_model,
                        get_speech_timestamps, read_audio,
                        cfg: Config, total_sec: float) -> List[Tuple[float, float]]:
    wav = read_audio(str(audio_path), sampling_rate=16000)
    timestamps = get_speech_timestamps(
        wav, vad_model, sampling_rate=16000,
        threshold=cfg.vad_threshold,
        min_speech_duration_ms=cfg.vad_min_speech_ms,
        min_silence_duration_ms=cfg.vad_min_silence_ms,
        return_seconds=True,
    )

    # 발화 못 찾으면 역치 낮춰서 재시도
    if not timestamps:
        timestamps = get_speech_timestamps(
            wav, vad_model, sampling_rate=16000,
            threshold=max(0.1, cfg.vad_threshold - 0.3),
            min_speech_duration_ms=50,
            min_silence_duration_ms=50,
            return_seconds=True,
        )

    # 그래도 없으면 전체 유지
    if not timestamps:
        return [(0.0, total_sec)]

    pad = cfg.vad_pad_ms / 1000.0
    result = []
    for t in timestamps:
        s = max(0.0, float(t["start"]) - pad)
        e = min(total_sec, float(t["end"]) + pad)
        result.append((s, e))
    return merge_intervals(result)


# =========================================================
# 단일 영상 처리
# =========================================================
def process_one_video(video_path: Path, output_dir: Path, cfg: Config,
                      vad_model, get_speech_timestamps, read_audio) -> dict:
    stem        = video_path.stem
    output_path = output_dir / video_path.name

    with tempfile.TemporaryDirectory(prefix=f"iemocap_vad_{stem}_") as tmpdir:
        tmp = Path(tmpdir)

        # ── avi에서 오디오 추출 ──
        audio_wav = tmp / "audio.wav"
        run_cmd(f"ffmpeg -y -i {video_path} -ac 2 -ar 44100 {audio_wav} -loglevel panic")

        # ── 총 길이 계산 ──
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
            capture_output=True, text=True
        )
        try:
            total_sec = float(probe.stdout.strip())
        except Exception:
            total_sec = 30.0

        # ── VAD용 16kHz mono 변환 ──
        audio_16k = tmp / "audio_16k.wav"
        run_cmd(f"ffmpeg -y -i {audio_wav} -ac 1 -ar 16000 {audio_16k} -loglevel panic")

        # ── Silero VAD — 발화 구간 감지 ──
        vad_intervals = silero_vad_segments(
            audio_16k, vad_model, get_speech_timestamps, read_audio, cfg, total_sec
        )

        # ── 마스킹 적용 ──
        wav_tensor, sr = torchaudio.load(str(audio_wav))
        masked_np      = apply_audio_mask(wav_tensor.numpy(), sr, vad_intervals)
        masked_path    = tmp / "masked.wav"
        torchaudio.save(str(masked_path), torch.from_numpy(masked_np), sr)

        # ── 원본 avi 영상 + masked audio → avi 출력 ──
        output_path.parent.mkdir(parents=True, exist_ok=True)
        run_cmd(
            f"ffmpeg -y -i {video_path} -i {masked_path} "
            f"-map 0:v:0 -map 1:a:0 -c:v copy -c:a aac "
            f"-shortest {output_path} -loglevel panic"
        )

    return {
        "file":           video_path.name,
        "output_path":    str(output_path),
        "status":         "ok",
        "n_vad_segments": len(vad_intervals),
        "total_sec":      round(total_sec, 2),
        "kept_sec":       round(sum(e - s for s, e in vad_intervals), 2),
    }


# =========================================================
# batch driver
# =========================================================
def main():
    ap = argparse.ArgumentParser(description="IEMOCAP audio intervention: Silero VAD only")
    ap.add_argument("--input_dir",  required=True,
                    help="sentences/avi/ 경로")
    ap.add_argument("--output_dir", required=True,
                    help="IEMOCAP_intervention/C2_audio/")
    ap.add_argument("--log_dir",    required=True,
                    help="IEMOCAP_intervention_log_csv/")
    ap.add_argument("--device",     default="cuda")
    ap.add_argument("--limit",      type=int, default=0)
    ap.add_argument("--vad_threshold",     type=float, default=0.6)
    ap.add_argument("--vad_pad_ms",        type=int,   default=30)
    ap.add_argument("--vad_min_speech_ms", type=int,   default=80)
    ap.add_argument("--vad_min_silence_ms",type=int,   default=80)
    ap.add_argument("--shutdown_when_done", action="store_true")
    args = ap.parse_args()

    ensure_ffmpeg()

    cfg = Config(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        log_dir=args.log_dir,
        device=args.device,
        limit=args.limit,
        vad_threshold=args.vad_threshold,
        vad_pad_ms=args.vad_pad_ms,
        vad_min_speech_ms=args.vad_min_speech_ms,
        vad_min_silence_ms=args.vad_min_silence_ms,
        shutdown_when_done=args.shutdown_when_done,
    )

    vad_model, get_speech_timestamps, read_audio = load_silero_vad()

    input_dir  = Path(cfg.input_dir)
    output_dir = Path(cfg.output_dir)
    log_dir    = Path(cfg.log_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    videos = find_videos(input_dir)
    if cfg.limit > 0:
        videos = videos[:cfg.limit]

    print(f"\n[videos] {len(videos)}개")
    print(f"[pipeline] Silero VAD only (threshold={cfg.vad_threshold}, pad={cfg.vad_pad_ms}ms)\n")

    logs, skipped = [], 0
    for idx, video in enumerate(videos, 1):
        output_path = output_dir / video.name
        if output_path.exists():
            skipped += 1
            print(f"[{idx:04d}/{len(videos):04d}] skip: {video.name}")
            continue

        print(f"[{idx:04d}/{len(videos):04d}] {video.name}", end=" ", flush=True)
        try:
            log = process_one_video(
                video_path=video,
                output_dir=output_dir,
                cfg=cfg,
                vad_model=vad_model,
                get_speech_timestamps=get_speech_timestamps,
                read_audio=read_audio,
            )
            logs.append(log)
            print(f"-> vad={log['n_vad_segments']} | "
                  f"kept={log['kept_sec']}s / {log['total_sec']}s")
        except Exception as e:
            logs.append({"file": video.name, "status": "exception", "error": str(e)})
            print(f"!! {e}")

    # CSV 저장
    csv_path = log_dir / "iemocap_session5_C2_audio_log.csv"
    if logs:
        cols = []
        for r in logs:
            for k in r:
                if k not in cols: cols.append(k)
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in logs: w.writerow(r)

    print(f"\n[done] 처리={len(logs)}개 | 스킵={skipped}개 | log -> {csv_path}")

    if cfg.shutdown_when_done:
        print("[shutdown] 5초 후 종료...")
        subprocess.run("sleep 5 && sudo shutdown -h now", shell=True)


if __name__ == "__main__":
    main()