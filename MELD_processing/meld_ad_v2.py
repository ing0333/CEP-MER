#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
meld_ad_v2.py

MELD audio intervention — 배경음/배경음악/배경소음만 제거

모든 사람의 음성은 그대로 유지.
배경음악, 효과음, 배경소음만 제거.

파이프라인:
  1. Demucs (mdx_extra_q) — vocals 트랙 추출 (BGM, 효과음, 배경음악 제거)
  2. DeepFilterNet         — 잔여 소음 제거 (공조음, 웅성거림 등)

설치:
    pip install demucs deepfilternet torchaudio

사용법:
    python audio_bg_remove.py \\
        --input_dir  /path/to/input \\
        --output_dir /path/to/output \\
        [--skip_deepfilter]   # DeepFilterNet 생략하고 Demucs만
        [--shutdown_when_done]
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import subprocess
import sys
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List

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
    device: str = "cuda"
    num_workers: int = 8
    skip_deepfilter: bool = False
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
    vids = []
    for ext in ("*.mp4", "*.avi", "*.mov", "*.mkv"):
        vids.extend(sorted(d.glob(ext)))
    return sorted(v for v in vids if not v.name.startswith("._"))


# =========================================================
# Step 1: Demucs — 한 번만 로드
# =========================================================
def load_demucs_model(device: str):
    try:
        from demucs.pretrained import get_model
    except ImportError:
        raise RuntimeError("demucs가 없습니다. `pip install demucs`")
    print("[init] Demucs 로드 중...")
    model = get_model("mdx_extra_q")
    model.to(device)
    model.eval()
    print("[init] Demucs 완료")
    return model


def demucs_extract_vocals(audio_path: Path, out_path: Path, model, device: str):
    """
    vocals 트랙만 추출.
    모든 사람 목소리 유지, BGM/효과음/배경음악 제거.
    """
    from demucs.apply import apply_model
    wav, sr = torchaudio.load(str(audio_path))
    if sr != model.samplerate:
        wav = torchaudio.functional.resample(wav, sr, model.samplerate)
        sr = model.samplerate
    if wav.shape[0] == 1:
        wav = wav.repeat(2, 1)
    wav = wav.to(device)
    with torch.no_grad():
        sources = apply_model(model, wav.unsqueeze(0), device=device)[0]
    vocals = sources[model.sources.index("vocals")].cpu()
    torchaudio.save(str(out_path), vocals, sr)


# =========================================================
# Step 2: DeepFilterNet — 한 번만 로드
# =========================================================
def load_deepfilter_model():
    try:
        from df.enhance import init_df
    except ImportError:
        raise RuntimeError("deepfilternet이 없습니다. `pip install deepfilternet`")
    print("[init] DeepFilterNet 로드 중...")
    model, df_state, _ = init_df()
    print("[init] DeepFilterNet 완료")
    return model, df_state


def deepfilter_denoise(vocals_path: Path, out_path: Path, df_model, df_state):
    from df.enhance import enhance, load_audio, save_audio
    audio, _ = load_audio(str(vocals_path), sr=df_state.sr())
    enhanced = enhance(df_model, df_state, audio)
    save_audio(str(out_path), enhanced, df_state.sr())


# =========================================================
# 단일 영상 처리
# =========================================================
def process_one_video(
    video_path: Path,
    output_dir: Path,
    cfg: Config,
    demucs_model,
    df_model,
    df_state,
) -> dict:
    stem        = video_path.stem
    output_path = output_dir / video_path.name

    with tempfile.TemporaryDirectory(prefix=f"bgrem_{stem}_") as tmpdir:
        tmp = Path(tmpdir)

        # 44100Hz stereo로 추출 (Demucs 요구사항)
        audio_44k = tmp / "audio_44k.wav"
        run_cmd(f"ffmpeg -y -i {video_path} -ac 2 -ar 44100 "
                f"{audio_44k} -loglevel panic")

        # ── Step 1: Demucs — vocals 추출 ──
        vocals_path = tmp / "vocals.wav"
        demucs_extract_vocals(audio_44k, vocals_path, demucs_model, cfg.device)

        # ── Step 2: DeepFilterNet — 잔여 소음 제거 (선택) ──
        if cfg.skip_deepfilter:
            final_audio = vocals_path
        else:
            denoised_path = tmp / "denoised.wav"
            deepfilter_denoise(vocals_path, denoised_path, df_model, df_state)
            final_audio = denoised_path

        # ── 원본 영상 + 처리된 오디오 mux ──
        output_path.parent.mkdir(parents=True, exist_ok=True)
        run_cmd(
            f"ffmpeg -y -i {video_path} -i {final_audio} "
            f"-map 0:v:0 -map 1:a:0 -c:v copy -c:a aac "
            f"-shortest {output_path} -loglevel panic"
        )

    return {
        "file":        video_path.name,
        "output_path": str(output_path),
        "status":      "ok",
    }


# =========================================================
# batch driver
# =========================================================
def main():
    ap = argparse.ArgumentParser(
        description="MELD audio: 배경음/배경음악/배경소음만 제거 (모든 음성 유지)"
    )
    ap.add_argument("--input_dir",  required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--device",     default="cuda")
    ap.add_argument("--limit",      type=int, default=0)
    ap.add_argument("--skip_deepfilter", action="store_true",
                    help="DeepFilterNet 생략 (Demucs만 적용)")
    ap.add_argument("--shutdown_when_done", action="store_true")
    args = ap.parse_args()

    ensure_ffmpeg()

    cfg = Config(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        device=args.device,
        limit=args.limit,
        skip_deepfilter=args.skip_deepfilter,
        shutdown_when_done=args.shutdown_when_done,
    )

    # ── 모델 한 번만 로드 ──
    demucs_model       = load_demucs_model(cfg.device)
    df_model, df_state = (None, None) if cfg.skip_deepfilter else load_deepfilter_model()

    input_dir  = Path(cfg.input_dir)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    videos = find_videos(input_dir)
    if cfg.limit > 0:
        videos = videos[:cfg.limit]

    mode = "Demucs only" if cfg.skip_deepfilter else "Demucs + DeepFilterNet"
    print(f"\n[videos] {len(videos)}개")
    print(f"[pipeline] {mode}\n")

    logs, skipped = [], 0
    for idx, video in enumerate(videos, 1):
        # resume
        if (output_dir / video.name).exists():
            skipped += 1
            print(f"[{idx:04d}/{len(videos):04d}] skip: {video.name}")
            continue

        print(f"[{idx:04d}/{len(videos):04d}] {video.name}", end=" ", flush=True)
        try:
            log = process_one_video(
                video_path=video,
                output_dir=output_dir,
                cfg=cfg,
                demucs_model=demucs_model,
                df_model=df_model,
                df_state=df_state,
            )
            logs.append(log)
            print("-> ok")
        except Exception as e:
            logs.append({"file": video.name, "status": "exception", "error": str(e)})
            print(f"!! {e}")

    # 로그 저장
    log_path = output_dir / "bg_remove_log.csv"
    if logs:
        cols = []
        for r in logs:
            for k in r:
                if k not in cols: cols.append(k)
        with open(log_path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in logs: w.writerow(r)

    print(f"\n[done] processed={len(logs)} skipped={skipped} | log -> {log_path}")

    if cfg.shutdown_when_done:
        print("[shutdown] 5초 후 종료...")
        subprocess.run("sleep 5 && sudo shutdown -h now", shell=True)


if __name__ == "__main__":
    main()