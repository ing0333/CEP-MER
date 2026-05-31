#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
meld_ad_v1.py

MELD audio intervention — 단순하고 안정적인 버전

TalkNet 없음. MELD 클립은 이미 utterance 단위로 잘려있기 때문에
"누가 말하는지" 찾을 필요 없이 아래만 하면 됨.

파이프라인:
  1. Demucs (mdx_extra_q)   — BGM / 배경음악 제거 → vocals 추출
  2. DeepFilterNet           — 잔여 noise 제거 (웃음, 환경음, 배경 대화 감쇠)
  3. Silero VAD              — 발화 구간 정밀 감지 → 앞뒤 타이트하게 trim
  4. 마스킹                  — VAD 구간만 유지, 나머지 무음

Dataset limitation:
  - overlapping speech 구간에서 non-active speaker 목소리 일부 잔류 가능
  - 얼굴 미검출 등으로 인한 active speaker 오인식 없음 (TalkNet 미사용)

설치:
    pip install demucs deepfilternet torchaudio

사용법:
    python audio_intervention_v4.py \\
        --input_dir  /path/to/C1_masked/test \\
        --output_dir /path/to/C2_audio/test \\
        [--vad_threshold 0.5] \\
        [--vad_pad_ms 30] \\
        [--shutdown_when_done]
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import shutil
import subprocess
import sys
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torchaudio
import tqdm

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

    # VAD — 타이트하게 자르기 위해 기본값을 높게 설정
    vad_threshold: float = 0.6      # 높을수록 더 타이트 (0~1)
    vad_min_speech_ms: int = 80     # 이보다 짧은 발화 무시
    vad_min_silence_ms: int = 80    # 이보다 짧은 침묵은 발화로 채움
    vad_pad_ms: int = 30            # 발화 앞뒤 패딩 (ms) — 타이트하게 30ms

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


def demucs_extract_vocals(
    audio_path: Path,
    out_path: Path,
    model,
    device: str,
):
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


def deepfilter_denoise(
    vocals_path: Path,
    out_path: Path,
    df_model,
    df_state,
):
    from df.enhance import enhance, load_audio, save_audio
    audio, _ = load_audio(str(vocals_path), sr=df_state.sr())
    enhanced = enhance(df_model, df_state, audio)
    save_audio(str(out_path), enhanced, df_state.sr())


# =========================================================
# Step 3: Silero VAD — 한 번만 로드
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


def silero_vad_segments(
    audio_path: Path,
    vad_model,
    get_speech_timestamps,
    read_audio,
    cfg: Config,
    total_sec: float,
) -> List[Tuple[float, float]]:
    """
    Silero VAD로 발화 구간 감지.
    vad_pad_ms 만큼 앞뒤 패딩 추가 (타이트하게 30ms 기본).
    발화가 감지되지 않으면 전체 구간 반환 (무음 방지).
    """
    wav = read_audio(str(audio_path), sampling_rate=16000)
    timestamps = get_speech_timestamps(
        wav,
        vad_model,
        sampling_rate=16000,
        threshold=cfg.vad_threshold,
        min_speech_duration_ms=cfg.vad_min_speech_ms,
        min_silence_duration_ms=cfg.vad_min_silence_ms,
        return_seconds=True,
    )

    pad = cfg.vad_pad_ms / 1000.0

    if not timestamps:
        # VAD가 발화를 못 찾으면 역치 낮춰서 재시도
        timestamps = get_speech_timestamps(
            wav,
            vad_model,
            sampling_rate=16000,
            threshold=max(0.1, cfg.vad_threshold - 0.3),
            min_speech_duration_ms=50,
            min_silence_duration_ms=50,
            return_seconds=True,
        )

    if not timestamps:
        # 그래도 없으면 전체 유지
        return [(0.0, total_sec)]

    result = []
    for t in timestamps:
        s = max(0.0, float(t["start"]) - pad)
        e = min(total_sec, float(t["end"]) + pad)
        result.append((s, e))

    return merge_intervals(result)


# =========================================================
# 오디오 마스킹
# =========================================================
def apply_audio_mask(
    audio: np.ndarray,
    sr: int,
    keep: List[Tuple[float, float]],
) -> np.ndarray:
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
# 단일 영상 처리
# =========================================================
def process_one_video(
    video_path: Path,
    output_dir: Path,
    cfg: Config,
    demucs_model,
    df_model,
    df_state,
    vad_model,
    get_speech_timestamps,
    read_audio,
) -> dict:
    stem        = video_path.stem
    output_path = output_dir / video_path.name

    with tempfile.TemporaryDirectory(prefix=f"aud_{stem}_") as tmpdir:
        tmp = Path(tmpdir)

        # ── 오디오 추출 ──
        audio_44k = tmp / "audio_44k.wav"
        run_cmd(f"ffmpeg -y -i {video_path} -ac 2 -ar 44100 "
                f"{audio_44k} -loglevel panic")

        # 총 길이 계산
        import subprocess as sp
        probe = sp.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
            capture_output=True, text=True
        )
        try:
            total_sec = float(probe.stdout.strip())
        except Exception:
            total_sec = 30.0  # fallback

        # ── Step 1: Demucs — vocals 추출 ──
        vocals_44k = tmp / "vocals_44k.wav"
        demucs_extract_vocals(audio_44k, vocals_44k, demucs_model, cfg.device)

        # ── Step 2: DeepFilterNet — noise 제거 ──
        denoised_path = tmp / "denoised.wav"
        deepfilter_denoise(vocals_44k, denoised_path, df_model, df_state)

        # VAD용 16kHz mono 변환
        denoised_16k = tmp / "denoised_16k.wav"
        run_cmd(f"ffmpeg -y -i {denoised_path} -ac 1 -ar 16000 "
                f"{denoised_16k} -loglevel panic")

        # ── Step 3: Silero VAD — 발화 구간 타이트하게 ──
        vad_intervals = silero_vad_segments(
            denoised_16k, vad_model, get_speech_timestamps,
            read_audio, cfg, total_sec,
        )

        # ── 마스킹 적용 ──
        wav_tensor, sr = torchaudio.load(str(denoised_path))
        masked_np   = apply_audio_mask(wav_tensor.numpy(), sr, vad_intervals)
        masked_path = tmp / "masked.wav"
        torchaudio.save(str(masked_path), torch.from_numpy(masked_np), sr)

        # ── 원본 영상 + masked audio mux ──
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
    ap = argparse.ArgumentParser(
        description="MELD audio intervention v4: Demucs + DeepFilterNet + Silero VAD"
    )
    ap.add_argument("--input_dir",  required=True,
                    help="C1 visual masked 영상 디렉토리")
    ap.add_argument("--output_dir", required=True,
                    help="출력 디렉토리")
    ap.add_argument("--device",     default="cuda")
    ap.add_argument("--limit",      type=int, default=0)
    ap.add_argument("--vad_threshold",    type=float, default=0.6,
                    help="Silero VAD 민감도 (높을수록 타이트, 기본 0.6)")
    ap.add_argument("--vad_pad_ms",       type=int,   default=30,
                    help="발화 앞뒤 패딩 ms (기본 30ms, 타이트)")
    ap.add_argument("--vad_min_speech_ms",type=int,   default=80)
    ap.add_argument("--vad_min_silence_ms",type=int,  default=80)
    ap.add_argument("--shutdown_when_done", action="store_true")
    args = ap.parse_args()

    ensure_ffmpeg()

    cfg = Config(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        device=args.device,
        limit=args.limit,
        vad_threshold=args.vad_threshold,
        vad_pad_ms=args.vad_pad_ms,
        vad_min_speech_ms=args.vad_min_speech_ms,
        vad_min_silence_ms=args.vad_min_silence_ms,
        shutdown_when_done=args.shutdown_when_done,
    )

    # ── 모델 한 번만 로드 ──
    demucs_model                             = load_demucs_model(cfg.device)
    df_model, df_state                       = load_deepfilter_model()
    vad_model, get_speech_timestamps, read_audio = load_silero_vad()

    input_dir  = Path(cfg.input_dir)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    videos = find_videos(input_dir)
    if cfg.limit > 0:
        videos = videos[:cfg.limit]

    print(f"\n[videos] {len(videos)}개")
    print(f"[pipeline] Demucs → DeepFilterNet → Silero VAD (threshold={cfg.vad_threshold}, pad={cfg.vad_pad_ms}ms)\n")

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

    log_path = output_dir / "audio_intervention_log.csv"
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