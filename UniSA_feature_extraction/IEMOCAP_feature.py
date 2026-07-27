#!/usr/bin/env python3
"""
UniSA feature extraction for IEMOCAP Session 5.

Supported conditions
--------------------
C0, C1_visual, C2_audio

Examples
--------
export MM_BASE_DIR=/path/to/CEP-MER

python IEMOCAP_feature.py --condition C0
python IEMOCAP_feature.py --condition C1_visual
python IEMOCAP_feature.py --condition C2_audio

Notes
-----
- The original IEMOCAP test CSV must be used because the excited class is
  retained during UniSA inference.
- UniSA uses six labels internally. The excited class may be merged into
  happy only during the final five-class evaluation.
- Use --portable-lists when the target environment cannot unpickle NumPy
  arrays created by a different NumPy major version.
"""

from __future__ import annotations

import argparse
import csv
import os
import pickle
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import librosa
import numpy as np
from tqdm import tqdm


SUPPORTED_CONDITIONS = (
    "C0",
    "C1_visual",
    "C2_audio",
)

UNISA_FEAT_DIM = 64
UNISA_MAX_IMG = 32
UNISA_MAX_AUD = 157

LABEL_MAP = {
    "ang": "angry",
    "hap": "happy",
    "exc": "excited",
    "sad": "sad",
    "neu": "neutral",
    "fru": "frustrate",
}


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_base = Path(
        os.environ.get("MM_BASE_DIR", script_dir.parent)
    ).resolve()

    parser = argparse.ArgumentParser(
        description="Extract UniSA-compatible IEMOCAP features."
    )
    parser.add_argument(
        "--condition",
        required=True,
        choices=SUPPORTED_CONDITIONS,
        help="Intervention condition to process.",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=default_base,
        help=(
            "Project root. Defaults to MM_BASE_DIR or the parent directory "
            "of this script."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Optional output root. Defaults to <base-dir>/extracted_features.",
    )
    parser.add_argument(
        "--overwrite-cache",
        action="store_true",
        help="Recompute cached NPZ files.",
    )
    parser.add_argument(
        "--portable-lists",
        action="store_true",
        help=(
            "Store feature arrays as Python lists for cross-NumPy "
            "compatibility."
        ),
    )
    return parser.parse_args()


def build_paths(
    base_dir: Path,
    condition: str,
    output_root: Path | None,
) -> dict[str, Path]:
    output_root = (
        output_root.resolve()
        if output_root is not None
        else base_dir / "extracted_features"
    )

    video_map = {
        "C0": base_dir / "IEMOCAP/Session5/sentences/avi_flat",
        "C1_visual": (
            base_dir / "IEMOCAP_intervention/C1_visual"
        ),
        "C2_audio": (
            base_dir / "IEMOCAP_intervention/C2_audio"
        ),
    }

    return {
        "csv_path": (
            base_dir
            / "IEMOCAP/Session5/sentences/iemocap_test.csv"
        ),
        "video_dir": video_map[condition],
        "output_dir": output_root / f"iemocap_{condition}",
    }


def load_iemocap_metadata(
    csv_path: Path,
) -> dict[int, list[dict[str, Any]]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"IEMOCAP CSV not found: {csv_path}")

    dialogue_data: dict[int, list[dict[str, Any]]] = defaultdict(list)
    dialogue_id_map: dict[str, int] = {}
    next_dialogue_id = 0

    with csv_path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            dialogue_name = row["dialogue_id"]

            if dialogue_name not in dialogue_id_map:
                dialogue_id_map[dialogue_name] = next_dialogue_id
                next_dialogue_id += 1

            dialogue_id = dialogue_id_map[dialogue_name]
            utterance_id = int(row["turn"])

            dialogue_data[dialogue_id].append(
                {
                    "dialogue_id": dialogue_id,
                    "utterance_id": utterance_id,
                    "utt_id": row["utt_id"],
                    "speaker": row.get("speaker", "Unknown"),
                    "emotion": row.get("emotion", "neu").lower(),
                    "text": row.get("text", ""),
                }
            )

    for dialogue_id in dialogue_data:
        dialogue_data[dialogue_id].sort(
            key=lambda item: item["utterance_id"]
        )

    return dict(dialogue_data)


def extract_visual_unisa(
    video_path: Path,
) -> tuple[np.ndarray, int]:
    """Return uniformly sampled visual features with shape (32, 64)."""
    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise RuntimeError("OpenCV could not open the video.")

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    if frame_count <= 0:
        capture.release()
        raise RuntimeError(f"Invalid frame count: {frame_count}")

    sample_count = min(frame_count, UNISA_MAX_IMG)
    frame_indices = np.linspace(
        0,
        frame_count - 1,
        sample_count,
        dtype=int,
    )

    frame_features: list[np.ndarray] = []

    for frame_index in frame_indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        success, frame_bgr = capture.read()

        if not success or frame_bgr is None:
            continue

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(frame_rgb, (8, 8))

        feature = (
            resized.astype(np.float32)
            .reshape(-1, 3)
            .mean(axis=1)
        )

        if feature.shape != (UNISA_FEAT_DIM,):
            capture.release()
            raise RuntimeError(
                f"Unexpected visual feature shape: {feature.shape}"
            )

        frame_features.append(feature)

    capture.release()

    if not frame_features:
        raise RuntimeError("No readable video frames were found.")

    result = np.zeros(
        (UNISA_MAX_IMG, UNISA_FEAT_DIM),
        dtype=np.float32,
    )
    valid_length = min(len(frame_features), UNISA_MAX_IMG)
    result[:valid_length] = np.stack(
        frame_features[:valid_length],
        axis=0,
    )

    if not np.isfinite(result).all():
        raise RuntimeError("Visual features contain NaN or Inf.")

    return result, valid_length


def extract_audio_unisa(
    video_path: Path,
) -> tuple[np.ndarray, int]:
    """Return log-Mel audio features with shape (157, 64)."""
    with tempfile.TemporaryDirectory(prefix="unisa_iemocap_") as temp_dir:
        wav_path = Path(temp_dir) / f"{video_path.stem}.wav"

        command = [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            str(video_path),
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            str(wav_path),
        ]

        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )

        if process.returncode != 0:
            raise RuntimeError(
                "ffmpeg audio extraction failed: "
                + process.stderr.strip()
            )

        waveform, sample_rate = librosa.load(
            wav_path,
            sr=16000,
            mono=True,
        )

        if waveform.size == 0:
            raise RuntimeError("The decoded waveform is empty.")

        mel = librosa.feature.melspectrogram(
            y=waveform,
            sr=sample_rate,
            n_mels=UNISA_FEAT_DIM,
            hop_length=512,
        )

        mel_db = librosa.power_to_db(
            mel,
            ref=np.max,
        ).T.astype(np.float32)

        result = np.zeros(
            (UNISA_MAX_AUD, UNISA_FEAT_DIM),
            dtype=np.float32,
        )
        valid_length = min(mel_db.shape[0], UNISA_MAX_AUD)

        if valid_length <= 0:
            raise RuntimeError("No valid Mel frames were generated.")

        result[:valid_length] = mel_db[:valid_length]

        if not np.isfinite(result).all():
            raise RuntimeError("Audio features contain NaN or Inf.")

        return result, valid_length


def convert_arrays_to_lists(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {
            key: convert_arrays_to_lists(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [convert_arrays_to_lists(item) for item in value]
    return value


def save_pickle(
    data: list[dict[str, Any]],
    output_path: Path,
    portable_lists: bool,
) -> None:
    payload = (
        convert_arrays_to_lists(data)
        if portable_lists
        else data
    )

    with output_path.open("wb") as file:
        pickle.dump(payload, file, protocol=2)


def process_condition(
    condition: str,
    dialogue_data: dict[int, list[dict[str, Any]]],
    video_dir: Path,
    output_dir: Path,
    overwrite_cache: bool,
    portable_lists: bool,
) -> None:
    if not video_dir.exists():
        raise FileNotFoundError(f"Video directory not found: {video_dir}")

    cache_dir = output_dir / "unisa_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    total_utterances = sum(
        len(utterances)
        for utterances in dialogue_data.values()
    )

    records: list[dict[str, Any]] = []
    missing_files = 0
    failed_files = 0
    visual_zero = 0
    audio_zero = 0
    global_index = 0

    progress = tqdm(
        total=total_utterances,
        desc=f"UniSA IEMOCAP [{condition}]",
    )

    for dialogue_id in sorted(dialogue_data):
        context_parts: list[str] = []

        for utterance in dialogue_data[dialogue_id]:
            utterance_id = utterance["utt_id"]
            video_path = video_dir / f"{utterance_id}.avi"
            cache_path = cache_dir / f"{utterance_id}.npz"

            visual_feature = np.zeros(
                (UNISA_MAX_IMG, UNISA_FEAT_DIM),
                dtype=np.float32,
            )
            audio_feature = np.zeros(
                (UNISA_MAX_AUD, UNISA_FEAT_DIM),
                dtype=np.float32,
            )

            try:
                if cache_path.exists() and not overwrite_cache:
                    cached = np.load(cache_path, allow_pickle=False)
                    visual_feature = cached["visual"].astype(np.float32)
                    audio_feature = cached["audio"].astype(np.float32)
                elif video_path.exists():
                    visual_feature, visual_length = extract_visual_unisa(
                        video_path
                    )
                    audio_feature, audio_length = extract_audio_unisa(
                        video_path
                    )

                    np.savez_compressed(
                        cache_path,
                        visual=visual_feature,
                        audio=audio_feature,
                        visual_length=visual_length,
                        audio_length=audio_length,
                        source_file=video_path.name,
                    )
                else:
                    missing_files += 1
                    tqdm.write(f"[WARN] Missing video: {video_path}")

            except Exception as error:
                failed_files += 1
                tqdm.write(
                    f"[WARN] Feature extraction failed for "
                    f"{video_path.name}: {error}"
                )

            visual_zero += int(np.allclose(visual_feature, 0))
            audio_zero += int(np.allclose(audio_feature, 0))

            raw_label = utterance["emotion"]
            mapped_label = LABEL_MAP.get(raw_label, raw_label)

            records.append(
                {
                    "image_features": visual_feature,
                    "audio_features": audio_feature,
                    "text": utterance["text"],
                    "label": mapped_label,
                    "task_type": "erc",
                    "speaker": utterance["speaker"],
                    "context": (
                        " ".join(context_parts)
                        if context_parts
                        else ""
                    ),
                    "index": global_index,
                }
            )

            context_parts.append(
                f"{utterance['speaker']}: {utterance['text']}"
            )
            global_index += 1
            progress.update(1)

    progress.close()

    output_path = output_dir / f"unisa_{condition}_data.pkl"
    save_pickle(records, output_path, portable_lists)

    invalid_records = 0

    for item in records:
        image = np.asarray(
            item["image_features"],
            dtype=np.float32,
        )
        audio = np.asarray(
            item["audio_features"],
            dtype=np.float32,
        )

        if (
            image.shape != (UNISA_MAX_IMG, UNISA_FEAT_DIM)
            or audio.shape != (UNISA_MAX_AUD, UNISA_FEAT_DIM)
            or not np.isfinite(image).all()
            or not np.isfinite(audio).all()
        ):
            invalid_records += 1

    label_counts: dict[str, int] = {}

    for record in records:
        label = record["label"]
        label_counts[label] = label_counts.get(label, 0) + 1

    print("\nExtraction summary")
    print("------------------")
    print(f"Condition        : {condition}")
    print(f"Records          : {len(records)}")
    print(f"Missing videos   : {missing_files}")
    print(f"Failed videos    : {failed_files}")
    print(f"Visual all-zero  : {visual_zero}")
    print(f"Audio all-zero   : {audio_zero}")
    print(f"Invalid records  : {invalid_records}")
    print(f"Label counts     : {label_counts}")
    print(f"Output pickle    : {output_path}")

    if invalid_records:
        raise RuntimeError(
            f"{invalid_records} invalid UniSA records were generated."
        )


def main() -> None:
    args = parse_args()
    base_dir = args.base_dir.resolve()
    paths = build_paths(
        base_dir=base_dir,
        condition=args.condition,
        output_root=args.output_root,
    )

    output_dir = Path(paths["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    dialogue_data = load_iemocap_metadata(
        Path(paths["csv_path"])
    )

    process_condition(
        condition=args.condition,
        dialogue_data=dialogue_data,
        video_dir=Path(paths["video_dir"]),
        output_dir=output_dir,
        overwrite_cache=args.overwrite_cache,
        portable_lists=args.portable_lists,
    )


if __name__ == "__main__":
    main()
