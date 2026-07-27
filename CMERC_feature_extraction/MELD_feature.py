#!/usr/bin/env python3
"""
CMERC feature extraction and PKL construction for MELD.

Supported conditions
--------------------
C0, C1_scene, C1_social, C2_v1, C2_v2, C3_strong, C3_weak

Modality-isolated processing
----------------------------
- C0 copies the original CMERC feature files.
- C1 conditions replace only visual features.
- C2 conditions replace only audio features.
- C3 conditions replace only RoBERTa text features.

Examples
--------
export MM_BASE_DIR=/path/to/CEP-MER

python MELD_feature.py --condition C0
python MELD_feature.py --condition C1_scene
python MELD_feature.py --condition C2_v1
python MELD_feature.py --condition C3_strong

The default outputs are the filenames expected by the original CMERC
inference script:

    CMERC/data/MELD_features_raw1.pkl
    CMERC/data/meld_features_roberta.pkl

Important
---------
The visual/audio extraction functions in this file are a cleaned and
modality-isolated refactoring of the reconstructed pipeline used in the
project workspace. Verify that these functions match the exact feature
pipeline used for the final reported experiments before public release.
"""

from __future__ import annotations

import argparse
import copy
import csv
import os
import pickle
import shutil
import subprocess
import tempfile
import warnings
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from tqdm import tqdm


SUPPORTED_CONDITIONS = (
    "C0",
    "C1_scene",
    "C1_social",
    "C2_v1",
    "C2_v2",
    "C3_strong",
    "C3_weak",
)

VISUAL_CONDITIONS = {
    "C1_scene",
    "C1_social",
}

AUDIO_CONDITIONS = {
    "C2_v1",
    "C2_v2",
}

TEXT_CONDITIONS = {
    "C3_strong",
    "C3_weak",
}

VISUAL_DIM = 342
AUDIO_DIM = 300
TEXT_DIM = 1024
ROBERTA_LAYERS = (17, 18, 19, 20)

warnings.filterwarnings("ignore")


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_base = Path(
        os.environ.get("MM_BASE_DIR", script_dir.parent)
    ).resolve()

    parser = argparse.ArgumentParser(
        description="Build CMERC-compatible MELD feature PKLs."
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
            "Project root. Defaults to MM_BASE_DIR or the parent "
            "directory of this script."
        ),
    )
    parser.add_argument(
        "--original-raw",
        type=Path,
        default=None,
        help=(
            "Original CMERC raw-feature PKL. Defaults to "
            "<base-dir>/CMERC/data/MELD_features_raw1_original.pkl."
        ),
    )
    parser.add_argument(
        "--original-roberta",
        type=Path,
        default=None,
        help=(
            "Original CMERC RoBERTa PKL. Defaults to "
            "<base-dir>/CMERC/data/meld_features_roberta_original.pkl."
        ),
    )
    parser.add_argument(
        "--output-raw",
        type=Path,
        default=None,
        help=(
            "Output raw-feature PKL. Defaults to "
            "<base-dir>/CMERC/data/MELD_features_raw1.pkl."
        ),
    )
    parser.add_argument(
        "--output-roberta",
        type=Path,
        default=None,
        help=(
            "Output RoBERTa PKL. Defaults to "
            "<base-dir>/CMERC/data/meld_features_roberta.pkl."
        ),
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=None,
        help=(
            "Cache root. Defaults to "
            "<base-dir>/extracted_features/cmerc_meld_<condition>/cache."
        ),
    )
    parser.add_argument(
        "--visual-stats",
        type=Path,
        default=None,
        help=(
            "Optional NPZ containing 'mean' and 'std' arrays for "
            "visual distribution matching."
        ),
    )
    parser.add_argument(
        "--overwrite-cache",
        action="store_true",
        help="Recompute cached features.",
    )
    parser.add_argument(
        "--on-error",
        choices=("raise", "keep-original"),
        default="keep-original",
        help=(
            "How to handle a missing video or extraction failure. "
            "The default preserves the original CMERC feature."
        ),
    )
    parser.add_argument(
        "--roberta-indices",
        type=int,
        nargs=4,
        default=(3, 4, 5, 6),
        metavar=("IDX1", "IDX2", "IDX3", "IDX4"),
        help=(
            "Tuple positions containing the four RoBERTa feature "
            "dictionaries. Defaults to the indices used by the project "
            "MELD pipeline: 3 4 5 6."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for the fixed visual projection.",
    )
    return parser.parse_args()


def build_paths(args: argparse.Namespace) -> dict[str, Path | None]:
    base_dir = args.base_dir.resolve()

    video_map: dict[str, Path | None] = {
        "C0": base_dir / "MELD/original/video/test",
        "C1_scene": (
            base_dir
            / "MELD_intervention/C1_scene/video/test"
        ),
        "C1_social": (
            base_dir
            / "MELD_intervention/C1_social/video/test"
        ),
        "C2_v1": (
            base_dir
            / "MELD_intervention/C2_v1/video/test"
        ),
        "C2_v2": (
            base_dir
            / "MELD_intervention/C2_v2/video/test"
        ),
        "C3_strong": None,
        "C3_weak": None,
    }

    c3_csv_map: dict[str, Path | None] = {
        "C0": None,
        "C1_scene": None,
        "C1_social": None,
        "C2_v1": None,
        "C2_v2": None,
        "C3_strong": (
            base_dir
            / "MELD_intervention/C3_pivot/"
            "reinference_input_strong.csv"
        ),
        "C3_weak": (
            base_dir
            / "MELD_intervention/C3_pivot/"
            "reinference_input_weak.csv"
        ),
    }

    original_raw = (
        args.original_raw.resolve()
        if args.original_raw is not None
        else base_dir
        / "CMERC/data/MELD_features_raw1_original.pkl"
    )
    original_roberta = (
        args.original_roberta.resolve()
        if args.original_roberta is not None
        else base_dir
        / "CMERC/data/meld_features_roberta_original.pkl"
    )
    output_raw = (
        args.output_raw.resolve()
        if args.output_raw is not None
        else base_dir
        / "CMERC/data/MELD_features_raw1.pkl"
    )
    output_roberta = (
        args.output_roberta.resolve()
        if args.output_roberta is not None
        else base_dir
        / "CMERC/data/meld_features_roberta.pkl"
    )
    cache_root = (
        args.cache_root.resolve()
        if args.cache_root is not None
        else base_dir
        / f"extracted_features/cmerc_meld_{args.condition}/cache"
    )
    visual_stats = (
        args.visual_stats.resolve()
        if args.visual_stats is not None
        else base_dir
        / "extracted_features/orig_visual_stats.npz"
    )

    return {
        "base_dir": base_dir,
        "test_csv": base_dir / "MELD/original/test_sent_emo.csv",
        "video_dir": video_map[args.condition],
        "c3_csv": c3_csv_map[args.condition],
        "original_raw": original_raw,
        "original_roberta": original_roberta,
        "output_raw": output_raw,
        "output_roberta": output_roberta,
        "cache_root": cache_root,
        "visual_stats": visual_stats,
    }


def load_pickle(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Required PKL not found: {path}")

    with path.open("rb") as file:
        try:
            return pickle.load(file, encoding="latin1")
        except TypeError:
            file.seek(0)
            return pickle.load(file)


def save_pickle(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("wb") as file:
        pickle.dump(data, file, protocol=2)


def copy_feature_file(source: Path, destination: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(f"Required file not found: {source}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def load_test_rows(csv_path: Path) -> list[dict[str, int]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"MELD test CSV not found: {csv_path}")

    rows: list[dict[str, int]] = []

    with csv_path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        for row in csv.DictReader(file):
            rows.append(
                {
                    "dialogue_id": int(row["Dialogue_ID"]),
                    "utterance_id": int(row["Utterance_ID"]),
                }
            )

    return rows


def build_position_mapping(
    original_raw: Any,
    original_roberta: Any,
) -> tuple[list[Any], list[tuple[Any, int]]]:
    try:
        test_dialogues = sorted(original_raw[8])
        roberta_anchor = original_roberta[7]
    except (IndexError, TypeError) as error:
        raise RuntimeError(
            "Unexpected CMERC MELD PKL structure."
        ) from error

    if not isinstance(roberta_anchor, dict):
        raise RuntimeError(
            "RoBERTa tuple index 7 is expected to be a dialogue dictionary."
        )

    position_to_location: list[tuple[Any, int]] = []

    for dialogue_key in test_dialogues:
        if dialogue_key not in roberta_anchor:
            raise RuntimeError(
                f"Dialogue key missing from RoBERTa anchor: {dialogue_key}"
            )

        for utterance_order in range(
            len(roberta_anchor[dialogue_key])
        ):
            position_to_location.append(
                (dialogue_key, utterance_order)
            )

    return test_dialogues, position_to_location


def validate_roberta_indices(
    original_roberta: Any,
    indices: tuple[int, int, int, int],
    test_dialogues: list[Any],
) -> None:
    for index in indices:
        if index >= len(original_roberta):
            raise RuntimeError(
                f"RoBERTa tuple index {index} is out of range."
            )

        item = original_roberta[index]

        if not isinstance(item, dict):
            raise RuntimeError(
                f"RoBERTa tuple index {index} is not a dictionary. "
                "Confirm --roberta-indices before running C3."
            )

        for dialogue_key in test_dialogues[:1]:
            if dialogue_key not in item:
                raise RuntimeError(
                    f"Dialogue key {dialogue_key} is missing from "
                    f"RoBERTa tuple index {index}."
                )


def run_c0(
    original_raw_path: Path,
    original_roberta_path: Path,
    output_raw_path: Path,
    output_roberta_path: Path,
) -> None:
    copy_feature_file(original_raw_path, output_raw_path)
    copy_feature_file(original_roberta_path, output_roberta_path)

    print("C0 feature files copied.")
    print(f"Raw output      : {output_raw_path}")
    print(f"RoBERTa output  : {output_roberta_path}")


def run_c3(
    condition: str,
    c3_csv_path: Path,
    original_raw: Any,
    original_roberta: Any,
    original_raw_path: Path,
    output_raw_path: Path,
    output_roberta_path: Path,
    test_dialogues: list[Any],
    position_to_location: list[tuple[Any, int]],
    roberta_indices: tuple[int, int, int, int],
    device: torch.device,
) -> None:
    from transformers import RobertaModel, RobertaTokenizer

    if not c3_csv_path.exists():
        raise FileNotFoundError(
            f"C3 replacement CSV not found: {c3_csv_path}"
        )

    validate_roberta_indices(
        original_roberta,
        roberta_indices,
        test_dialogues,
    )

    modifications: dict[int, str] = {}

    with c3_csv_path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        for row in csv.DictReader(file):
            modifications[int(row["index"])] = row["modified_text"]

    print(f"Loading RoBERTa-large on {device}...")
    tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
    model = RobertaModel.from_pretrained(
        "roberta-large",
        output_hidden_states=True,
    ).to(device).eval()

    new_roberta = [
        copy.deepcopy(item)
        if isinstance(item, dict)
        else item
        for item in original_roberta
    ]

    replaced = 0
    skipped = 0

    for global_index, modified_text in tqdm(
        sorted(modifications.items()),
        desc=f"CMERC MELD [{condition}] text",
    ):
        if global_index >= len(position_to_location):
            skipped += 1
            continue

        dialogue_key, utterance_order = (
            position_to_location[global_index]
        )

        inputs = tokenizer(
            modified_text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(device)

        with torch.inference_mode():
            outputs = model(**inputs)

        hidden_states = outputs.hidden_states

        for layer_index, tuple_index in zip(
            ROBERTA_LAYERS,
            roberta_indices,
        ):
            feature = (
                hidden_states[layer_index][0, 0, :]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )

            target = new_roberta[tuple_index][dialogue_key]
            target[utterance_order] = feature

        replaced += 1

    copy_feature_file(original_raw_path, output_raw_path)

    output_container = (
        tuple(new_roberta)
        if isinstance(original_roberta, tuple)
        else new_roberta
    )
    save_pickle(output_container, output_roberta_path)

    changed = 0
    first_index = roberta_indices[0]

    for dialogue_key in test_dialogues:
        for utterance_order in range(
            len(original_roberta[first_index][dialogue_key])
        ):
            before = np.asarray(
                original_roberta[first_index]
                [dialogue_key][utterance_order]
            )
            after = np.asarray(
                new_roberta[first_index]
                [dialogue_key][utterance_order]
            )

            if not np.allclose(before, after, atol=1e-5):
                changed += 1

    print("\nText intervention summary")
    print("-------------------------")
    print(f"Condition        : {condition}")
    print(f"Requested        : {len(modifications)}")
    print(f"Replaced         : {replaced}")
    print(f"Skipped          : {skipped}")
    print(f"Changed records  : {changed}")
    print(f"Raw output       : {output_raw_path}")
    print(f"RoBERTa output   : {output_roberta_path}")


class VisualExtractor:
    """DenseNet-based visual feature extractor used by the project pipeline."""

    def __init__(
        self,
        device: torch.device,
        seed: int,
        stats_path: Path | None,
    ) -> None:
        import cv2
        import timm
        from torchvision import transforms

        self.cv2 = cv2
        self.device = device
        self.model = timm.create_model(
            "densenet121",
            pretrained=True,
            num_classes=0,
        ).to(device).eval()

        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades
            + "haarcascade_frontalface_default.xml"
        )

        if self.face_cascade.empty():
            raise RuntimeError(
                "OpenCV face detector could not be loaded."
            )

        projection_rng = np.random.RandomState(seed)
        self.projection = (
            projection_rng.randn(1024, VISUAL_DIM)
            .astype(np.float32)
            * 0.01
        )

        self.original_mean: np.ndarray | None = None
        self.original_std: np.ndarray | None = None

        if stats_path is not None and stats_path.exists():
            stats = np.load(stats_path)
            self.original_mean = np.asarray(
                stats["mean"],
                dtype=np.float32,
            )
            self.original_std = np.asarray(
                stats["std"],
                dtype=np.float32,
            )

    def _extract_frame(self, frame_rgb: np.ndarray) -> np.ndarray:
        tensor = (
            self.transform(frame_rgb)
            .unsqueeze(0)
            .to(self.device)
        )

        with torch.inference_mode():
            feature = self.model(tensor)

        feature = (
            feature.squeeze(0)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        if feature.shape != (1024,):
            raise RuntimeError(
                f"Unexpected DenseNet feature shape: {feature.shape}"
            )

        return feature

    def _project(self, feature: np.ndarray) -> np.ndarray:
        projected = feature @ self.projection
        projected = np.maximum(projected, 0)

        if (
            self.original_mean is not None
            and self.original_std is not None
        ):
            projected = (
                projected - projected.mean()
            ) / (projected.std() + 1e-8)
            projected = (
                projected * self.original_std.mean()
                + self.original_mean.mean()
            )
            projected = np.maximum(projected, 0)

        projected = np.nan_to_num(
            projected,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        return projected.astype(np.float64)

    def __call__(self, video_path: Path) -> np.ndarray:
        cv2 = self.cv2
        capture = cv2.VideoCapture(str(video_path))

        if not capture.isOpened():
            raise RuntimeError("OpenCV could not open the video.")

        frame_count = int(
            capture.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        if frame_count <= 0:
            capture.release()
            raise RuntimeError(f"Invalid frame count: {frame_count}")

        sample_count = min(8, frame_count)
        frame_indices = np.linspace(
            0,
            frame_count - 1,
            sample_count,
            dtype=int,
        )

        full_features: list[np.ndarray] = []
        face_features: list[np.ndarray] = []

        for frame_index in frame_indices:
            capture.set(
                cv2.CAP_PROP_POS_FRAMES,
                int(frame_index),
            )
            success, frame_bgr = capture.read()

            if not success or frame_bgr is None:
                continue

            frame_rgb = cv2.cvtColor(
                frame_bgr,
                cv2.COLOR_BGR2RGB,
            )

            full_features.append(
                self._extract_frame(frame_rgb)
            )

            gray = cv2.cvtColor(
                frame_bgr,
                cv2.COLOR_BGR2GRAY,
            )
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
            )

            if len(faces) == 0:
                continue

            x, y, width, height = max(
                faces,
                key=lambda box: box[2] * box[3],
            )
            margin = int(0.1 * max(width, height))

            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(
                frame_rgb.shape[1],
                x + width + margin,
            )
            y2 = min(
                frame_rgb.shape[0],
                y + height + margin,
            )

            face_crop = frame_rgb[y1:y2, x1:x2]

            if face_crop.size > 0:
                face_features.append(
                    self._extract_frame(face_crop)
                )

        capture.release()

        if not full_features:
            raise RuntimeError(
                "No readable frames were found."
            )

        full_average = np.mean(full_features, axis=0)

        if face_features:
            combined = (
                0.7 * np.mean(face_features, axis=0)
                + 0.3 * full_average
            )
        else:
            combined = full_average

        feature = self._project(combined)

        if feature.shape != (VISUAL_DIM,):
            raise RuntimeError(
                f"Unexpected visual shape: {feature.shape}"
            )

        if not np.isfinite(feature).all():
            raise RuntimeError(
                "Visual feature contains NaN or Inf."
            )

        return feature


class AudioExtractor:
    """openSMILE-based audio feature extractor used by the project pipeline."""

    def __init__(self) -> None:
        import opensmile

        self.smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.IS10,
            feature_level=(
                opensmile.FeatureLevel.LowLevelDescriptors
            ),
        )

        print(
            "openSMILE raw feature dimension: "
            f"{len(self.smile.feature_names)}"
        )

    def __call__(self, video_path: Path) -> np.ndarray:
        with tempfile.TemporaryDirectory(
            prefix="cmerc_meld_"
        ) as temp_dir:
            wav_path = Path(temp_dir) / f"{video_path.stem}.wav"

            command = [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                "-i",
                str(video_path),
                "-vn",
                "-ar",
                "16000",
                "-ac",
                "1",
                "-acodec",
                "pcm_s16le",
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

            if (
                not wav_path.exists()
                or wav_path.stat().st_size < 200
            ):
                raise RuntimeError(
                    "The decoded WAV file is missing or empty."
                )

            dataframe = self.smile.process_file(
                str(wav_path)
            )

            if dataframe.empty:
                raise RuntimeError(
                    "openSMILE returned an empty feature frame."
                )

            feature = (
                dataframe.values.mean(axis=0)
                .astype(np.float32)
            )

            if feature.shape[0] > AUDIO_DIM:
                feature = feature[:AUDIO_DIM]
            elif feature.shape[0] < AUDIO_DIM:
                feature = np.pad(
                    feature,
                    (0, AUDIO_DIM - feature.shape[0]),
                    mode="constant",
                )

            feature = np.nan_to_num(
                feature,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            feature = np.clip(
                feature,
                -1.0,
                1.0,
            ).astype(np.float32)

            if feature.shape != (AUDIO_DIM,):
                raise RuntimeError(
                    f"Unexpected audio shape: {feature.shape}"
                )

            if not np.isfinite(feature).all():
                raise RuntimeError(
                    "Audio feature contains NaN or Inf."
                )

            return feature


def copy_original_feature(
    original_raw: Any,
    modality_index: int,
    dialogue_key: Any,
    utterance_order: int,
    dtype: np.dtype[Any],
) -> np.ndarray:
    return np.asarray(
        original_raw[modality_index]
        [dialogue_key][utterance_order],
        dtype=dtype,
    ).copy()


def process_modality_condition(
    condition: str,
    modality: str,
    test_rows: list[dict[str, int]],
    position_to_location: list[tuple[Any, int]],
    test_dialogues: list[Any],
    video_dir: Path,
    cache_root: Path,
    original_raw: Any,
    original_roberta_path: Path,
    output_raw_path: Path,
    output_roberta_path: Path,
    extractor: Callable[[Path], np.ndarray],
    overwrite_cache: bool,
    on_error: str,
) -> None:
    if not video_dir.exists():
        raise FileNotFoundError(
            f"Intervention video directory not found: {video_dir}"
        )

    if len(test_rows) != len(position_to_location):
        raise RuntimeError(
            "MELD CSV utterance count does not match the CMERC "
            f"test mapping: {len(test_rows)} vs "
            f"{len(position_to_location)}."
        )

    cache_root.mkdir(parents=True, exist_ok=True)

    modality_index = 5 if modality == "visual" else 4
    expected_shape = (
        (VISUAL_DIM,)
        if modality == "visual"
        else (AUDIO_DIM,)
    )
    expected_dtype = (
        np.float64
        if modality == "visual"
        else np.float32
    )

    extracted: dict[Any, dict[int, np.ndarray]] = {}
    missing_videos = 0
    failed_extractions = 0
    kept_original = 0
    all_zero = 0

    for global_index, row in enumerate(
        tqdm(
            test_rows,
            desc=f"CMERC MELD [{condition}] {modality}",
        )
    ):
        dialogue_key, utterance_order = (
            position_to_location[global_index]
        )
        video_name = (
            f"dia{row['dialogue_id']}_"
            f"utt{row['utterance_id']}.mp4"
        )
        video_path = video_dir / video_name
        cache_path = cache_root / f"{video_path.stem}.npz"

        feature: np.ndarray

        try:
            if cache_path.exists() and not overwrite_cache:
                cached = np.load(
                    cache_path,
                    allow_pickle=False,
                )
                feature = np.asarray(
                    cached[modality],
                    dtype=expected_dtype,
                )
            else:
                if not video_path.exists():
                    missing_videos += 1
                    raise FileNotFoundError(
                        f"Video not found: {video_path}"
                    )

                feature = np.asarray(
                    extractor(video_path),
                    dtype=expected_dtype,
                )

                np.savez_compressed(
                    cache_path,
                    **{
                        modality: feature,
                        "source_file": video_path.name,
                    },
                )

        except Exception as error:
            if on_error == "raise":
                raise RuntimeError(
                    f"{video_name}: {error}"
                ) from error

            failed_extractions += 1
            kept_original += 1
            tqdm.write(
                f"[WARN] {video_name}: {error}; "
                "keeping the original CMERC feature."
            )
            feature = copy_original_feature(
                original_raw=original_raw,
                modality_index=modality_index,
                dialogue_key=dialogue_key,
                utterance_order=utterance_order,
                dtype=expected_dtype,
            )

        if feature.shape != expected_shape:
            raise RuntimeError(
                f"{video_name}: expected {expected_shape}, "
                f"received {feature.shape}."
            )

        if not np.isfinite(feature).all():
            raise RuntimeError(
                f"{video_name}: feature contains NaN or Inf."
            )

        all_zero += int(np.allclose(feature, 0))
        extracted.setdefault(dialogue_key, {})[
            utterance_order
        ] = feature

    merged_raw = list(original_raw)
    merged_raw[4] = dict(original_raw[4])
    merged_raw[5] = dict(original_raw[5])

    for dialogue_key, utterance_features in extracted.items():
        if modality == "visual":
            original_values = original_raw[5][dialogue_key]

            if isinstance(original_values, np.ndarray):
                new_values = [
                    original_values[index].copy()
                    for index in range(len(original_values))
                ]
            else:
                new_values = [
                    np.asarray(value).copy()
                    for value in original_values
                ]

            for utterance_order, feature in (
                utterance_features.items()
            ):
                new_values[utterance_order] = feature.astype(
                    np.float64
                )

            merged_raw[5][dialogue_key] = new_values

        else:
            original_values = np.asarray(
                original_raw[4][dialogue_key]
            )
            new_values = original_values.copy()

            for utterance_order, feature in (
                utterance_features.items()
            ):
                new_values[utterance_order] = feature.astype(
                    np.float32
                )

            merged_raw[4][dialogue_key] = new_values

    output_container = (
        tuple(merged_raw)
        if isinstance(original_raw, tuple)
        else merged_raw
    )
    save_pickle(output_container, output_raw_path)
    copy_feature_file(
        original_roberta_path,
        output_roberta_path,
    )

    if modality == "visual":
        validation_values = np.stack(
            [
                np.asarray(value, dtype=np.float64)
                for dialogue_key in test_dialogues
                for value in merged_raw[5][dialogue_key]
            ]
        )
    else:
        validation_values = np.vstack(
            [
                np.asarray(
                    merged_raw[4][dialogue_key],
                    dtype=np.float32,
                )
                for dialogue_key in test_dialogues
            ]
        )

    invalid_rows = int(
        np.sum(
            ~np.isfinite(validation_values).all(axis=1)
        )
    )

    print("\nFeature extraction summary")
    print("--------------------------")
    print(f"Condition          : {condition}")
    print(f"Replaced modality  : {modality}")
    print(f"Processed records  : {len(test_rows)}")
    print(f"Missing videos     : {missing_videos}")
    print(f"Failed extractions : {failed_extractions}")
    print(f"Kept original      : {kept_original}")
    print(f"All-zero rows      : {all_zero}")
    print(f"Invalid rows       : {invalid_rows}")
    print(f"Feature range      : "
          f"[{validation_values.min():.4f}, "
          f"{validation_values.max():.4f}]")
    print(f"Raw output         : {output_raw_path}")
    print(f"RoBERTa output     : {output_roberta_path}")

    if invalid_rows:
        raise RuntimeError(
            f"{invalid_rows} invalid feature rows were generated."
        )


def print_inference_command(
    base_dir: Path,
    condition: str,
) -> None:
    print("\nCMERC inference")
    print("---------------")
    print(f"cd {base_dir / 'CMERC'}")
    print(
        'sed -i "s/cmerc_.*_predictions.csv/'
        f'cmerc_{condition}_predictions.csv/" train.py'
    )
    print(
        "python train.py --Dataset MELD --testing "
        "--modals avl --multi_modal \\\n"
        "  --base-model GRU --mm_fusion_mthd concat_DHT "
        "--use_modal \\\n"
        "  --graph_type hyper --graph_construct direct "
        "--num_K 3 --seed 67137 --epochs 0"
    )


def main() -> None:
    args = parse_args()
    paths = build_paths(args)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    original_raw_path = Path(paths["original_raw"])
    original_roberta_path = Path(paths["original_roberta"])
    output_raw_path = Path(paths["output_raw"])
    output_roberta_path = Path(paths["output_roberta"])
    base_dir = Path(paths["base_dir"])

    print("=" * 70)
    print("CMERC MELD feature construction")
    print(f"Condition : {args.condition}")
    print(f"Device    : {device}")
    print(f"Base dir  : {base_dir}")
    print("=" * 70)

    if args.condition == "C0":
        run_c0(
            original_raw_path=original_raw_path,
            original_roberta_path=original_roberta_path,
            output_raw_path=output_raw_path,
            output_roberta_path=output_roberta_path,
        )
        print_inference_command(base_dir, args.condition)
        return

    original_raw = load_pickle(original_raw_path)
    original_roberta = load_pickle(original_roberta_path)
    test_dialogues, position_to_location = (
        build_position_mapping(
            original_raw,
            original_roberta,
        )
    )

    if args.condition in TEXT_CONDITIONS:
        run_c3(
            condition=args.condition,
            c3_csv_path=Path(paths["c3_csv"]),
            original_raw=original_raw,
            original_roberta=original_roberta,
            original_raw_path=original_raw_path,
            output_raw_path=output_raw_path,
            output_roberta_path=output_roberta_path,
            test_dialogues=test_dialogues,
            position_to_location=position_to_location,
            roberta_indices=tuple(args.roberta_indices),
            device=device,
        )
        print_inference_command(base_dir, args.condition)
        return

    test_rows = load_test_rows(Path(paths["test_csv"]))
    video_dir = Path(paths["video_dir"])
    cache_root = Path(paths["cache_root"])

    if args.condition in VISUAL_CONDITIONS:
        print("Loading visual feature extractor...")
        extractor = VisualExtractor(
            device=device,
            seed=args.seed,
            stats_path=Path(paths["visual_stats"]),
        )
        modality = "visual"
    elif args.condition in AUDIO_CONDITIONS:
        print("Loading audio feature extractor...")
        extractor = AudioExtractor()
        modality = "audio"
    else:
        raise RuntimeError(
            f"Unsupported condition: {args.condition}"
        )

    process_modality_condition(
        condition=args.condition,
        modality=modality,
        test_rows=test_rows,
        position_to_location=position_to_location,
        test_dialogues=test_dialogues,
        video_dir=video_dir,
        cache_root=cache_root,
        original_raw=original_raw,
        original_roberta_path=original_roberta_path,
        output_raw_path=output_raw_path,
        output_roberta_path=output_roberta_path,
        extractor=extractor,
        overwrite_cache=args.overwrite_cache,
        on_error=args.on_error,
    )

    print_inference_command(base_dir, args.condition)


if __name__ == "__main__":
    main()
