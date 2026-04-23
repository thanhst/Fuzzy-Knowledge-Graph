#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd


PATIENT_ID_PATTERN = re.compile(r"(\d{2}\.\d{4})")


def source_root() -> Path:
    return Path(__file__).resolve().parents[2]


def normalize_brightness(image_bgr: np.ndarray, target_mean: float = 127.5, target_std: float = 50.0) -> np.ndarray:
    image = image_bgr.astype(np.float32)
    current_mean = float(np.mean(image))
    current_std = float(np.std(image))
    if current_std < 1e-6:
        current_std = 1.0
    adjusted = (image - current_mean) * (target_std / current_std) + target_mean
    return np.clip(adjusted, 0, 255).astype(np.uint8)


def apply_clahe(image_bgr: np.ndarray, clip_limit: float = 2.0, tile_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    l2 = clahe.apply(l)
    lab2 = cv2.merge([l2, a, b])
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)


def preprocess_image(image_bgr: np.ndarray, size: int = 512) -> np.ndarray:
    resized = cv2.resize(image_bgr, (size, size), interpolation=cv2.INTER_LINEAR)
    brightness_norm = normalize_brightness(resized, target_mean=127.5, target_std=50.0)
    enhanced = apply_clahe(brightness_norm, clip_limit=2.0, tile_size=(8, 8))
    denoised = cv2.GaussianBlur(enhanced, (3, 3), sigmaX=0.8, sigmaY=0.8)
    return denoised


def run_kmeans_hsv(image_bgr: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0].astype(np.float32) / 180.0
    s = hsv[:, :, 1].astype(np.float32) / 255.0
    v = hsv[:, :, 2].astype(np.float32) / 255.0

    pixels = np.stack([h, s, v], axis=-1).reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 300, 1e-4)
    compactness, labels, centers = cv2.kmeans(
        pixels,
        K=k,
        bestLabels=None,
        criteria=criteria,
        attempts=10,
        flags=cv2.KMEANS_PP_CENTERS,
    )
    _ = compactness
    labels_2d = labels.reshape(hsv.shape[:2])
    return hsv, labels_2d, centers


def hue_in_range(h_deg: float, low: float, high: float) -> bool:
    if low <= high:
        return low <= h_deg <= high
    return h_deg >= low or h_deg <= high


def classify_cluster(center_hsv: np.ndarray) -> Optional[str]:
    h_deg = float(center_hsv[0] * 360.0)
    s = float(center_hsv[1])
    v = float(center_hsv[2])

    # Reference ranges from DLFKG expanded document.
    if hue_in_range(h_deg, 0.0, 15.0) and 0.4 <= s <= 0.65 and 0.5 <= v <= 0.8:
        return "lesion_mild"
    if hue_in_range(h_deg, 355.0, 10.0) and 0.6 <= s <= 0.9 and 0.3 <= v <= 0.6:
        return "lesion_severe"
    if hue_in_range(h_deg, 340.0, 20.0) and 0.2 <= s <= 0.5 and 0.6 <= v <= 0.9:
        return "gingiva_normal"
    if hue_in_range(h_deg, 30.0, 60.0) and 0.0 <= s <= 0.25 and 0.7 <= v <= 1.0:
        return "tooth"
    if hue_in_range(h_deg, 40.0, 80.0) and 0.15 <= s <= 0.4 and 0.4 <= v <= 0.75:
        return "plaque"
    return None


def lesion_cluster_ids(centers: np.ndarray) -> List[int]:
    selected = []
    for i, center in enumerate(centers):
        label = classify_cluster(center)
        if label in {"lesion_mild", "lesion_severe"}:
            selected.append(i)

    if selected:
        return selected

    # Fallback if no cluster hits reference thresholds: pick cluster nearest
    # to lesion prototypes.
    lesion_targets = np.array(
        [
            [0.02, 0.52, 0.65],  # mild lesion prototype
            [0.99, 0.75, 0.45],  # severe lesion prototype (hue wraps around red)
        ],
        dtype=np.float32,
    )
    best_idx = 0
    best_dist = float("inf")
    for i, center in enumerate(centers):
        # Compute circular hue distance to handle wrap-around near 0/1.
        hue_dists = np.minimum(
            np.abs(center[0] - lesion_targets[:, 0]),
            1.0 - np.abs(center[0] - lesion_targets[:, 0]),
        )
        sv_dists = np.sqrt(np.sum((center[1:] - lesion_targets[:, 1:]) ** 2, axis=1))
        dist = float(np.min(hue_dists + sv_dists))
        if dist < best_dist:
            best_dist = dist
            best_idx = i
    return [best_idx]


def build_lesion_mask(labels_2d: np.ndarray, centers: np.ndarray) -> np.ndarray:
    selected = lesion_cluster_ids(centers)
    mask = np.isin(labels_2d, selected).astype(np.uint8) * 255
    kernel = np.ones((3, 3), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask


def extract_features_from_mask(hsv: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    mask_binary = (mask > 0).astype(np.uint8)
    area = int(mask_binary.sum())

    if area == 0:
        return {
            "Area": 0.0,
            "Perimeter": 0.0,
            "Circularity": 0.0,
            "BoundingBoxRatio": 0.0,
            "BoundaryIrregularity": 0.0,
            "MeanHue": 0.0,
            "MeanSaturation": 0.0,
        }

    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    perimeter = float(sum(cv2.arcLength(c, True) for c in contours))

    if perimeter > 1e-9:
        circularity = float((4.0 * math.pi * area) / (perimeter * perimeter))
    else:
        circularity = 0.0

    ys, xs = np.where(mask_binary > 0)
    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())
    bb_w = max(1, x_max - x_min + 1)
    bb_h = max(1, y_max - y_min + 1)
    bbr = float(area / float(bb_w * bb_h))

    if area > 0:
        bi = float(perimeter / (2.0 * math.sqrt(math.pi * area)))
    else:
        bi = 0.0

    hue_deg = hsv[:, :, 0].astype(np.float32) * 2.0
    sat = hsv[:, :, 1].astype(np.float32) / 255.0
    mean_h = float(hue_deg[mask_binary > 0].mean())
    mean_s = float(sat[mask_binary > 0].mean())

    return {
        "Area": float(area),
        "Perimeter": perimeter,
        "Circularity": circularity,
        "BoundingBoxRatio": bbr,
        "BoundaryIrregularity": bi,
        "MeanHue": mean_h,
        "MeanSaturation": mean_s,
    }


def extract_single_image_features(image_path: Path, image_size: int = 512) -> Dict[str, float]:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Cannot read image: {image_path}")
    processed = preprocess_image(image, size=image_size)
    hsv, labels_2d, centers = run_kmeans_hsv(processed, k=5)
    mask = build_lesion_mask(labels_2d, centers)
    return extract_features_from_mask(hsv, mask)


def parse_patient_id(folder_name: str) -> Optional[str]:
    match = PATIENT_ID_PATTERN.search(folder_name)
    if not match:
        return None
    return match.group(1)


def load_table_labels(table_csv: Path) -> pd.DataFrame:
    table_df = pd.read_csv(table_csv, dtype={"patient_id": str})
    if "patient_id" not in table_df.columns or "Outcome" not in table_df.columns:
        raise ValueError("Table CSV must contain 'patient_id' and 'Outcome' columns.")
    labels = table_df[["patient_id", "Outcome"]].copy()
    labels["patient_id"] = labels["patient_id"].astype(str)
    return labels


def process_image_folder(
    images_root: Path,
    table_labels_csv: Path,
    output_dir: Path,
    required_images: int = 5,
    strict_five_images: bool = True,
    image_size: int = 512,
) -> Dict[str, Path]:
    labels_df = load_table_labels(table_labels_csv)
    label_map = dict(zip(labels_df["patient_id"], labels_df["Outcome"]))

    per_image_rows: List[Dict[str, object]] = []
    per_patient_rows: List[Dict[str, object]] = []
    skipped_rows: List[Dict[str, object]] = []

    patient_dirs = [p for p in images_root.iterdir() if p.is_dir()]
    for folder in sorted(patient_dirs, key=lambda p: p.name):
        patient_id = parse_patient_id(folder.name)
        if not patient_id:
            skipped_rows.append({"folder": folder.name, "reason": "invalid_patient_id"})
            continue
        if patient_id not in label_map:
            skipped_rows.append({"folder": folder.name, "reason": "patient_not_in_table"})
            continue

        # Some patient folders store images inside nested subfolders.
        # Use recursive search to avoid silently dropping valid samples.
        image_paths = sorted([p for p in folder.rglob("*.jpg")])
        if strict_five_images and len(image_paths) < required_images:
            skipped_rows.append(
                {
                    "folder": folder.name,
                    "patient_id": patient_id,
                    "image_count": len(image_paths),
                    "reason": f"require_at_least_{required_images}_images",
                }
            )
            continue

        if required_images > 0:
            image_paths = image_paths[:required_images]

        feature_rows: List[Dict[str, float]] = []
        for image_path in image_paths:
            try:
                feats = extract_single_image_features(image_path, image_size=image_size)
                feature_rows.append(feats)
                per_image_rows.append(
                    {
                        "patient_id": patient_id,
                        "image_name": image_path.name,
                        **feats,
                    }
                )
            except Exception as exc:
                skipped_rows.append(
                    {
                        "folder": folder.name,
                        "patient_id": patient_id,
                        "image_name": image_path.name,
                        "reason": f"feature_error: {exc}",
                    }
                )

        if not feature_rows:
            continue

        feature_df = pd.DataFrame(feature_rows)
        agg_row = {"patient_id": patient_id, "n_images": int(len(feature_rows))}
        for col in feature_df.columns:
            agg_row[f"{col}_mean"] = float(feature_df[col].mean())
            agg_row[f"{col}_std"] = float(feature_df[col].std(ddof=0))
        agg_row["Outcome"] = int(label_map[patient_id])
        per_patient_rows.append(agg_row)

    output_dir.mkdir(parents=True, exist_ok=True)
    per_image_csv = output_dir / "image_features_per_image.csv"
    per_patient_csv = output_dir / "image_features_patient.csv"
    skipped_csv = output_dir / "image_processing_skipped.csv"

    pd.DataFrame(per_image_rows).to_csv(per_image_csv, index=False, encoding="utf-8")
    per_patient_df = pd.DataFrame(per_patient_rows)
    if not per_patient_df.empty and "patient_id" in per_patient_df.columns:
        per_patient_df = per_patient_df.sort_values("patient_id")
    per_patient_df.to_csv(per_patient_csv, index=False, encoding="utf-8")
    pd.DataFrame(skipped_rows).to_csv(skipped_csv, index=False, encoding="utf-8")

    return {"per_image_csv": per_image_csv, "per_patient_csv": per_patient_csv, "skipped_csv": skipped_csv}


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract DLFKG image features and combine 5 images per patient.")
    parser.add_argument(
        "--images-root",
        type=Path,
        default=source_root() / "Data" / "Raw data" / "Image" / "images",
        help="Root folder containing patient image subfolders.",
    )
    parser.add_argument(
        "--table-csv",
        type=Path,
        default=source_root() / "Data" / "processing" / "table" / "ICTA_table.csv",
        help="Processed table CSV containing patient_id and Outcome.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=source_root() / "Data" / "processing" / "image",
        help="Output directory for image features.",
    )
    parser.add_argument("--required-images", type=int, default=5, help="Number of images to combine per patient.")
    parser.add_argument(
        "--strict-five-images",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip patient if image count is below required-images.",
    )
    parser.add_argument("--image-size", type=int, default=512, help="Image size for preprocessing.")
    args = parser.parse_args()

    if not args.images_root.exists():
        raise FileNotFoundError(f"Images root not found: {args.images_root}")
    if not args.table_csv.exists():
        raise FileNotFoundError(f"Table CSV not found: {args.table_csv}")

    outputs = process_image_folder(
        images_root=args.images_root,
        table_labels_csv=args.table_csv,
        output_dir=args.out_dir,
        required_images=max(1, int(args.required_images)),
        strict_five_images=bool(args.strict_five_images),
        image_size=max(64, int(args.image_size)),
    )
    print(f"[OK] Per-image features: {outputs['per_image_csv']}")
    print(f"[OK] Per-patient features: {outputs['per_patient_csv']}")
    print(f"[OK] Skipped log: {outputs['skipped_csv']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
