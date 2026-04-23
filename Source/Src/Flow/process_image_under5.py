#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from process_image import process_image_folder, source_root


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Image processing pipeline that allows patients with fewer than 5 images."
    )
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
        default=source_root() / "Data" / "processing" / "image_under5",
        help="Output folder for under-5 image pipeline.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=5,
        help="Maximum number of images used per patient (take first N sorted images).",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=512,
        help="Image size for preprocessing.",
    )
    args = parser.parse_args()

    outputs = process_image_folder(
        images_root=args.images_root,
        table_labels_csv=args.table_csv,
        output_dir=args.out_dir,
        required_images=max(1, int(args.max_images)),
        strict_five_images=False,
        image_size=max(64, int(args.image_size)),
    )

    patient_df = pd.read_csv(outputs["per_patient_csv"], dtype={"patient_id": str})
    per_image_df = pd.read_csv(outputs["per_image_csv"], dtype={"patient_id": str})

    print(f"[OK] Under-5 pipeline output: {args.out_dir}")
    print(f"[OK] Patients with image features: {len(patient_df)}")
    print(f"[OK] Total processed images: {len(per_image_df)}")
    print(f"[OK] Per-patient CSV: {outputs['per_patient_csv']}")
    print(f"[OK] Per-image CSV: {outputs['per_image_csv']}")
    print(f"[OK] Skipped log: {outputs['skipped_csv']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
