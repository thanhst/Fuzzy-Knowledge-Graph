"""Create the compact 224x224 image set shipped with this branch."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, help="Directory containing the original BRSET JPG files")
    parser.add_argument("--manifest", type=Path, default=Path("data/train_test_selection/all_images.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/fundus_photos_224"))
    parser.add_argument("--quality", type=int, default=90)
    args = parser.parse_args()

    frame = pd.read_csv(args.manifest)
    image_ids = sorted(frame["image_id"].astype(str).unique())
    missing = [image_id for image_id in image_ids if not (args.source / f"{image_id}.jpg").is_file()]
    if missing:
        raise SystemExit(
            f"Source directory is missing {len(missing)} required BRSET images; "
            f"examples: {', '.join(missing[:5])}"
        )
    args.output.mkdir(parents=True, exist_ok=True)
    for index, image_id in enumerate(image_ids, start=1):
        source = args.source / f"{image_id}.jpg"
        target = args.output / source.name
        with Image.open(source) as image:
            image.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR).save(
                target, "JPEG", quality=args.quality, optimize=True
            )
        if index % 250 == 0 or index == len(image_ids):
            print(f"[{index}/{len(image_ids)}] {target}")


if __name__ == "__main__":
    main()
