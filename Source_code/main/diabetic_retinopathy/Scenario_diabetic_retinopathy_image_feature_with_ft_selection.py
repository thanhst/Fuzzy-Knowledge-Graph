import argparse
import json
import os
import random
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif


current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))

if project_root not in sys.path:
    sys.path.append(project_root)

from module.FIS.FIS import FIS
from module.FKG.FKG_S import FKGS


MODALITY = "Diabetic Retinopathy Image Feature FT Selection"
SOURCE_RELATIVE_PATH = "data/Dataset_diabetic/Image_feature/data_process.csv"
OUTPUT_RELATIVE_DIR = "data/Dataset_diabetic/Image_feature_FT_selection"

IMAGE_FEATURE_NAMES = {
    "0": "Contrast Feature",
    "1": "Dissimilarity Feature",
    "2": "Homogeneity Feature",
    "3": "Energy Feature",
    "4": "Correlation Feature",
    "5": "ASM Feature",
    "6": "Mean Feature",
    "7": "Variance Feature",
    "8": "Standard Deviation Feature",
    "9": "RMS Feature",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run image-only diabetic retinopathy feature selection without table fusion."
    )
    parser.add_argument("--k-img", type=int, default=7)
    parser.add_argument("--seed", type=int, default=20260709)
    parser.add_argument("--ran", type=int, nargs="+", default=[20])
    parser.add_argument("--e", type=float, nargs="+", default=[0.2, 0.3])
    parser.add_argument("--skip-fis", action="store_true")
    return parser.parse_args()


def build_image_only_dataset(k_img, seed):
    source_path = os.path.join(project_root, SOURCE_RELATIVE_PATH)
    output_dir = os.path.join(project_root, OUTPUT_RELATIVE_DIR)
    os.makedirs(output_dir, exist_ok=True)

    run_date = datetime.now().strftime("%Y-%m-%d")
    df = pd.read_csv(source_path)
    x = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    if k_img < 1 or k_img > x.shape[1]:
        raise ValueError(f"k_img must be in [1, {x.shape[1]}], got {k_img}")

    selector = SelectKBest(score_func=f_classif, k=k_img)
    x_selected = selector.fit_transform(x, y)
    selected_source_columns = list(x.columns[selector.get_support()])
    selected_output_columns = [str(i) for i in range(k_img)]

    selected_df = pd.DataFrame(x_selected, columns=selected_output_columns)
    selected_df["diabetic_retinopathy"] = y.reset_index(drop=True)

    dated_data_path = os.path.join(output_dir, f"data_process_{run_date}.csv")
    active_data_path = os.path.join(output_dir, "data_process.csv")
    selected_df.to_csv(dated_data_path, index=False)
    selected_df.to_csv(active_data_path, index=False)

    feature_rows = []
    selected_set = set(selected_source_columns)
    for source_column, score, p_value in zip(x.columns, selector.scores_, selector.pvalues_):
        output_column = ""
        if source_column in selected_set:
            output_column = str(selected_source_columns.index(source_column))
        feature_rows.append(
            {
                "run_date": run_date,
                "selected": source_column in selected_set,
                "source_column": source_column,
                "selected_output_column": output_column,
                "feature_name": IMAGE_FEATURE_NAMES.get(str(source_column), str(source_column)),
                "f_score": score,
                "p_value": p_value,
            }
        )

    selected_features_path = os.path.join(output_dir, f"selected_features_{run_date}.csv")
    pd.DataFrame(feature_rows).to_csv(selected_features_path, index=False)

    metadata = {
        "run_date": run_date,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "seed": seed,
        "pipeline": "image_only_feature_selection",
        "is_fusion": False,
        "source_data": source_path,
        "active_data_path": active_data_path,
        "dated_data_path": dated_data_path,
        "selected_features_path": selected_features_path,
        "k_img": k_img,
        "selected_source_columns": selected_source_columns,
        "selected_feature_names": [
            IMAGE_FEATURE_NAMES.get(str(column), str(column)) for column in selected_source_columns
        ],
        "label_counts": {str(k): int(v) for k, v in y.value_counts().sort_index().items()},
        "shape": list(selected_df.shape),
    }
    metadata_path = os.path.join(output_dir, f"run_metadata_{run_date}.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("=" * 100)
    print(f"Run date: {run_date}")
    print(f"Run timestamp: {metadata['created_at']}")
    print("Dataset: Diabetic Retinopathy")
    print("Pipeline: Image Feature FT Selection")
    print("Fusion: NO")
    print(f"Source data: {source_path}")
    print(f"Active selected data: {active_data_path}")
    print(f"Dated selected data: {dated_data_path}")
    print(f"Selected feature metadata: {selected_features_path}")
    print(f"Run metadata: {metadata_path}")
    print(f"Selected features: {metadata['selected_feature_names']}")
    print(f"Output shape: {selected_df.shape}")
    print("=" * 100)

    return active_data_path


def load_fis_rules():
    train_path = os.path.join(
        project_root,
        "data/FIS/output",
        MODALITY,
        "FRB/TrainDataRule.csv",
    )
    test_path = os.path.join(
        project_root,
        "data/FIS/output",
        MODALITY,
        "FRB/TestDataRule.csv",
    )
    traindf = pd.read_csv(train_path)
    testdf = pd.read_csv(test_path)
    base = [[int(float(x)) for x in row] for row in traindf.values]
    test = [[int(float(x)) for x in row] for row in testdf.values]
    return pd.DataFrame(base), test


def main():
    args = parse_args()
    np.random.seed(args.seed)
    random.seed(args.seed)

    print(MODALITY)
    active_data_path = build_image_only_dataset(k_img=args.k_img, seed=args.seed)

    if not args.skip_fis:
        print("__________Running FIS___________")
        FIS(
            fileName=MODALITY,
            filePath=active_data_path,
            cluster=[5] * args.k_img + [2],
        )
        print("--------------------------------")

    for ran in args.ran:
        for e_value in args.e:
            combo_seed = args.seed + int(ran * 100) + int(e_value * 1000)
            random.seed(combo_seed)
            np.random.seed(combo_seed)
            print("__________Running FKG-S___________")
            print(f"Run configuration: ran={ran}, e={e_value}, seed={combo_seed}")
            base, test = load_fis_rules()
            fkg_instance = FKGS()
            fkg_instance.FKGS(
                df=base,
                testdf=test,
                Turn=None,
                Modality=MODALITY,
                ran=ran,
                e=e_value,
                folderPath=project_root,
            )
            print("-" * 100)


if __name__ == "__main__":
    main()
