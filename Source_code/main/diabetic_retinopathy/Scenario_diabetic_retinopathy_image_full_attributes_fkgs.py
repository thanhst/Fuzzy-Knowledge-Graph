import argparse
import os
import random
import sys
from datetime import datetime

import numpy as np
import pandas as pd


current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))

if project_root not in sys.path:
    sys.path.append(project_root)

from module.FKG.FKG_S import FKGS


SOURCE_MODALITY = "Diabetic Retinopathy Image Feature"
RUN_MODALITY = "Diabetic Retinopathy Image Feature Full Attributes 2026-07-11"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run image-only diabetic retinopathy FKGS with all image attributes."
    )
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--ran", type=int, nargs="+", default=[20])
    parser.add_argument("--e", type=float, nargs="+", default=[0.2, 0.3])
    return parser.parse_args()


def load_fis_rules():
    train_path = os.path.join(
        project_root,
        "data",
        "FIS",
        "output",
        SOURCE_MODALITY,
        "FRB",
        "TrainDataRule.csv",
    )
    test_path = os.path.join(
        project_root,
        "data",
        "FIS",
        "output",
        SOURCE_MODALITY,
        "FRB",
        "TestDataRule.csv",
    )
    traindf = pd.read_csv(train_path)
    testdf = pd.read_csv(test_path)
    base = [[int(float(x)) for x in row] for row in traindf.values]
    test = [[int(float(x)) for x in row] for row in testdf.values]
    return pd.DataFrame(base), test, traindf.shape, testdf.shape, train_path, test_path


def print_run_metadata():
    data_path = os.path.join(
        project_root,
        "data",
        "Dataset_diabetic",
        "Image_feature",
        "data_process.csv",
    )
    raw_df = pd.read_csv(data_path)

    print(RUN_MODALITY)
    print("=" * 100)
    print("Run date: 2026-07-11")
    print("Run timestamp:", datetime.now().isoformat(timespec="seconds"))
    print("Dataset: Diabetic Retinopathy")
    print("Pipeline: Image Feature Full Attributes")
    print("Feature selection: NO")
    print("Fusion: NO")
    print("Source data:", data_path)
    print("Source data shape:", raw_df.shape)
    print("Source label counts:", raw_df.iloc[:, -1].value_counts(dropna=False).sort_index().to_dict())
    print("Feature count:", raw_df.shape[1] - 1)
    print("=" * 100)


def main():
    args = parse_args()
    print_run_metadata()

    for ran in args.ran:
        for e_value in args.e:
            seed = args.seed + int(ran * 100) + int(e_value * 1000)
            random.seed(seed)
            np.random.seed(seed)
            print("__________Running FKG-S___________")
            print(f"Run configuration: ran={ran}, e={e_value}, seed={seed}")
            base, test, train_shape, test_shape, train_path, test_path = load_fis_rules()
            print("FRB train:", train_path)
            print("FRB test:", test_path)
            print("TrainDataRule shape:", train_shape)
            print("TestDataRule shape:", test_shape)
            fkg_instance = FKGS()
            fkg_instance.FKGS(
                df=base,
                testdf=test,
                Turn=None,
                Modality=RUN_MODALITY,
                ran=ran,
                e=e_value,
                folderPath=project_root,
            )
            print("-" * 100)


if __name__ == "__main__":
    main()
