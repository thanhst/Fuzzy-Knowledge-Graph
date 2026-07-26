import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[1]

if str(PROJECT_ROOT) not in os.sys.path:
    os.sys.path.append(str(PROJECT_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description="Run one FKGS config in an isolated process.")
    parser.add_argument("--train-rule", required=True)
    parser.add_argument("--test-rule", required=True)
    parser.add_argument("--modality", required=True)
    parser.add_argument("--ran", type=int, required=True)
    parser.add_argument("--epsilon", type=float, required=True)
    parser.add_argument("--turns", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--summary-path", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.turns < 1:
        raise ValueError("--turns must be at least 1.")

    os.environ.setdefault("MPLBACKEND", "Agg")
    os.chdir(PROJECT_ROOT)
    random.seed(args.seed)
    np.random.seed(args.seed)

    from module.FKG.FKG_S import FKGS

    legacy_module = os.sys.modules.get("fisa_module")
    if legacy_module is None or not hasattr(legacy_module, "calculateA"):
        module_path = "" if legacy_module is None else str(getattr(legacy_module, "__file__", ""))
        raise ImportError(f"FKGS requires legacy fisa_module.calculateA, got: {module_path}")

    train_df = pd.read_csv(args.train_rule)
    test_df = pd.read_csv(args.test_rule)
    base = pd.DataFrame([[int(float(x)) for x in row] for row in train_df.values])
    test = [[int(float(x)) for x in row] for row in test_df.values]

    fkg_instance = FKGS()
    summary = fkg_instance.FKGS(
        df=base,
        testdf=test,
        Turn=args.turns,
        Modality=args.modality,
        ran=args.ran,
        e=args.epsilon,
        folderPath=str(PROJECT_ROOT),
    )
    summary = dict(summary or {})
    summary["seed"] = args.seed
    summary["fkgs_module_path"] = str(getattr(legacy_module, "__file__", ""))
    summary["bar_scores_png"] = str(
        PROJECT_ROOT / "data" / "FKG" / args.modality / f"bar_scores_e{args.epsilon}_ran{args.ran}.png"
    )

    summary_path = Path(args.summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[FKGS-SUMMARY] {summary_path}")


if __name__ == "__main__":
    main()
