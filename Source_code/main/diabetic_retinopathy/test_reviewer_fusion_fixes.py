import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd


MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

import Preprocess_kfold_feature_selection as pipeline


class ReviewerFusionFixTests(unittest.TestCase):
    def test_cross_modal_svd_projections_have_compatible_dimensions(self):
        rng = np.random.default_rng(42)
        image = rng.normal(size=(12, 4))
        table = rng.normal(size=(12, 3))

        w_img, w_tab, singular_values = pipeline.fit_cross_modal_svd(image, table, 2)
        image_latent, table_latent = pipeline.apply_cross_modal_projection(
            image,
            table,
            w_img,
            w_tab,
            singular_values,
        )

        self.assertEqual(w_img.shape, (4, 2))
        self.assertEqual(w_tab.shape, (3, 2))
        self.assertEqual(image_latent.shape, (12, 2))
        self.assertEqual(table_latent.shape, (12, 2))
        self.assertEqual((image_latent * table_latent).shape, (12, 2))

    def test_filter_rejects_cross_modal_duplicate(self):
        image = np.array([[0.0], [1.0], [2.0], [3.0]])
        table = image.copy()

        selected_img, selected_tab = pipeline.select_filter_multimodal_indices(
            image,
            table,
            np.array([1.0]),
            np.array([0.9]),
            k_img=1,
            k_tab=1,
            threshold=0.95,
        )

        self.assertEqual(selected_img, [0])
        self.assertEqual(selected_tab, [])

    def test_wrapper_stops_when_addition_does_not_improve_baseline(self):
        image = np.arange(16, dtype=float).reshape(8, 2)
        table = np.arange(16, 32, dtype=float).reshape(8, 2)
        target = np.array([0, 1] * 4)
        args = type(
            "Args",
            (),
            {
                "wrapper_cv": 2,
                "wrapper_rf_estimators": 1,
                "wrapper_min_img": 1,
                "wrapper_max_img": 2,
                "wrapper_min_tab": 1,
                "wrapper_max_tab": 2,
            },
        )()

        def score_by_width(values, *_args, **_kwargs):
            return {1: 0.6, 2: 0.9}.get(values.shape[1], 0.8)

        with patch.object(pipeline, "evaluate_wrapper_feature_set", side_effect=score_by_width):
            selected_img, selected_tab = pipeline.select_wrapper_indices(
                image,
                table,
                target,
                args,
                seed=42,
            )

        self.assertEqual(selected_img, [0])
        self.assertEqual(selected_tab, [0])

    def test_scenario_tables_include_all_requested_metrics(self):
        row = {
            "modality": "table",
            "ran": 15,
            "epsilon": 0.2,
            "folds": 5,
            "feature_count_mean": 13,
            "fkgs_accuracy_pct_mean": 90.0,
            "fkgs_accuracy_pct_std": 1.0,
            "fkgs_full_train_time_seconds_mean": 2.0,
            "fkgs_full_train_time_seconds_std": 0.1,
            "fkgs_test_time_seconds_mean": 1.0,
            "fkgs_test_time_seconds_std": 0.1,
            "fkgs_total_time_seconds_mean": 3.0,
            "fkgs_total_time_seconds_std": 0.2,
            "fkgs_end_to_end_time_seconds_mean": 3.5,
            "fkgs_end_to_end_time_seconds_std": 0.3,
        }
        for metric, mean, std in (
            ("sensitivity", 0.7, 0.02),
            ("specificity", 0.8, 0.03),
            ("f1", 0.6, 0.04),
            ("auc_roc", 0.9, 0.01),
            ("auc_pr", 0.5, 0.05),
        ):
            row[f"fkgs_{metric}_mean"] = mean
            row[f"fkgs_{metric}_std"] = std

        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "tables.csv"
            markdown_path = Path(temp_dir) / "tables.md"
            pipeline.write_fkgs_tables(pd.DataFrame([row]), csv_path, markdown_path)
            table = pd.read_csv(csv_path)
            markdown = markdown_path.read_text(encoding="utf-8")

        for column in (
            "sensitivity_pct",
            "specificity_pct",
            "f1_pct",
            "auc_roc_pct",
            "auc_pr_pct",
        ):
            self.assertIn(column, table.columns)
        self.assertAlmostEqual(table.loc[0, "sensitivity_pct"], 70.0)
        self.assertIn("Sensitivity (%)", markdown)
        self.assertIn("AUC-PR (%)", markdown)


if __name__ == "__main__":
    unittest.main()
