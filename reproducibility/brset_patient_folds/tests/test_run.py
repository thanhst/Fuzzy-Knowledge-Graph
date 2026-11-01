import importlib.util
import tempfile
import unittest
from pathlib import Path

import pandas as pd

RUNNER = Path(__file__).resolve().parents[1] / "run.py"
SPEC = importlib.util.spec_from_file_location("brset_fold_release_runner", RUNNER)
RUN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUN)


def synthetic_records():
    # Mixed labels within patients verify that labels remain image-specific.
    return pd.DataFrame([
        {"image_id": f"synthetic_{patient:02}_{eye}", "patient_id": f"patient_{patient:02}",
         "label": (patient + eye) % 2}
        for patient in range(20) for eye in range(1 + patient % 3)
    ])


class PatientFoldReleaseTests(unittest.TestCase):
    def test_both_methods_keep_patient_and_image_invariants(self):
        clean, _ = RUN.clean_records(synthetic_records(), "label")
        for method in RUN.METHODS:
            with self.subTest(method=method):
                assigned, actual = RUN.create_assignments(clean, "label", method, 5, 42)
                repeat, _ = RUN.create_assignments(clean, "label", method, 5, 42)
                self.assertTrue(assigned["fold"].equals(repeat["fold"]))
                self.assertTrue(assigned[["image_id", "patient_id", "label"]].equals(clean))
                checks = RUN.verify_assignments(assigned, "label", 5)
                self.assertEqual(checks["patient_overlap_count"], 0)
                self.assertEqual(sum(row["test_images"] for row in checks["folds"]), len(clean))
                if method == "stratified-group":
                    self.assertEqual(actual, "StratifiedGroupKFold")

    def test_dirty_records_are_removed_without_picking_a_conflicting_duplicate(self):
        dirty = pd.DataFrame([
            {"image_id": " keep ", "patient_id": " p1 ", "label": "1"},
            {"image_id": "duplicate", "patient_id": "p2", "label": 0},
            {"image_id": "duplicate", "patient_id": "p3", "label": 1},
            {"image_id": "missing", "patient_id": " ", "label": 0},
            {"image_id": "bad_label", "patient_id": "p4", "label": "bad"},
            {"image_id": "out_of_range", "patient_id": "p5", "label": 2},
        ])
        clean, audit = RUN.clean_records(dirty, "label")
        self.assertEqual(clean.to_dict("records"), [{"image_id": "keep", "patient_id": "p1", "label": 1}])
        self.assertEqual(audit["removed_missing_ids"], 1)
        self.assertEqual(audit["removed_duplicate_image_ids"], 2)
        self.assertEqual(audit["removed_invalid_binary_labels"], 2)

    def test_reference_comparison_detects_a_changed_partition(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "synthetic.csv"
            synthetic_records().to_csv(path, index=False)
            first = RUN.run(path, root / "first", "label", "custom", 5, 42)
            matched = RUN.run(path, root / "matched", "label", "custom", 5, 42, root / "first")
            changed = RUN.run(path, root / "changed", "label", "stratified-group", 5, 42, root / "first")
            self.assertTrue(first["checks"]["each_image_held_out_once"])
            self.assertTrue(matched["reference_comparison"]["all_test_image_sets_match"])
            self.assertFalse(changed["reference_comparison"]["all_test_image_sets_match"])

    def test_existing_output_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            marker = root / "keep.txt"
            marker.write_text("preserve", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "new or empty"):
                RUN.run(root / "not_read.csv", root, "label", "custom", 5, 42)
            self.assertEqual(marker.read_text(encoding="utf-8"), "preserve")


if __name__ == "__main__":
    unittest.main()
