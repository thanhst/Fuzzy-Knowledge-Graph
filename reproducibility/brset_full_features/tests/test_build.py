"""Protect ID alignment and missing-value semantics in the full-data export."""
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("_brset_full_feature_builder_test", ROOT / "build.py")
builder = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = builder
spec.loader.exec_module(builder)


class PrepareTests(unittest.TestCase):
    def setUp(self):
        private = ROOT / "private"
        private.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="unit_", dir=private)
        self.directory = Path(self.temporary.name).resolve()
        self.assertTrue(self.directory.is_relative_to(private.resolve()))
        self.images = self.directory / "images.csv"
        self.labels = self.directory / "labels.csv"
        # Image features deliberately arrive in the opposite order.
        images = pd.DataFrame({"image_id": ["img_b", "img_a"],
                               **{c: [0.9, 0.1] for c in builder.IMAGE_COLUMNS}})
        labels = pd.DataFrame({"image_id": ["img_a", "img_b"], "patient_id": ["p_a", "p_b"],
                               builder.TARGET: [0, 1], **{c: ["1", "1"] for c in builder.TABLE_COLUMNS}})
        labels["patient_age"] = ["NA", "40"]
        labels["insuline"] = ["yes", "NA"]
        labels["diabetes"] = ["No", "yes"]
        labels["quality"] = ["Adequate", "Inadequate"]
        images.to_csv(self.images, index=False)
        labels.to_csv(self.labels, index=False)

    def tearDown(self):
        self.temporary.cleanup()

    def test_join_uses_image_id_and_preserves_original_patient_label_pair(self):
        result, audit, _ = builder.prepare(self.images, self.labels)
        self.assertEqual(result.image_id.tolist(), ["img_a", "img_b"])
        self.assertEqual(result.patient_id.tolist(), ["p_a", "p_b"])
        self.assertEqual(result[builder.TARGET].tolist(), [0, 1])
        self.assertEqual(result["Contrast Feature"].tolist(), [0.1, 0.9])
        self.assertEqual(audit["retained_images"], 2)

    def test_missing_clinical_values_are_not_globally_imputed(self):
        result, _, _ = builder.prepare(self.images, self.labels)
        self.assertTrue(pd.isna(result.loc[0, "patient_age"]))
        self.assertEqual(result.loc[1, "patient_age"], 40)
        self.assertEqual(result.loc[0, "insuline"], 2)
        self.assertTrue(pd.isna(result.loc[1, "insuline"]))
        self.assertEqual(result.quality.tolist(), [2, 1])

    def test_duplicate_or_incomplete_image_coverage_is_rejected(self):
        images = pd.read_csv(self.images)
        pd.concat([images, images.iloc[:1]]).to_csv(self.images, index=False)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            builder.prepare(self.images, self.labels)
        images.iloc[:1].to_csv(self.images, index=False)
        with self.assertRaisesRegex(ValueError, "complete retained label population"):
            builder.prepare(self.images, self.labels)


if __name__ == "__main__":
    unittest.main()
