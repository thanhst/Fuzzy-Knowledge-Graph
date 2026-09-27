import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import config as C
from data.frb_package import load_frb_folds
from data.fkg_io import generate_synthetic_prefuzzified_records, token_attribute
from data.pipeline_interface import PrefuzzifiedRulePipeline
from models.fisa import FISA
from models.metrics import classification_metrics


class ProtocolTests(unittest.TestCase):
    def test_continuous_score_auc(self):
        metrics = classification_metrics(
            ["class-0", "class-1", "class-0", "class-1"],
            ["class-0", "class-1", "class-0", "class-1"],
            [0.1, 0.9, 0.2, 0.8],
            ["class-0", "class-1"],
        )
        self.assertEqual(metrics["auc_roc"], 1.0)
        self.assertEqual(metrics["auc_pr"], 1.0)

    def test_pipeline_builds_node_edges(self):
        records = generate_synthetic_prefuzzified_records(n_patients=20, seed=7)
        fkg, _ = PrefuzzifiedRulePipeline().fit_and_mine(records)
        self.assertGreater(len(fkg.edges), 0)

    def test_fisa_modes_are_equivalent(self):
        records = generate_synthetic_prefuzzified_records(n_patients=20, seed=8)
        fkg, samples = PrefuzzifiedRulePipeline().fit_and_mine(records)
        sequential = FISA(fkg, "sequential").fit().evaluate(samples)
        lookup = FISA(fkg, "lookup").fit().evaluate(samples)
        self.assertEqual(sequential["y_pred"], lookup["y_pred"])
        self.assertAlmostEqual(sequential["auc_roc"], lookup["auc_roc"])

    def test_brset_package_has_verified_patient_folds(self):
        folds = load_frb_folds(
            C.PATHS.BRSET_FRB_PACKAGE, modality=C.BRSET_PRIMARY_MODALITY)
        self.assertEqual(len(folds), 5)
        self.assertTrue(all(
            fold["metadata"]["patient_overlap_count"] == 0 for fold in folds
        ))
        self.assertTrue(all(
            fold["metadata"]["manifest_image_id_match"] for fold in folds
        ))
        self.assertTrue(all(
            fold["metadata"]["input_graph_kind"] == "FKG-MM" for fold in folds
        ))
        self.assertTrue(all(
            set(fold["metadata"]["modalities"]) == {"image", "table"}
            for fold in folds
        ))

    def test_fusion_pipeline_builds_cross_modal_edges(self):
        fold = load_frb_folds(
            C.PATHS.BRSET_FRB_PACKAGE, modality=C.BRSET_PRIMARY_MODALITY)[0]
        fkg, _ = PrefuzzifiedRulePipeline().fit_and_mine(fold["train_records"])
        self.assertEqual(fkg.meta["input_graph_kind"], "FKG-MM")
        self.assertGreater(fkg.meta["intra_modal_edge_count"], 0)
        self.assertGreater(fkg.meta["cross_modal_edge_count"], 0)

    def test_real_frb_token_attribute_parsing(self):
        self.assertEqual(
            token_attribute("image::Dissimilarity Feature=L3"),
            "image::Dissimilarity Feature",
        )


if __name__ == "__main__":
    unittest.main()
