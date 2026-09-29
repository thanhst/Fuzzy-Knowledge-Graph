import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import config as C
from data.frb_package import load_frb_folds
from data.fkg_io import generate_synthetic_prefuzzified_records, token_attribute
from data.pipeline_interface import PrefuzzifiedRulePipeline
from experiments.common import compact_fold_observations, fold_bootstrap_summary
from models.fisa import FISA
from models.metrics import classification_metrics, fidelity_metrics
from models.fkge import FKGE, _gradient_check


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

    def test_fold_bootstrap_averages_seeds_before_resampling(self):
        rows = [
            {"fold": 1, "seed": 42, "auc_roc": 0.4},
            {"fold": 1, "seed": 43, "auc_roc": 0.6},
            {"fold": 2, "seed": 42, "auc_roc": 0.8},
            {"fold": 2, "seed": 43, "auc_roc": 1.0},
        ]
        summary = fold_bootstrap_summary(rows, ("auc_roc",))
        repeated = fold_bootstrap_summary(rows * 3, ("auc_roc",))
        self.assertEqual(summary, repeated)
        self.assertAlmostEqual(summary["auc_roc_ci95_low"], 0.5)
        self.assertAlmostEqual(summary["auc_roc_ci95_high"], 0.9)
        self.assertEqual(len(compact_fold_observations(rows, ("auc_roc",))), 4)

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

    def test_kb2_subset_reserves_both_rule_classes(self):
        fold = load_frb_folds(
            C.PATHS.BRSET_FRB_PACKAGE, modality=C.BRSET_PRIMARY_MODALITY)[0]
        fkg, _ = PrefuzzifiedRulePipeline().fit_and_mine(fold["train_records"])
        sampled = fkg.sample_subset(ratio=0.3, seed=7, min_class_fraction=0.4)
        import math
        expected_size = math.ceil(len(fkg.rules) * 0.3)
        self.assertEqual(len(sampled.rules), expected_size)
        counts = sampled.meta["subset_sampling"]["rule_class_counts"]
        self.assertTrue(all(count >= math.ceil(expected_size * 0.4)
                            for count in counts.values()))

    def test_real_frb_token_attribute_parsing(self):
        self.assertEqual(
            token_attribute("image::Dissimilarity Feature=L3"),
            "image::Dissimilarity Feature",
        )

    def test_full_rule_cooccurrence_and_class_max(self):
        records = generate_synthetic_prefuzzified_records(n_patients=20, seed=8)
        fkg, samples = PrefuzzifiedRulePipeline().fit_and_mine(records)
        model = FKGE(fkg, d=4, w=None, K_neg=1, seed=9)
        self.assertEqual(len(model.sg_pairs), sum(
            len(ids) * (len(ids) - 1) for ids in model.rule_token_ids))
        rule_emb = model.rule_embeddings()
        probability, _, scores, _, _ = model._forward_predict(
            samples[0]["membership"], rule_emb)
        class_maxima = [max(scores[model.rule_class_indices == index])
                        for index in range(model.n_classes)]
        import numpy as np
        expected = np.exp(class_maxima - np.max(class_maxima))
        expected /= expected.sum()
        np.testing.assert_allclose(probability, expected)

    def test_linear_rule_pooling_matches_per_rule_reference(self):
        import numpy as np

        records = generate_synthetic_prefuzzified_records(n_patients=20, seed=8)
        fkg, _ = PrefuzzifiedRulePipeline().fit_and_mine(records)
        for pooling in ("mean", "weighted"):
            model = FKGE(fkg, d=8, pooling=pooling, seed=19)
            expected = np.stack([
                model._pool(ids, model.Es) for ids in model.rule_token_ids
            ])
            np.testing.assert_allclose(model.rule_embeddings(), expected,
                                       rtol=0, atol=1e-15)

    def test_fidelity_reports_teacher_balance_and_bound(self):
        reference = [{"class-0": .9, "class-1": .1},
                     {"class-0": .2, "class-1": .8}]
        candidate = [{"class-0": .8, "class-1": .2},
                     {"class-0": .1, "class-1": .9}]
        result = fidelity_metrics(reference, candidate, ["class-0", "class-1"])
        self.assertEqual(result["agreement"], 1.0)
        self.assertEqual(result["fidelity_balanced_accuracy"], 1.0)
        self.assertEqual(result["cohen_kappa"], 1.0)
        self.assertEqual(result["fidelity_bound_violations"], 0)

    def test_class_max_gradient(self):
        _gradient_check()


if __name__ == "__main__":
    unittest.main()
