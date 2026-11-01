"""Numerical equivalence to the original ZIP and safeguards against leakage."""
import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from autograd import grad
from imblearn.over_sampling import BorderlineSMOTE

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from compact_fkge import CompactFISA, compress_rules, connected_units
from preprocess import FusionPreprocessor, balance_training, balance_teacher_parts
from train_fkge import CompactFKGE
from run import synthetic_data


def reference_module(name):
    spec = importlib.util.spec_from_file_location(f"_source_{name}", ROOT / "reference" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rulebase(R, y, groups):
    rules, samples = [], []
    for index, row in enumerate(R):
        tokens = [f"a{j:02}-L{value}" for j, value in enumerate(row)]
        rules.append({"id": index, "antecedent_tokens": tokens, "consequent_token": f"class-{y[index]}",
                      "group": int(groups[index])})
        samples.append({"membership": dict.fromkeys(tokens, 1.0), "label": f"class-{y[index]}",
                        "group": int(groups[index])})
    vocab = sorted({t for r in rules for t in r["antecedent_tokens"]}) + ["class-0", "class-1"]
    return SimpleNamespace(rules=rules, vocab=vocab, class_tokens=["class-0", "class-1"]), samples


class EquivalenceTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.RandomState(8)
        base = self.rng.randint(0, 3, (24, 4))
        self.R = np.r_[base, base[:6]]
        self.y = np.r_[np.arange(24) % 2, np.arange(6) % 2]
        self.compact = compress_rules(self.R, self.y, [f"p{i}" for i in range(len(self.R))])
        self.rb, self.samples = rulebase(self.R, self.y, self.compact["original_units"])

    def test_fisa_values_and_attributions_match_original_with_duplicate_rules(self):
        source = reference_module("fisa").FISA(self.rb, aggregator="geomean", node_size=3, eps=1e-9).fit()
        compact = self.compact
        optimized = CompactFISA(compact["R"], compact["y"], compact["weights"])
        query = self.R[:8]
        logs, _ = optimized.log_values(query)
        for index, sample in enumerate(self.samples[:8]):
            D, _, _ = source.decision_values(sample["membership"])
            np.testing.assert_allclose(logs[index], np.log([D["class-0"] + 1e-12, D["class-1"] + 1e-12]), atol=1e-11)
            label = int(np.argmax(logs[index]))
            aggregated = np.zeros(len(compact["R"]))
            for rule, value in source.attribution(sample["membership"]).items():
                aggregated[compact["inverse"][rule]] += value
            actual = optimized.attribution(query[index:index + 1], np.array([label]))[0]
            np.testing.assert_allclose(actual, aggregated, atol=1e-11)

    def test_fkge_full_loss_and_gradient_match_original_and_block_sizes(self):
        c = self.compact
        n = len(c["R"])
        pF = self.rng.uniform(0.1, 1, (n, 2))
        pF /= pF.sum(1, keepdims=True)
        yF = c["y"].copy()
        eF = self.rng.uniform(0.1, 1, (n, n))
        eF *= (c["y"][None, :] == yF[:, None]) & (c["units"][:, None] != c["units"][None, :])
        eF /= eF.sum(1, keepdims=True)
        source = reference_module("fkge").FKGE(self.rb, alpha=0.3, d_e=4, K=5, lam_I=0.3, lam_P=0.5,
                                               lam_E=1, lam_N=1, lam_R=1, lam_C=1e-3, seed=42)
        inv = c["inverse"]
        e_full = eF[inv][:, inv] / c["weights"][inv][None, :]
        Q = source._query_matrix(self.samples)
        objective = lambda params: source.loss(params, Q, self.y, pF[inv], e_full, yF[inv], G=source._mask(self.samples))
        expected_value, expected_gradient = float(objective(source.params)), grad(objective)(source.params)
        for block_size in (2, 11):
            model = CompactFKGE(c["R"], c["y"], c["weights"], c["units"], pF, eF, yF,
                                alpha=0.3, d_e=4, lam_I=0.3, lam_P=0.5, seed=42, device="cpu", block_size=block_size)
            for actual, expected in zip(model.params, source.params):
                np.testing.assert_array_equal(actual.detach().numpy(), expected)
            value, gradient = model.value_grad()
            self.assertAlmostEqual(value, expected_value, places=9)
            for actual, expected in zip(gradient, expected_gradient):
                np.testing.assert_allclose(actual.numpy(), expected, atol=1e-9, rtol=1e-8)


class LeakageTests(unittest.TestCase):
    def test_teacher_partitions_keep_real_patients_and_both_smote_parents_together(self):
        data, _ = synthetic_data(42)
        X = data[["Contrast Feature", "Energy Feature"]].to_numpy()
        original_y = data.diabetic_retinopathy.to_numpy()
        _, y, groups, parents, parts = balance_teacher_parts(X, original_y, data.patient_id, 42)
        self.assertGreater(len(parents), 0)
        for part in range(5):
            self.assertEqual(set(y[parts != part]), {0, 1})
        for offset, (left, right) in enumerate(parents):
            self.assertEqual(parts[len(data) + offset], parts[left])
            self.assertEqual(parts[len(data) + offset], parts[right])
        units = connected_units(groups, np.zeros(len(groups)), parents, merge_signatures=False)
        for unit in np.unique(units):
            self.assertEqual(len(np.unique(parts[units == unit])), 1)

    def test_equal_fuzzy_patterns_do_not_merge_different_real_patients(self):
        units = connected_units(["p1", "p2"], ["same_rule", "same_rule"], merge_signatures=False)
        self.assertNotEqual(units[0], units[1])

    def test_smote_matches_original_and_every_child_is_linked_to_both_parents(self):
        rng = np.random.RandomState(19)
        X = rng.normal(size=(90, 4))
        y = np.r_[np.zeros(70, dtype=int), np.ones(20, dtype=int)]
        patients = [f"p{i}" for i in range(len(y))]
        actual, target, groups, parents = balance_training(X, y, patients, 42)
        expected, expected_y = BorderlineSMOTE(random_state=42, k_neighbors=5, m_neighbors=10).fit_resample(X, y)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(target, expected_y)
        self.assertGreater(len(parents), 0)
        units = connected_units(groups, np.arange(len(groups)), parents)
        for offset, (left, right) in enumerate(parents):
            self.assertEqual(units[len(y) + offset], units[left])
            self.assertEqual(units[len(y) + offset], units[right])

    def test_validation_does_not_change_fitted_medians_or_feature_selection(self):
        data, _ = synthetic_data(42)
        train, val = data.iloc[:150].copy(), data.iloc[150:].copy()
        model = FusionPreprocessor(2, 2).fit(train, train.diabetic_retinopathy)
        medians = [branch[1].copy() for branch in model.branches]
        selected = model.feature_names.copy()
        val.loc[:, "patient_age"] = 1e12
        model.transform(val)
        self.assertEqual(model.feature_names, selected)
        for expected, branch in zip(medians, model.branches):
            np.testing.assert_array_equal(expected, branch[1])


if __name__ == "__main__":
    unittest.main()
