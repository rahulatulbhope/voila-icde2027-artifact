import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import percom_experiments as percom
import run_all_experiments as experiments


class RoutingPolicyTests(unittest.TestCase):
    def setUp(self):
        self.specs = [
            experiments.FidelitySpec("caption", "caption", "answer_caption", "prob_caption", "lpd_caption"),
            experiments.FidelitySpec("full", "full", "answer_full", "prob_full", "lpd_full"),
        ]
        self.costs = {"caption": 1.0, "full": 10.0}

    def test_exact_argmax_uses_global_utility(self):
        predictions = pd.DataFrame({"p_caption": [0.60], "p_full": [0.95]})
        selected = experiments.select_voi(predictions, self.specs, self.costs, lam=0.02)
        self.assertEqual(selected.tolist(), ["full"])

    def test_tie_breaks_toward_lower_cost(self):
        predictions = pd.DataFrame({"p_caption": [0.50], "p_full": [0.68]})
        selected = experiments.select_voi(predictions, self.specs, self.costs, lam=0.02)
        self.assertEqual(selected.tolist(), ["caption"])

    def test_missing_fidelity_is_allowed_but_missing_row_is_not(self):
        available = pd.DataFrame({"p_caption": [0.7]})
        self.assertEqual(
            experiments.select_voi(available, self.specs, self.costs, lam=0.0).tolist(),
            ["caption"],
        )
        missing_row = pd.DataFrame({"p_caption": [np.nan], "p_full": [np.nan]})
        with self.assertRaisesRegex(ValueError, "no finite fidelity prediction"):
            experiments.select_voi(missing_row, self.specs, self.costs, lam=0.0)


class GroupSplitTests(unittest.TestCase):
    def setUp(self):
        self.cfg = {"qid": "question_id", "question": "question"}

    def test_primary_floodnet_folds_have_no_image_overlap(self):
        frame = pd.DataFrame(
            {
                "question_id": range(12),
                "image_id": np.repeat(["a", "b", "c", "d"], 3),
                "question": ["How many buildings?"] * 12,
            }
        )
        prepared, group_col, strategy = experiments.prepare_split_groups(
            frame, self.cfg, dataset_name="floodnet", split_strategy="auto"
        )
        self.assertEqual(strategy, "image_grouped")
        for train_groups, test_groups in experiments.make_outer_splits(prepared, group_col, 4, 42):
            self.assertFalse(set(train_groups) & set(test_groups))

    def test_stress_folds_have_no_template_overlap(self):
        frame = pd.DataFrame(
            {
                "question_id": range(8),
                "image_id": range(8),
                "question": ["Road blocked?", "ROAD blocked?", "How many cars?", "HOW many cars?",
                             "Any flood?", "ANY flood?", "Bridge safe?", "BRIDGE safe?"],
            }
        )
        prepared, group_col, strategy = experiments.prepare_split_groups(
            frame, self.cfg, dataset_name="floodnet", split_strategy="question_grouped"
        )
        self.assertEqual(strategy, "question_grouped")
        for train_groups, test_groups in experiments.make_outer_splits(prepared, group_col, 2, 42):
            self.assertFalse(set(train_groups) & set(test_groups))


class MissionBudgetTests(unittest.TestCase):
    def test_infeasible_budget_raises(self):
        values = np.array([[0.2, 0.9], [0.3, 0.8]])
        costs = np.array([[2, 4], [2, 4]])
        with self.assertRaisesRegex(ValueError, "infeasible"):
            percom.mckp_dynamic_program(values, costs, budget=3, bucket_bytes=1)

    def test_dynamic_program_matches_bruteforce_and_respects_budget(self):
        values = np.array([[0.2, 0.9, 0.7], [0.3, 0.8, 0.6], [0.4, 0.5, 0.95]])
        costs = np.array([[1, 5, 3], [1, 4, 2], [1, 3, 6]])
        budget = 10
        dynamic = percom.mckp_dynamic_program(values, costs, budget, bucket_bytes=1)
        brute = percom.mckp_bruteforce(values, costs, budget)
        self.assertAlmostEqual(dynamic["value"], brute["value"])
        self.assertLessEqual(dynamic["actual_cost"], budget)

    def test_bucket_rounding_never_violates_requested_budget(self):
        values = np.array([[0.1, 0.9], [0.2, 0.8]])
        costs = np.array([[100, 1025], [100, 1025]])
        result = percom.mckp_dynamic_program(values, costs, budget=2050, bucket_bytes=1024)
        self.assertLessEqual(result["actual_cost"], 2050)


class PopulationTests(unittest.TestCase):
    def test_cross_model_intersection_uses_image_question_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            left = pd.DataFrame(
                {"image_id": ["a", "b", "c"], "question": ["q1", "q2", "q3"]}
            )
            right = pd.DataFrame(
                {"image_id": ["b", "c", "d"], "question": ["q2", "q3", "q4"]}
            )
            left_path, right_path = directory / "left.csv", directory / "right.csv"
            left.to_csv(left_path, index=False)
            right.to_csv(right_path, index=False)
            common, metadata = percom.build_common_intersection(
                [left_path, right_path], directory / "common.csv"
            )
            self.assertEqual(metadata["common_sample_count"], 2)
            self.assertEqual(set(map(tuple, common.to_numpy())), {("b", "q2"), ("c", "q3")})


if __name__ == "__main__":
    unittest.main()
