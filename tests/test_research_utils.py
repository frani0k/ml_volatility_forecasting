import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from audit_experiment import audit_data
from bootstrap_uncertainty import bootstrap_block_ci
from research_utils import (
    BASELINE,
    FEATURES,
    TARGET,
    build_feature_dataset,
    calculate_daily_returns,
    chronological_split,
    clip_nonnegative_predictions,
    rolling_splits,
)


class ResearchUtilityTests(unittest.TestCase):
    def setUp(self):
        changes = np.array([0.4, -0.2, 0.7, -0.5, 0.1, 0.3, -0.4])
        close = 100 + np.cumsum(np.resize(changes, 40))
        self.prices = pd.DataFrame(
            {"Close": close},
            index=pd.date_range("2020-01-01", periods=len(close), freq="D"),
        )

    def test_daily_returns_are_close_to_close_simple_returns(self):
        prices = pd.DataFrame({"Close": [100.0, 110.0, 99.0, 108.9]})
        returns = calculate_daily_returns(prices)
        np.testing.assert_allclose(returns.iloc[1:].to_numpy(), [0.1, -0.1, 0.1])
        self.assertTrue(np.isnan(returns.iloc[0]))

    def test_target_uses_the_next_five_returns(self):
        dataset = build_feature_dataset(self.prices)
        position = 25
        date = self.prices.index[position]
        returns = calculate_daily_returns(self.prices)
        expected = np.sqrt(np.square(returns.iloc[position + 1:position + 6]).sum())
        self.assertAlmostEqual(dataset.loc[date, TARGET], expected, places=14)

    def test_features_do_not_use_prices_after_the_forecast_date(self):
        position = 25
        date = self.prices.index[position]
        changed = self.prices.copy()
        changed.iloc[position + 1:, 0] *= 1.5

        original_features = build_feature_dataset(self.prices).loc[date, FEATURES]
        changed_features = build_feature_dataset(changed).loc[date, FEATURES]
        pd.testing.assert_series_equal(original_features, changed_features)

    def test_baseline_uses_the_previous_five_returns_through_today(self):
        dataset = build_feature_dataset(self.prices)
        position = 25
        date = self.prices.index[position]
        returns = calculate_daily_returns(self.prices)
        expected = np.sqrt(np.square(returns.iloc[position - 4:position + 1]).sum())
        self.assertAlmostEqual(dataset.loc[date, BASELINE], expected, places=14)

    def test_feature_dataset_contains_only_complete_rows_in_original_order(self):
        dataset = build_feature_dataset(self.prices)
        self.assertEqual(list(dataset.columns), FEATURES + [TARGET, BASELINE])
        self.assertEqual(dataset.index[0], self.prices.index[20])
        self.assertEqual(dataset.index[-1], self.prices.index[-6])
        self.assertFalse(dataset.isna().any().any())

    def test_chronological_split_leaves_exactly_the_requested_gap(self):
        data = pd.DataFrame({"position": range(20)})
        train, test = chronological_split(data, train_fraction=0.75, gap=5)
        self.assertEqual(train["position"].tolist(), list(range(10)))
        self.assertEqual(test["position"].tolist(), list(range(15, 20)))

    def test_rolling_splits_expand_training_and_keep_a_gap(self):
        data = pd.DataFrame({"position": range(100)})
        splits = list(rolling_splits(data, 0.6, 0.1, gap=5))
        self.assertEqual(len(splits), 4)
        for i, (train, test) in enumerate(splits):
            start = 60 + i * 10
            self.assertEqual(len(train), start - 5)
            self.assertEqual(test["position"].tolist(), list(range(start, start + 10)))

    def test_negative_predictions_are_clipped_at_zero(self):
        actual = clip_nonnegative_predictions(np.array([-1.0, 0.0, 0.5]))
        np.testing.assert_array_equal(actual, [0.0, 0.0, 0.5])

    def test_audit_detects_a_changed_saved_target(self):
        saved = build_feature_dataset(self.prices)
        checks, _ = audit_data(self.prices, saved)
        self.assertTrue(all(checks.values()))

        corrupted = saved.copy()
        corrupted.iloc[0, corrupted.columns.get_loc(TARGET)] += 0.01
        checks, _ = audit_data(self.prices, corrupted)
        self.assertFalse(checks[f"{TARGET} matches reconstruction"])

    def test_audit_rejects_saved_rows_in_the_wrong_order(self):
        saved = build_feature_dataset(self.prices).iloc[::-1]
        checks, _ = audit_data(self.prices, saved)
        self.assertFalse(checks["dates are sorted"])
        self.assertFalse(checks["dates match reconstructed dataset"])

    def test_block_bootstrap_is_deterministic_and_returns_ordered_intervals(self):
        linear = np.array([0.2, 0.1, 0.3, 0.15, 0.25, 0.12])
        ridge = np.array([0.18, 0.11, 0.29, 0.14, 0.24, 0.13])
        first = bootstrap_block_ci(linear, ridge, block_length=2, n_bootstraps=200)
        second = bootstrap_block_ci(linear, ridge, block_length=2, n_bootstraps=200)
        self.assertEqual(first, second)
        self.assertLessEqual(first["MAE_difference_CI_lower"], first["MAE_difference_CI_upper"])
        self.assertEqual(first["Bootstrap_samples"], 200)


if __name__ == "__main__":
    unittest.main()
