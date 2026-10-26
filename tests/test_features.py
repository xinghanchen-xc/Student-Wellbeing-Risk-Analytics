from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.features import (  # noqa: E402
    ENGINEERED_FEATURES,
    LEAKAGE_COLUMNS,
    MODEL_FEATURES,
    RISK_TO_CODE,
    TARGET_CODE_COLUMN,
    add_engineered_features,
    prepare_source_dataframe,
)


class FeaturePipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sample_path = REPO_ROOT / "data" / "sample_50k.csv"

    def test_sample_contains_source_fields_not_prior_predictions(self):
        raw = pd.read_csv(self.sample_path, nrows=25)
        self.assertIn("risk_level", raw.columns)
        self.assertNotIn("predicted_risk_label", raw.columns)
        self.assertNotIn("predicted_risk_index", raw.columns)
        self.assertFalse(any(col in raw.columns for col in ENGINEERED_FEATURES))

    def test_feature_formulas_and_target_encoding(self):
        raw = pd.read_csv(self.sample_path, nrows=25)
        prepared, report = prepare_source_dataframe(raw)
        first_raw = raw.iloc[0]
        first = prepared.iloc[0]

        self.assertAlmostEqual(
            float(first["academic_resilience"]),
            float(first_raw["exam_pressure"] / (first_raw["sleep_hours"] + 1.0)),
            places=8,
        )
        self.assertAlmostEqual(
            float(first["socio_economic_multiplier"]),
            float(first_raw["financial_stress"] * first_raw["exam_pressure"]),
            places=8,
        )
        self.assertAlmostEqual(
            float(first["effort_reward_index"]),
            float(first_raw["study_hours_per_day"] / (first_raw["academic_performance"] + 0.1)),
            places=8,
        )
        self.assertEqual(int(first[TARGET_CODE_COLUMN]), RISK_TO_CODE[first["risk_level"]])
        self.assertEqual(report["rows_after_target_validation"], len(prepared))

    def test_model_allowlist_excludes_outcomes_and_prediction_columns(self):
        self.assertTrue(all(col not in MODEL_FEATURES for col in LEAKAGE_COLUMNS))
        self.assertIn("gender", MODEL_FEATURES)
        self.assertIn("combined_distress_index", MODEL_FEATURES)

    def test_existing_engineered_fields_are_recomputed(self):
        raw = pd.read_csv(self.sample_path, nrows=5)
        with_stale_feature = raw.copy()
        with_stale_feature["academic_resilience"] = -999
        prepared, _ = prepare_source_dataframe(with_stale_feature)
        self.assertNotEqual(float(prepared.iloc[0]["academic_resilience"]), -999.0)


if __name__ == "__main__":
    unittest.main()
