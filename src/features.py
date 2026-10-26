"""Shared data schema and feature engineering for CPU and GPU experiments.

The feature formulas below intentionally match the original Task 1 data-engineering
workflow. Both the 50K CPU sample and the full-scale GPU run import this module so
that the two paths do not silently drift apart.
"""
from __future__ import annotations

import re
from typing import Iterable

import numpy as np
import pandas as pd

RANDOM_STATE = 42
RISK_ORDER = ["Low", "Medium", "High"]
RISK_TO_CODE = {label: idx for idx, label in enumerate(RISK_ORDER)}
CODE_TO_RISK = {idx: label for label, idx in RISK_TO_CODE.items()}

CATEGORICAL_FEATURES = ["gender"]
RAW_NUMERIC_FEATURES = [
    "age",
    "academic_year",
    "study_hours_per_day",
    "exam_pressure",
    "academic_performance",
    "stress_level",
    "anxiety_score",
    "depression_score",
    "sleep_hours",
    "physical_activity",
    "social_support",
    "screen_time",
    "internet_usage",
    "financial_stress",
    "family_expectation",
]
RAW_FEATURES = CATEGORICAL_FEATURES + RAW_NUMERIC_FEATURES

ENGINEERED_FEATURES = [
    "academic_resilience",
    "study_sleep_gap",
    "socio_economic_multiplier",
    "effort_reward_index",
    "total_cognitive_load",
    "anxiety_pressure_impact",
    "social_support_ratio",
    "combined_distress_index",
]
NUMERIC_FEATURES = RAW_NUMERIC_FEATURES + ENGINEERED_FEATURES
MODEL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
TARGET_COLUMN = "risk_level"
TARGET_CODE_COLUMN = "target_risk_code"

# These columns may be retained for descriptive dashboards, but are never model inputs.
CONTEXT_COLUMNS = ["burnout_score", "mental_health_index", "dropout_risk"]
LEAKAGE_COLUMNS = {
    "burnout_score",
    "mental_health_index",
    "dropout_risk",
    "risk_level",
    "target",
    "target_risk_code",
    "is_high_risk",
    "predicted_risk_index",
    "predicted_risk_label",
    "predicted_risk_label_model",
    "predicted_risk_code_model",
    "p_low",
    "p_medium",
    "p_high",
    "prediction_confidence",
    "prediction_correct",
    "error_type",
}


def snake_case(name: object) -> str:
    value = str(name).strip()
    value = re.sub(r"[^0-9a-zA-Z]+", "_", value)
    value = re.sub(r"_+", "_", value)
    return value.strip("_").lower()


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [snake_case(c) for c in out.columns]
    return out


def standardize_gender(series: pd.Series) -> pd.Series:
    values = series.astype("string").str.strip().str.title()
    values = values.replace(
        {
            "M": "Male",
            "Man": "Male",
            "F": "Female",
            "Woman": "Female",
            "Non Binary": "Other",
            "Non-Binary": "Other",
            "Prefer Not To Say": "Other",
            "Nan": pd.NA,
            "None": pd.NA,
            "": pd.NA,
        }
    )
    return values.where(values.isin(["Male", "Female", "Other"]), "Other").fillna("Other")


def standardize_risk_level(series: pd.Series) -> pd.Series:
    values = series.astype("string").str.strip().str.title()
    values = values.replace({"Med": "Medium", "Moderate": "Medium", "Hi": "High"})
    return values.where(values.isin(RISK_ORDER), pd.NA)


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with the canonical eight interaction features.

    Formulas are inherited from the original 1M-row Task 1 notebook. Names are
    project labels, not claims that these are validated clinical constructs.
    """
    out = df.copy()
    required = set(RAW_NUMERIC_FEATURES)
    missing = sorted(required.difference(out.columns))
    if missing:
        raise ValueError(f"Cannot engineer features; missing source columns: {missing}")

    # Keep formulas consistent with the original GPU data-engineering notebook.
    out["academic_resilience"] = out["exam_pressure"] / (out["sleep_hours"] + 1.0)
    out["study_sleep_gap"] = out["study_hours_per_day"] - out["sleep_hours"]
    out["socio_economic_multiplier"] = out["financial_stress"] * out["exam_pressure"]
    out["effort_reward_index"] = out["study_hours_per_day"] / (out["academic_performance"] + 0.1)
    out["total_cognitive_load"] = out["study_hours_per_day"] + out["screen_time"]
    out["anxiety_pressure_impact"] = out["anxiety_score"] * out["exam_pressure"]
    out["social_support_ratio"] = out["social_support"] / (out["family_expectation"] + 1.0)
    out["combined_distress_index"] = out["depression_score"] + out["anxiety_score"]
    return out


def prepare_source_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Build a safe modeling/dashboard frame from raw or previously processed data.

    Only source input columns, the source risk label, and a small optional set of
    descriptive outcome columns survive. Existing engineered columns, encoded
    targets, and prior predictions are deliberately discarded and recomputed.
    """
    source = normalize_columns(df)
    required = RAW_FEATURES + [TARGET_COLUMN]
    missing = [col for col in required if col not in source.columns]
    if missing:
        raise ValueError(
            "Input must contain the original source features and risk_level. "
            f"Missing columns: {missing}. Do not use a prediction-only export as input."
        )

    keep_columns = RAW_FEATURES + [TARGET_COLUMN] + [
        col for col in CONTEXT_COLUMNS if col in source.columns
    ]
    ignored_columns = [col for col in source.columns if col not in keep_columns]
    out = source.loc[:, keep_columns].copy()

    for col in RAW_NUMERIC_FEATURES + [c for c in CONTEXT_COLUMNS if c in out.columns]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["gender"] = standardize_gender(out["gender"])
    out[TARGET_COLUMN] = standardize_risk_level(out[TARGET_COLUMN])

    before_rows = len(out)
    out = out.dropna(subset=[TARGET_COLUMN]).reset_index(drop=True)
    invalid_target_rows_dropped = before_rows - len(out)
    if out.empty:
        raise ValueError("No valid risk_level labels remain after normalization.")

    out = add_engineered_features(out)
    out[TARGET_CODE_COLUMN] = out[TARGET_COLUMN].map(RISK_TO_CODE).astype("int8")
    out["is_high_risk"] = (out[TARGET_COLUMN] == "High").astype("int8")
    report = {
        "input_rows": int(len(source)),
        "rows_after_target_validation": int(len(out)),
        "invalid_target_rows_dropped": int(invalid_target_rows_dropped),
        "ignored_columns": ignored_columns,
        "model_features": MODEL_FEATURES,
        "context_columns_retained_for_descriptive_analysis": [
            col for col in CONTEXT_COLUMNS if col in out.columns
        ],
        "leakage_columns_excluded_from_model_inputs": sorted(LEAKAGE_COLUMNS),
        "risk_distribution": out[TARGET_COLUMN].value_counts().reindex(RISK_ORDER, fill_value=0).astype(int).to_dict(),
    }
    return out, report


def add_dashboard_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Add presentation-only fields; none are used as model inputs."""
    out = df.copy()
    if "student_id" not in out.columns:
        out.insert(0, "student_id", np.arange(1, len(out) + 1, dtype=np.int64))
    out["sleep_band"] = pd.cut(
        out["sleep_hours"],
        bins=[-np.inf, 5, 6, 7, 8, np.inf],
        labels=["<5h", "5-6h", "6-7h", "7-8h", "8h+"],
        ordered=True,
    ).astype("string")
    out["study_band"] = pd.cut(
        out["study_hours_per_day"],
        bins=[-np.inf, 2, 4, 6, 8, np.inf],
        labels=["<2h", "2-4h", "4-6h", "6-8h", "8h+"],
        ordered=True,
    ).astype("string")
    out["stress_band"] = pd.cut(
        out["stress_level"],
        bins=[-np.inf, 3, 6, 8, np.inf],
        labels=["Low stress", "Moderate stress", "High stress", "Severe stress"],
        ordered=True,
    ).astype("string")
    out["screen_band"] = pd.cut(
        out["screen_time"],
        bins=[-np.inf, 3, 5, 7, np.inf],
        labels=["Low screen", "Medium screen", "High screen", "Very high screen"],
        ordered=True,
    ).astype("string")

    median_study = out["study_hours_per_day"].median()
    high_study = out["study_hours_per_day"] >= median_study
    enough_sleep = out["sleep_hours"] >= 7.0
    out["behavior_quadrant"] = np.select(
        [high_study & enough_sleep, high_study & ~enough_sleep, ~high_study & enough_sleep, ~high_study & ~enough_sleep],
        ["High study / adequate sleep", "High study / low sleep", "Lower study / adequate sleep", "Lower study / low sleep"],
        default="Unclassified",
    )
    out["sleep_deficit_hours"] = np.maximum(0.0, 7.0 - out["sleep_hours"])
    out["overstudy_flag"] = (out["study_hours_per_day"] > 8.0).astype("int8")
    out["low_sleep_flag"] = (out["sleep_hours"] < 6.0).astype("int8")
    out["high_pressure_flag"] = (out["exam_pressure"] >= 8.0).astype("int8")
    return out


def feature_formula_descriptions() -> dict[str, str]:
    return {
        "academic_resilience": "exam_pressure / (sleep_hours + 1); inherited from the original Task 1 workflow.",
        "study_sleep_gap": "study_hours_per_day - sleep_hours.",
        "socio_economic_multiplier": "financial_stress * exam_pressure; inherited from the original Task 1 workflow.",
        "effort_reward_index": "study_hours_per_day / (academic_performance + 0.1).",
        "total_cognitive_load": "study_hours_per_day + screen_time.",
        "anxiety_pressure_impact": "anxiety_score * exam_pressure.",
        "social_support_ratio": "social_support / (family_expectation + 1).",
        "combined_distress_index": "depression_score + anxiety_score.",
    }
