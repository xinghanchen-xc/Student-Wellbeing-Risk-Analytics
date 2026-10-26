"""Prepare source data for the shared CPU/GPU modeling pipeline.

Supported input formats: CSV, CSV inside ZIP, and Parquet. The input can be the
original Kaggle CSV or the previously saved 1M-row Parquet file. Any precomputed
features, target encodings, or prediction columns are discarded and rebuilt from
the canonical functions in src/features.py.

Example (50K public sample):
  python src/01_clean_data.py --input data/sample_50k.csv --out-dir outputs/processed

Example (full dataset):
  python src/01_clean_data.py --input /path/to/processed_features_1M.parquet --out-dir outputs/full_1m/processed
"""
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from typing import Optional

import pandas as pd

try:
    from .features import (
        CONTEXT_COLUMNS,
        ENGINEERED_FEATURES,
        MODEL_FEATURES,
        RAW_FEATURES,
        TARGET_CODE_COLUMN,
        TARGET_COLUMN,
        add_dashboard_fields,
        feature_formula_descriptions,
        prepare_source_dataframe,
    )
except ImportError:  # running as `python src/01_clean_data.py`
    from features import (
        CONTEXT_COLUMNS,
        ENGINEERED_FEATURES,
        MODEL_FEATURES,
        RAW_FEATURES,
        TARGET_CODE_COLUMN,
        TARGET_COLUMN,
        add_dashboard_fields,
        feature_formula_descriptions,
        prepare_source_dataframe,
    )

RANDOM_STATE = 42


def _largest_csv_in_zip(path: Path) -> str:
    with zipfile.ZipFile(path) as archive:
        candidates = [
            name for name in archive.namelist()
            if name.lower().endswith(".csv") and not name.startswith("__MACOSX/")
        ]
        if not candidates:
            raise ValueError(f"No CSV file found inside ZIP: {path}")
        return max(candidates, key=lambda name: archive.getinfo(name).file_size)


def read_source(path: Path, nrows: Optional[int] = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        try:
            frame = pd.read_parquet(path)
            return frame.head(nrows) if nrows is not None else frame
        except ImportError as exc:
            raise ImportError(
                "Reading Parquet requires pyarrow or fastparquet. Install project dependencies with `pip install -r requirements.txt`."
            ) from exc
    if suffix == ".zip":
        member = _largest_csv_in_zip(path)
        with zipfile.ZipFile(path) as archive, archive.open(member) as handle:
            return pd.read_csv(handle, nrows=nrows, low_memory=False)
    if suffix in {".csv", ".gz"} or path.name.lower().endswith(".csv.gz"):
        return pd.read_csv(path, nrows=nrows, low_memory=False)
    raise ValueError("Unsupported input format. Use .csv, .csv.gz, .zip, or .parquet.")


def add_data_dictionary(out_dir: Path) -> None:
    descriptions = {
        "age": "Age value supplied by the source dataset.",
        "gender": "Gender category supplied by the source dataset.",
        "academic_year": "Academic year level supplied by the source dataset.",
        "study_hours_per_day": "Daily study hours supplied by the source dataset.",
        "exam_pressure": "Exam pressure score supplied by the source dataset.",
        "academic_performance": "Academic performance score supplied by the source dataset.",
        "stress_level": "Stress score supplied by the source dataset.",
        "anxiety_score": "Anxiety score supplied by the source dataset.",
        "depression_score": "Depression score supplied by the source dataset.",
        "sleep_hours": "Sleep hours supplied by the source dataset.",
        "physical_activity": "Physical activity measure supplied by the source dataset.",
        "social_support": "Social support score supplied by the source dataset.",
        "screen_time": "Daily screen time supplied by the source dataset.",
        "internet_usage": "Internet usage measure supplied by the source dataset.",
        "financial_stress": "Financial stress score supplied by the source dataset.",
        "family_expectation": "Family expectation pressure supplied by the source dataset.",
        "risk_level": "Source-provided multiclass target label; Low, Medium, or High.",
        **feature_formula_descriptions(),
        "target_risk_code": "Canonical target encoding: Low=0, Medium=1, High=2.",
        "dropout_risk": "Descriptive outcome retained only when present; excluded from model inputs.",
        "burnout_score": "Descriptive outcome retained only when present; excluded from model inputs.",
        "mental_health_index": "Descriptive outcome retained only when present; excluded from model inputs.",
    }
    pd.DataFrame([{"column": col, "description": desc} for col, desc in descriptions.items()]).to_csv(
        out_dir / "data_dictionary.csv", index=False
    )


def add_summary_tables(df: pd.DataFrame, out_dir: Path) -> None:
    agg = {
        "students": ("student_id", "count"),
        "avg_sleep_hours": ("sleep_hours", "mean"),
        "avg_study_hours": ("study_hours_per_day", "mean"),
        "avg_stress_level": ("stress_level", "mean"),
        "avg_exam_pressure": ("exam_pressure", "mean"),
        "avg_social_support": ("social_support", "mean"),
        "avg_financial_stress": ("financial_stress", "mean"),
    }
    if "burnout_score" in df.columns:
        agg["avg_burnout_score"] = ("burnout_score", "mean")
    if "mental_health_index" in df.columns:
        agg["avg_mental_health_index"] = ("mental_health_index", "mean")
    risk_summary = df.groupby(TARGET_COLUMN, observed=False).agg(**agg).reset_index()
    risk_summary.to_csv(out_dir / "eda_summary_by_risk.csv", index=False)

    behavior_agg = {
        "students": ("student_id", "count"),
        "high_risk_rate": ("is_high_risk", "mean"),
        "avg_sleep_hours": ("sleep_hours", "mean"),
        "avg_study_hours": ("study_hours_per_day", "mean"),
        "avg_stress_level": ("stress_level", "mean"),
    }
    behavior = df.groupby(["sleep_band", "study_band", "behavior_quadrant"], observed=False).agg(**behavior_agg).reset_index()
    behavior.to_csv(out_dir / "behavior_sleep_study_summary.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean source data using shared feature definitions.")
    parser.add_argument("--input", required=True, help="Path to source CSV, CSV ZIP, CSV.GZ, or Parquet file.")
    parser.add_argument("--out-dir", default="outputs/processed", help="Output directory for prepared data and reports.")
    parser.add_argument("--nrows", type=int, default=0, help="Optional row limit for smoke tests; 0 means all rows.")
    parser.add_argument("--dashboard-sample-size", type=int, default=50000, help="Rows for lightweight dashboard sample.")
    args = parser.parse_args()

    input_path = Path(args.input).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    nrows = args.nrows if args.nrows > 0 else None
    raw = read_source(input_path, nrows=nrows)
    prepared, report = prepare_source_dataframe(raw)
    prepared = add_dashboard_fields(prepared)

    # Explicit column order keeps the model schema visible and excludes stale derived/prediction fields.
    ordered = (
        ["student_id"]
        + RAW_FEATURES
        + ENGINEERED_FEATURES
        + [c for c in CONTEXT_COLUMNS if c in prepared.columns]
        + [TARGET_COLUMN, TARGET_CODE_COLUMN, "is_high_risk"]
        + [
            "sleep_band", "study_band", "stress_band", "screen_band", "behavior_quadrant",
            "sleep_deficit_hours", "overstudy_flag", "low_sleep_flag", "high_pressure_flag",
        ]
    )
    prepared = prepared.loc[:, [c for c in ordered if c in prepared.columns]]

    cleaned_path = out_dir / "student_burnout_cleaned.csv.gz"
    prepared.to_csv(cleaned_path, index=False, compression={"method": "gzip", "compresslevel": 1})

    sample_size = min(max(args.dashboard_sample_size, 1), len(prepared))
    dashboard_sample = prepared.sample(n=sample_size, random_state=RANDOM_STATE) if sample_size < len(prepared) else prepared.copy()
    dashboard_sample.to_csv(out_dir / "student_burnout_dashboard_sample.csv", index=False)

    add_summary_tables(prepared, out_dir)
    add_data_dictionary(out_dir)

    report.update({
        "input_file": str(input_path),
        "input_shape": [int(raw.shape[0]), int(raw.shape[1])],
        "prepared_shape": [int(prepared.shape[0]), int(prepared.shape[1])],
        "prepared_columns": prepared.columns.tolist(),
        "missing_values_in_model_inputs": prepared[MODEL_FEATURES].isna().sum().to_dict(),
        "feature_formulas": feature_formula_descriptions(),
        "model_features": MODEL_FEATURES,
        "outputs": {
            "cleaned_dataset": str(cleaned_path),
            "dashboard_sample": str(out_dir / "student_burnout_dashboard_sample.csv"),
            "eda_summary_by_risk": str(out_dir / "eda_summary_by_risk.csv"),
            "behavior_sleep_study_summary": str(out_dir / "behavior_sleep_study_summary.csv"),
            "data_dictionary": str(out_dir / "data_dictionary.csv"),
        },
        "note": "Model-input imputation is fit inside the training pipeline on the training split only.",
    })
    with open(out_dir / "cleaning_report.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=str)

    print(json.dumps(report, indent=2, default=str))
    print(f"Prepared data saved to: {cleaned_path}")


if __name__ == "__main__":
    main()
