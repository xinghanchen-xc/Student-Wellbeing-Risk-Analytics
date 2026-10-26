# Data and provenance

## Source

The project uses the Kaggle dataset [Student Mental Health and Burnout](https://www.kaggle.com/datasets/sharmajicoder/student-mental-health-and-burnout).

The full source file is intentionally not committed to GitHub. It is large and should be downloaded separately by the person reproducing the full-scale run.

## Included 50K sample

`sample_50k.csv` is the lightweight preview included for CPU reproducibility. It contains the 16 source input fields plus the source-provided `risk_level` label. Previously engineered features, target encodings, outcome fields, and model-prediction columns have been removed. The shared feature code recomputes all eight interaction features at runtime, so the sample does not carry stale feature definitions or predictions from earlier model runs.

The sample was prepared from the project's existing 50K preview file, which belongs to the same Kaggle dataset workflow. The exact row-selection procedure used to create that earlier preview was not independently verified; do not describe this file as a newly drawn random sample of the 1M-row source.

## Full-scale local file

The cleaner accepts the original Kaggle CSV/ZIP or the existing `processed_features_1M.parquet` file, provided it contains the raw source fields and `risk_level`. If a previously processed Parquet file is used, the pipeline discards its existing engineered features and prior predictions and recomputes features using `src/features.py`.

Example:

```bash
python src/01_clean_data.py \
  --input /path/to/processed_features_1M.parquet \
  --out-dir outputs/full_1m/processed
```

## Data limitations

The dataset is described by its publisher as synthetic. Results are useful for demonstrating data engineering and modeling workflows, but do not establish clinical validity or predictive performance on real student populations.
