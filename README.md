# Student Well-Being Risk Analytics

A portfolio project demonstrating a reproducible data workflow for a student well-being dataset: data cleaning, shared feature engineering, multiclass XGBoost, evaluation, SHAP interpretation, and dashboards.

> **Scope and responsible use:** The Kaggle publisher describes this dataset as synthetic. This project is an educational modeling and data-engineering exercise, not a clinically validated assessment. Results on this dataset should not be assumed to generalize to real students or used to make real student-support decisions.

## Project highlights

- **Data source:** [Student Mental Health and Burnout — Kaggle](https://www.kaggle.com/datasets/sharmajicoder/student-mental-health-and-burnout).
- **Full-scale workflow:** supports the original 1M-row source file or the previously saved `processed_features_1M.parquet` file. The full dataset is not committed to GitHub.
- **Reproducible sample:** includes a 50,000-row preview containing source input fields and the source `risk_level` label only. Engineered features and prior model predictions are removed and recomputed at runtime.
- **Shared model logic:** CPU and GPU runs use the same feature definitions, label mapping, train/test split strategy, XGBoost parameters, class-weighting approach, and evaluation code. Only data size and requested training device differ.
- **Leakage prevention:** the model uses an explicit input-feature allowlist. `burnout_score`, `mental_health_index`, `dropout_risk`, prior predictions, and target-derived columns are excluded from model inputs.
- **Recorded ingestion benchmark:** a prior Google Colab T4 single-read test recorded pandas at 4.9373 seconds and RAPIDS cuDF at 0.2279 seconds (about 21.67× for that particular CSV read). This is a historical single-run result, not an end-to-end pipeline speedup.
- **Interpretability and presentation:** SHAP feature importance, Plotly outputs, a model-backed Streamlit prototype, and Power BI dashboard screenshots.

## Repository structure

```text
Student-Wellbeing-Risk-Analytics/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── README.md
│   └── sample_50k.csv
├── src/
│   ├── features.py
│   ├── 01_clean_data.py
│   ├── 02_train_model_generate_results.py
│   └── 03_plotly_dashboard.py
├── notebooks/
│   ├── 01_csv_ingestion_benchmark.ipynb
│   └── 02_full_scale_gpu_experiment.ipynb
├── app/
│   └── streamlit_app.py
├── reports/
│   ├── reference_run/
│   ├── figures/
│   └── powerbi/
├── PUBLISH_TO_GITHUB.md
└── tests/
    └── test_features.py
```

This is a curated portfolio version of a DATA 230 team project. Keep team attribution accurate when presenting it; the original work was completed in a collaborative course context.

- **Original team repository:** [DATA230-Final-Group-Project](https://github.com/xinghanchen-xc/DATA230-Final-Group-Project)
- **Portfolio repository:** [Student-Wellbeing-Risk-Analytics](https://github.com/xinghanchen-xc/Student-Wellbeing-Risk-Analytics)

## Quick start: reproduce the 50K CPU sample

Python 3.10 or 3.11 is recommended. Run these commands from the repository root.

### 1. Create an environment and install dependencies

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

**macOS / Linux**

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 2. Prepare the sample data

```bash
python src/01_clean_data.py --input data/sample_50k.csv --out-dir outputs/processed --dashboard-sample-size 10000
```

This creates a cleaned dataset, data dictionary, data-quality report, and dashboard summaries. Imputation for model inputs is fitted inside the training pipeline on the training split only; it is not calculated from the full dataset before splitting.

### 3. Train and evaluate on CPU

```bash
python src/02_train_model_generate_results.py --cleaned outputs/processed/student_burnout_cleaned.csv.gz --out-dir outputs --device cpu --test-size 0.2 --dashboard-sample-size 10000 --shap-sample-size 0
```

The run writes a model package, metrics JSON, classification report, confusion matrix, feature importance, dashboard tables, and Plotly figures under `outputs/`. SHAP can be enabled with `--shap-sample-size 300` after the basic pipeline succeeds.

### 4. Launch the Streamlit prototype

```bash
streamlit run app/streamlit_app.py
```

The app reads artifacts from `outputs/` by default. To use artifacts from another run, set `WELLBEING_OUTPUT_DIR` to that run's output directory.

**Windows PowerShell example:**

```powershell
$env:WELLBEING_OUTPUT_DIR = "outputs/full_1m/run"
streamlit run app/streamlit_app.py
```

**macOS / Linux example:**

```bash
WELLBEING_OUTPUT_DIR=outputs/full_1m/run streamlit run app/streamlit_app.py
```

## Full-scale 1M GPU experiment

The full source dataset is not committed. Download it from Kaggle or use the local `processed_features_1M.parquet` file if it contains the raw source fields and `risk_level` label. The cleaner deliberately drops previously engineered features and prediction outputs, then recomputes features from `src/features.py`.

```bash
python src/01_clean_data.py --input /path/to/processed_features_1M.parquet --out-dir outputs/full_1m/processed --dashboard-sample-size 100000
python src/02_train_model_generate_results.py --cleaned outputs/full_1m/processed/student_burnout_cleaned.csv.gz --out-dir outputs/full_1m/run --device cuda --test-size 0.2 --dashboard-sample-size 100000 --shap-sample-size 300
```

Replace the input path with your actual local file path. On Google Colab, the notebook under `notebooks/` demonstrates how to point the workflow to a file on Google Drive.

The GPU run intentionally **does not silently fall back to CPU**. If CUDA training fails, the run stops with an error so the reported device is truthful. XGBoost is trained on the requested device; inference is explicitly run on CPU after training because the scikit-learn preprocessing pipeline produces NumPy arrays on CPU. Both device settings are recorded in `model_metrics.json`.

## Shared feature definitions

Both CPU and GPU paths import the same functions and feature lists from `src/features.py`. The eight features are recomputed from the 16 source input fields. The formulas follow the original Task 1 workflow for consistency; their names are project-specific analytical features, not validated clinical constructs.

| Feature | Formula |
|---|---|
| `academic_resilience` | `exam_pressure / (sleep_hours + 1)` |
| `study_sleep_gap` | `study_hours_per_day - sleep_hours` |
| `socio_economic_multiplier` | `financial_stress * exam_pressure` |
| `effort_reward_index` | `study_hours_per_day / (academic_performance + 0.1)` |
| `total_cognitive_load` | `study_hours_per_day + screen_time` |
| `anxiety_pressure_impact` | `anxiety_score * exam_pressure` |
| `social_support_ratio` | `social_support / (family_expectation + 1)` |
| `combined_distress_index` | `depression_score + anxiety_score` |

The model target is the source-provided `risk_level`, encoded consistently as Low=0, Medium=1, High=2. The model feature allowlist includes the original input fields plus the eight engineered features. The following are excluded from model inputs even if present in the input file: `burnout_score`, `mental_health_index`, `dropout_risk`, `risk_level`, target encodings, and prior prediction/probability fields.

## Evaluation and reporting

The committed `reports/reference_run/` files were regenerated after the feature pipeline was unified. The included 50K sample produced the following reference result on a stratified 80/20 holdout split (10,000 test rows):

| Metric | 50K CPU reference run |
|---|---:|
| Accuracy | 0.8177 |
| Balanced accuracy | 0.7476 |
| Macro F1 | 0.6613 |
| Weighted F1 | 0.8308 |
| High-risk precision | 0.3418 |
| High-risk recall | 0.6309 |

These are the actual outputs of the repaired pipeline in the build environment, not the earlier pre-refactor metrics and not the full 1M-row GPU result. The run uses random seed 42, balanced sample weights computed from the training split, and the model parameters saved in `model_metrics.json`. Re-run the commands above to regenerate metrics in your environment.

Report more than accuracy: inspect macro F1, weighted F1, per-class precision/recall, and the confusion matrix. The High-risk class is rare in the included sample (149 examples in the 10,000-row holdout), so a high overall accuracy alone can hide poor performance for that class.

## Data and experiment limitations

- The sample and full-scale workflow use the same Kaggle dataset source and the same source target label, but the included preview's exact row-selection procedure was not independently verified. Do not describe it as a newly drawn random sample from the 1M source.
- The 50K sample is a lightweight CPU reproduction path. The full 1M dataset remains local and is not committed to GitHub.
- The historical 21.67× result is from one CSV-ingestion timing run. It should not be described as a general speedup for feature engineering or model training.
- GPU and CPU model metrics should be compared only after confirming that the same dataset rows, feature definitions, label encoding, split, and training parameters were used.
- Because the dataset is synthetic, model performance does not establish real-world or clinical validity.

## Project background and contributions

This repository is a portfolio-oriented organization of work from a DATA 230 team project at San José State University. The portfolio owner led substantial technical implementation across data engineering, modeling, interpretation, and visualization; the work originated in a collaborative course setting. Team credit and role descriptions should remain consistent with the [original course repository](https://github.com/xinghanchen-xc/DATA230-Final-Group-Project).

Developed for **DATA 230: Data Visualization**, San José State University.
