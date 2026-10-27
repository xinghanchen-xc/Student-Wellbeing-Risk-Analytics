# Student Well-Being Risk Analytics

A portfolio project demonstrating a reproducible data workflow for
student well-being data: data cleaning, shared feature engineering,
multiclass XGBoost, model evaluation, SHAP interpretation, and
dashboards.

> **Data and responsible-use note:** The source dataset is described as
> synthetic (artificially generated), not as verified records from real
> students. This project demonstrates data engineering and
> machine-learning workflows; it is not a clinically validated
> assessment, and its results should not be used to make real
> student-support decisions or assumed to generalize to real-world
> populations.

## Project highlights

- **Dataset source:** [Student Mental Health and Burnout —
  Kaggle](https://www.kaggle.com/datasets/sharmajicoder/student-mental-health-and-burnout).
- **Full-scale experiment:** Reprocessed the 1M-row dataset with a
  shared feature pipeline and evaluated the model on a 200,000-row
  stratified holdout set.
- **Full-scale model result:** 79.78% accuracy, 0.8168 weighted F1, and
  0.6279 macro F1. The model is trained with XGBoost on CUDA; inference
  is performed on CPU.
- **Fast reproduction path:** Includes a 50,000-row sample for running
  the modular CPU workflow without downloading the full dataset.
- **Shared model logic:** CPU and GPU runs use the same feature
  definitions, label mapping, split strategy, XGBoost configuration,
  class-weighting approach, and evaluation code.
- **Leakage prevention:** Uses an explicit model-input allowlist.
  `burnout_score`, `mental_health_index`, `dropout_risk`, prior
  predictions, target encodings, and other target-derived fields are
  excluded from model inputs.
- **GPU data-ingestion benchmark:** A prior Google Colab T4 single-read
  test recorded pandas at 4.9373 seconds and RAPIDS cuDF at 0.2279
  seconds (about 21.7× faster for that specific read). This is a
  single-run CSV-ingestion result, not an end-to-end pipeline or
  model-training speedup.
- **Interpretability and presentation:** Includes SHAP outputs, feature
  importance, Plotly visualizations, a Streamlit prototype, and Power BI
  dashboard screenshots.

## Repository structure

``` text
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
│   ├── reference_run/       # 50K CPU reference outputs
│   ├── full_1m/             # 1M GPU experiment outputs
│   ├── figures/
│   └── powerbi/
├── tests/
│   └── test_features.py
└── PUBLISH_TO_GITHUB.md
```

The full dataset and large intermediate files are intentionally not
committed to GitHub. See [`data/README.md`](data/README.md) for data
provenance and local setup notes.

This repository is a curated portfolio version of a DATA 230 team
project. The original work was completed collaboratively; keep team
attribution and descriptions of individual contributions accurate.

- **Original team repository:**
  [DATA230-Final-Group-Project](https://github.com/xinghanchen-xc/DATA230-Final-Group-Project)
- **Portfolio repository:**
  [Student-Wellbeing-Risk-Analytics](https://github.com/xinghanchen-xc/Student-Wellbeing-Risk-Analytics)

## Quick start: reproduce the 50K CPU sample

Python 3.10 or 3.11 is recommended. Run the commands from the repository
root.

### 1. Create an environment and install dependencies

**Windows PowerShell**

``` powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

**macOS / Linux**

``` bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 2. Clean the sample data

``` bash
python src/01_clean_data.py --input data/sample_50k.csv --out-dir outputs/processed --dashboard-sample-size 10000
```

This creates the cleaned dataset, a data-quality report, and dashboard
summaries. Missing-value imputation for model inputs is fitted inside
the model pipeline using the training split, rather than calculated from
the entire dataset before splitting.

### 3. Train and evaluate on CPU

``` bash
python src/02_train_model_generate_results.py --cleaned outputs/processed/student_burnout_cleaned.csv.gz --out-dir outputs --device cpu --test-size 0.2 --dashboard-sample-size 10000 --shap-sample-size 0
```

The run generates model artifacts, metrics, a classification report, a
confusion matrix, feature-importance outputs, dashboard tables, and
Plotly figures under `outputs/`. After the basic pipeline succeeds, SHAP
can be enabled with `--shap-sample-size 300`.

### 4. Launch the Streamlit prototype

``` bash
streamlit run app/streamlit_app.py
```

By default, the app reads artifacts from `outputs/`. To point it at
another run, set `WELLBEING_OUTPUT_DIR` to that run’s output directory.

**Windows PowerShell example**

``` powershell
$env:WELLBEING_OUTPUT_DIR = "outputs/full_1m/run"
streamlit run app/streamlit_app.py
```

**macOS / Linux example**

``` bash
WELLBEING_OUTPUT_DIR=outputs/full_1m/run streamlit run app/streamlit_app.py
```

## Full-scale 1M experiment

The full source dataset is not committed to GitHub. Download it from
Kaggle or use the local `processed_features_1M.parquet` file if it
contains the source input fields and the `risk_level` label. The cleaner
discards precomputed interaction features and prior prediction outputs,
then recomputes features from `src/features.py`.

Example commands:

``` bash
python src/01_clean_data.py --input /path/to/processed_features_1M.parquet --out-dir outputs/full_1m/processed --dashboard-sample-size 100000
python src/02_train_model_generate_results.py --cleaned outputs/full_1m/processed/student_burnout_cleaned.csv.gz --out-dir outputs/full_1m/run --device cuda --test-size 0.2 --dashboard-sample-size 100000 --shap-sample-size 300
```

Replace the input path with the actual local path. The notebook
`notebooks/02_full_scale_gpu_experiment.ipynb` demonstrates how to point
the workflow to a file in Google Drive from Colab.

The full-scale run uses a stratified 80/20 split with random seed 42 and
balanced sample weights computed from the training split. The requested
training device is CUDA. Inference is explicitly run on CPU because the
preprocessing pipeline produces NumPy arrays on CPU; the metrics file
records both `training_device` and `inference_device`. GPU training does
not silently fall back to CPU if CUDA fails.

## Shared feature definitions

Both CPU and GPU workflows import the same feature definitions from
`src/features.py`. Eight interaction features are recomputed from the
source inputs. These are project-specific engineered features, not
validated clinical constructs.

| Feature                     | Formula                                              |
|:----------------------------|:-----------------------------------------------------|
| `academic_resilience`       | `exam_pressure / (sleep_hours + 1)`                  |
| `study_sleep_gap`           | `study_hours_per_day - sleep_hours`                  |
| `socio_economic_multiplier` | `financial_stress * exam_pressure`                   |
| `effort_reward_index`       | `study_hours_per_day / (academic_performance + 0.1)` |
| `total_cognitive_load`      | `study_hours_per_day + screen_time`                  |
| `anxiety_pressure_impact`   | `anxiety_score * exam_pressure`                      |
| `social_support_ratio`      | `social_support / (family_expectation + 1)`          |
| `combined_distress_index`   | `depression_score + anxiety_score`                   |

The target is the source-provided `risk_level`, encoded consistently as
Low=0, Medium=1, and High=2. The model uses an explicit allowlist of
source inputs and engineered features. It excludes `burnout_score`,
`mental_health_index`, `dropout_risk`, `risk_level`, target encodings,
and prior prediction/probability fields even if these columns exist in
an input file.

## Evaluation results

### Full-scale 1M-row GPU-training experiment

The latest repaired full-scale run used 1,000,000 rows: 800,000 for
training and 200,000 for testing. The model trained on CUDA and
performed inference on CPU.

| Metric                 | Holdout result |
|:-----------------------|---------------:|
| Accuracy               |         0.7978 |
| Balanced accuracy      |         0.7935 |
| Macro F1               |         0.6279 |
| Weighted F1            |         0.8168 |
| Macro ROC-AUC (OvR)    |         0.9259 |
| Weighted ROC-AUC (OvR) |         0.9190 |

**Per-class results on the 200,000-row holdout set**

| Risk class | Precision | Recall | F1-score | Support |
|:-----------|----------:|-------:|---------:|--------:|
| Low        |    0.9585 | 0.8217 |   0.8849 | 153,329 |
| Medium     |    0.5299 | 0.7102 |   0.6069 |  43,655 |
| High       |    0.2548 | 0.8485 |   0.3919 |   3,016 |

The model identifies 2,559 of 3,016 actual High-risk rows (84.85%
recall), but High-risk precision is 25.48%, so the model also produces
many false positives for this class. The High-risk class is rare in the
holdout set; overall accuracy alone therefore does not fully describe
performance. See `reports/full_1m/classification_report.csv` and
`reports/full_1m/confusion_matrix.csv` for the detailed outputs.

The confusion matrix below uses actual classes as rows and predicted
classes as columns:

| Actual / Predicted |     Low | Medium |  High |
|:------------------|--------:|-------:|------:|
| Low               | 125,994 | 27,046 |   289 |
| Medium            |   5,456 | 31,003 | 7,196 |
| High              |       0 |    457 | 2,559 |

### 50K CPU reference workflow

The included 50K sample provides a lightweight CPU reproduction path.
The reference run used a stratified 80/20 split, with 40,000 training
rows and 10,000 test rows.

| Metric              | 50K CPU reference result |
|:--------------------|-------------------------:|
| Accuracy            |                   0.8177 |
| Balanced accuracy   |                   0.7476 |
| Macro F1            |                   0.6613 |
| Weighted F1         |                   0.8308 |
| High-risk precision |                   0.3418 |
| High-risk recall    |                   0.6309 |

These are the recorded results from the repaired pipeline. Small
differences may occur across software or hardware versions.

**Important:** Do not interpret the difference between the 50K CPU and
1M GPU metrics as a CPU-versus-GPU performance comparison. The datasets
differ in size, and the exact row-selection procedure used for the
existing 50K preview was not independently verified as a random sample
drawn from the 1M source. The two results should be read as separate
reference runs.

## Data provenance and limitations

- The source dataset is described as synthetic/artificially generated.
  It should not be treated as verified observations from real students.
- The included `data/sample_50k.csv` is derived from the project’s
  existing 50K preview from the same Kaggle dataset workflow. Its exact
  original row-selection procedure was not independently verified; this
  repository does not claim it is a newly drawn random sample from the
  1M-row source.
- The complete source data and large intermediate files are not
  committed to GitHub. See `data/README.md` for setup guidance.
- The 21.7× cuDF result is a single-run CSV-ingestion benchmark. It is
  not a general speedup claim for data processing, model training, or
  the entire pipeline.
- Model performance on synthetic data does not establish generalization
  to real student populations, clinical validity, or suitability for
  real student-support decisions.
- The High-risk class has high recall but low precision in the
  full-scale result. Any real-world use would require representative
  real data, careful validation, and human oversight; this project is
  for educational demonstration only.

## Reports and artifacts

The `reports/` directory contains the committed 50K CPU reference
outputs and full-scale evaluation files, including:

- `model_metrics.json`
- `classification_report.csv`
- `confusion_matrix.csv`
- `feature_importance.csv`
- `shap_feature_importance.csv`
- `shap_local_strength_sample.csv`
- ROC and precision-recall curve point data

The full-scale metrics were generated by the repaired workflow. The
historical 0.88 Accuracy / 0.87 Weighted F1 results from the earlier,
pre-repair experiment are not reported as the final model results
because the feature allowlist and leakage handling changed.

## Project background

This repository is a portfolio-oriented organization of work from a DATA
230 team project at San José State University. It consolidates data
engineering, modeling, interpretation, and visualization work into a
reproducible structure. The original work was collaborative, so team
credit and individual contribution statements should accurately reflect
that context.

- **Original team repository:**
  [DATA230-Final-Group-Project](https://github.com/xinghanchen-xc/DATA230-Final-Group-Project)
- **Portfolio repository:**
  [Student-Wellbeing-Risk-Analytics](https://github.com/xinghanchen-xc/Student-Wellbeing-Risk-Analytics)

Developed for **DATA 230: Data Visualization**, San José State
University.
