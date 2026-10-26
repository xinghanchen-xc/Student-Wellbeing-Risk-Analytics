# Student Well-Being Risk Analytics

A portfolio-focused data science project exploring a reproducible workflow for student well-being analytics: data cleaning, feature engineering, multiclass XGBoost classification, model evaluation, SHAP-based interpretation, and dashboarding.

> **Responsible use:** This is an educational analytics prototype. It predicts labels supplied by the dataset and is not a clinically validated mental-health assessment. Do not use it to diagnose individuals or make real student-support decisions.

## Project at a glance

- **Dataset:** [Student Mental Health and Burnout (Kaggle)](https://www.kaggle.com/datasets/sharmajicoder/student-mental-health-and-burnout). This repository includes a 50,000-row preview; the full 1M-row source file and trained model artifacts are not committed.
- **Data engineering:** A modular cleaning pipeline creates eight interaction features and exports processed data for downstream modeling and visualization.
- **GPU benchmark:** In one recorded Google Colab T4 CSV-ingestion test, pandas took **4.9373 s** and RAPIDS cuDF took **0.2279 s** (about **21.67× faster for that single read test**). This is not an end-to-end pipeline speedup.
- **Modeling:** Multiclass XGBoost for the dataset-provided Low / Medium / High risk labels, evaluated on a stratified holdout split.
- **Interpretability and delivery:** SHAP feature analysis, Plotly outputs, a model-backed Streamlit prototype, and Power BI dashboard screenshots.

## Repository structure

```text
student-wellbeing-risk-analytics/
├── README.md
├── requirements.txt
├── data/
│   └── Burnout_Predictions_Preview_50k.csv
├── src/
│   ├── 01_clean_data.py
│   ├── 02_train_model_generate_results.py
│   └── 03_plotly_dashboard.py
├── app/
│   └── streamlit_app.py
├── notebooks/
│   └── task1_gpu_data_engineering.ipynb
└── reports/
    └── powerbi/
        ├── Page1 Executive Overview Burnout Risk Analysis.png
        └── Page2 Predictive Diagnostics & Feature Impact.png
```

This is a curated portfolio version of a DATA 230 team project. It focuses on the modular pipeline and selected presentation assets rather than reproducing every file from the original course submission. Keep team attribution accurate when presenting the project; the technical work was completed in a team context.

**Original team repository:** [DATA230-Final-Group-Project](https://github.com/xinghanchen-xc/DATA230-Final-Group-Project). This repository is the cleaned, portfolio-oriented version; the original repository preserves the course project's history and collaborative context.

## Quick start

Python 3.10 or 3.11 is recommended. Commands below run from the repository root and use the included 50,000-row preview on CPU.

### 1. Create an environment and install dependencies

```bash
python -m venv .venv
```

Activate it, then install requirements:

```bash
# macOS / Linux
source .venv/bin/activate

# Windows PowerShell: .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 2. Clean the preview data

```bash
python src/01_clean_data.py \
  --input data/Burnout_Predictions_Preview_50k.csv \
  --out-dir outputs/processed \
  --dashboard-sample-size 10000
```

### 3. Train and evaluate XGBoost

```bash
python src/02_train_model_generate_results.py \
  --cleaned outputs/processed/student_burnout_cleaned.csv.gz \
  --out-dir outputs \
  --device cpu \
  --dashboard-sample-size 10000 \
  --shap-sample-size 300
```

The run writes a model package, holdout metrics, classification report, confusion matrix, feature importance, optional SHAP outputs, dashboard tables, and Plotly figures under `outputs/`. Generated outputs are excluded from Git by `.gitignore`.

### 4. Launch the Streamlit prototype

Run after the cleaning and training steps above:

```bash
streamlit run app/streamlit_app.py
```

The app loads the locally generated model and dashboard artifacts from `outputs/`. Model artifacts are generated locally and are intentionally not committed.

### Optional: create a standalone Plotly dashboard

```bash
python src/03_plotly_dashboard.py \
  --data outputs/dashboard/powerbi_dashboard_sample.csv \
  --out outputs/figures/plotly_behavior_ml_dashboard.html
```

## Reference evaluation on the included preview

One local CPU run of the modular pipeline on the included 50,000-row preview (random seed 42; stratified 80/20 train/test split) produced:

| Metric | Reference result |
|---|---:|
| Accuracy | 0.794 |
| Balanced accuracy | 0.786 |
| Macro F1 | 0.618 |
| Weighted F1 | 0.814 |
| High-risk precision | 0.240 |
| High-risk recall | 0.846 |
| High-risk F1 | 0.373 |

These are reference results from one run on the preview dataset, **not** results from a full 1M-row GPU run. Because the High-risk class is relatively rare, precision and recall should be considered together. Re-run the pipeline to generate current metrics at `outputs/reports/model_metrics.json` and `outputs/reports/classification_report.csv` before quoting results elsewhere.

## Dashboard previews

**Power BI — executive overview**

![Power BI executive overview](reports/powerbi/Page1%20Executive%20Overview%20Burnout%20Risk%20Analysis.png)

**Power BI — predictive diagnostics and feature impact**

![Power BI predictive diagnostics](reports/powerbi/Page2%20Predictive%20Diagnostics%20%26%20Feature%20Impact.png)

These are screenshots, not editable Power BI workbooks. The repository does not include a `.pbix` file or a Tableau workbook.

## Reproducibility and interpretation notes

- The original Colab notebook is retained only to document the data-engineering/GPU-ingestion experiment. Its feature formulas differ from the modular pipeline; do not combine metrics or feature explanations across the two workflows.
- The modular training script uses an explicit feature allowlist and excludes outcome- and prediction-derived fields from the model inputs. This is the recommended training path for this repository.
- The CSV-ingestion benchmark is a single recorded run. For a rigorous performance comparison, repeat the test under controlled conditions and report a summary across runs.
- The dataset's label-generation process and external validity have not been independently established here. High predictive performance on this dataset would not, by itself, validate real-world student-risk assessment.
- The saved Power BI images are presentation snapshots. Regenerate model-dependent figures after changing the dataset, features, or training configuration.

## Project background and contributions

This repository is a portfolio-oriented organization of work from a DATA 230 team project at San José State University. The portfolio owner led substantial technical implementation across data engineering, modeling, interpretation, and visualization; the work originated in a collaborative course setting. Team credit and role descriptions should remain consistent with the [original course repository](https://github.com/xinghanchen-xc/DATA230-Final-Group-Project).

Developed for **DATA 230: Data Visualization**, San José State University.
