# Data

`Burnout_Predictions_Preview_50k.csv` is the repository's 50,000-row preview file. It is included to make the modular CPU workflow testable without downloading the full source dataset.

The full dataset is not included. The original course notebook identifies the Kaggle dataset as `sharmajicoder/student-mental-health-and-burnout`; check the dataset's current terms, provenance, and label-generation method before reuse.

The preview includes target/outcome and precomputed prediction columns for analysis. The model pipeline uses an explicit feature allowlist and excludes direct outcome and prior-prediction columns from model inputs. Do not use this project for diagnosis or real student-support decisions.
