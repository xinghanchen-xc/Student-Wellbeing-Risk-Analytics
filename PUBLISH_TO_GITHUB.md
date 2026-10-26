# Publish the repaired project to GitHub

This archive is a source package. It does not contain your `.git` history or the full 1M-row dataset.

1. Download and extract this archive to a temporary folder.
2. Open your existing local clone of `Student-Wellbeing-Risk-Analytics` (the folder that contains the `.git` directory).
3. Copy the archive's project files into the local clone, replacing files with the same names. **Do not copy or replace the `.git` directory.**
4. Remove obsolete files from the old version if they remain:
   - `data/Burnout_Predictions_Preview_50k.csv`
   - `notebooks/task1_gpu_data_engineering.ipynb`
5. Do not copy the full Parquet/CSV dataset, model artifacts, or generated `outputs/` directory into the repository.
6. From PowerShell, inspect the change list before committing:

```powershell
git status
git add -A
git status
```

Check that `data/sample_50k.csv`, the shared feature module, the repaired scripts, tests, notebooks, and updated reports are included. Check that the old preview and notebook are deleted, and that no full data file, `.parquet`, `.joblib`, credential, or `outputs/` file is staged.

If `outputs/` or model artifacts were tracked by Git in an earlier commit, `.gitignore` alone will not untrack them. Remove tracked copies from the Git index while keeping local files with:

```powershell
git rm -r --cached outputs
```

Only run that command if `outputs/` is actually tracked. Repeat the check with `git status`.

Then commit and push:

```powershell
git commit -m "Unify CPU and GPU modeling pipeline"
git push origin main
```

If your local branch is not named `main`, use your actual branch name. If the remote repository has changes not present locally, inspect and pull/merge them before pushing; do not force-push as a shortcut.

After pushing, open the repository in a private/incognito browser window and follow the README Quick Start from a clean environment.
