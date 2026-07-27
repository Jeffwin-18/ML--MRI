# MRI QC Dashboard

A Streamlit dashboard for batch quality-control of MRI scans, built around
the feature/artifact pipeline from `Model1.ipynb`.

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL Streamlit prints (usually `http://localhost:8501`).

## What's included

- **Data loading** — upload any CSV with the QC feature columns, or the
  bundled `final_artifact_features.csv` loads automatically as a demo.
- **XGBoost artifact classifier** — trained live (cached) on the loaded
  reference data using the same 6 features and hyperparameters as the
  notebook (`SNR`, `Entropy`, `LaplacianVariance`, `GLCMContrast`,
  `GLCMEnergy`, `GLCMHomogeneity` → 5-class artifact prediction).
- **QC Score (0–100) & PASS/REVIEW/FAIL status** — computed as the
  probability-weighted "quality" across all predicted classes (not just
  the top class), so a scan the model is unsure about lands between
  categories rather than snapping hard to one label. Class weights and
  PASS/REVIEW thresholds are adjustable from the sidebar.
- **Batch Scorecard** — KPI summary, color-coded results table, confusion
  matrix on the hold-out split, and a CSV download of the full QC report.
- **Analytics** — artifact distribution, PASS/REVIEW/FAIL split, per-feature
  box plots, average feature profile bars, and a normalized radar chart
  comparing predicted classes.
- **Scan Drill-down** — pick any single scan to see its class probabilities,
  a radar of its feature profile against the dataset average, and raw
  feature values.
- **Roadmap page** — placeholders describing the planned NIfTI viewer,
  skull-strip overlay, and ABIDE site analysis, plus notes on how to wire
  them in using the `extract_features_from_volume` pipeline already written
  in the notebook.

## Notes / assumptions

- The notebook's own `quality_map` only covered `original/blur/noise/motion`
  and didn't handle the `bias` class (a gap in `Model1.ipynb`). This app
  replaces that hard map with configurable class-quality weights that cover
  all 5 classes, so nothing crashes on a `bias` prediction.
- Any CSV you upload for **scoring** must contain the 6 model feature columns.
  The **reference/training** data (used to fit the model each session) is
  the same loaded file, and must also include an `Artifact` label column —
  swap in your own labeled dataset to retrain on different data.
- `MRI_ID` repeats across synthetic artifact/level variants in the sample
  data, so the dashboard builds a unique `Scan_ID` (`MRI_ID | Artifact | Level`)
  for drill-down selection.
