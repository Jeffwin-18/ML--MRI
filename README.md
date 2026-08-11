# Automated AI-Based Quality Control Dashboard for MRI Brain Scans

## Overview

The **Automated AI-Based Quality Control Dashboard for MRI Brain Scans** is an AI-powered system developed to automatically assess the quality of brain MRI scans. The project detects imaging artifacts, classifies scan quality, and generates an interpretable Quality Control (QC) score using a machine learning model.

The system combines MRI image processing, feature extraction, synthetic artifact generation, and an interactive Streamlit dashboard to assist researchers and healthcare professionals in identifying poor-quality scans before further analysis.

---

## Features

* Automated MRI quality assessment
* Synthetic artifact generation for model training
* XGBoost-based artifact classification
* Quality Control (QC) score generation
* PASS / REVIEW / FAIL scan categorization
* Interactive Streamlit dashboard
* Kaggle dataset integration
* Local MRI folder support
* Batch MRI processing
* Downloadable QC reports
* Interactive analytics and visualizations

---

## Project Workflow

```
MRI Dataset
      │
      ▼
Data Collection
      │
      ▼
Preprocessing
      │
      ▼
Feature Extraction
      │
      ▼
Synthetic Artifact Generation
      │
      ▼
Dataset Organization
      │
      ▼
XGBoost Model Training
      │
      ▼
Model Evaluation
      │
      ▼
Streamlit Dashboard
      │
      ▼
QC Score & Artifact Prediction
```

---

## Dataset

### OASIS Dataset

Used for training the machine learning model.

* Original MRI scans: **148**
* Generated dataset: **1,924 MRI images**
* Dataset split:

  * Training: **80%**
  * Testing: **20%**

### IXI Dataset

* Source: Kaggle
* Total MRI scans: **1,231**
* Used for external model validation and testing.

---

## Synthetic Artifact Generation

Synthetic MRI artifacts were generated using heuristic-based image processing techniques.

Artifacts include:

* Original
* Blur
* Noise
* Motion

Image processing methods:

* Sobel Operator
* Laplacian Filter
* Fast Fourier Transform (FFT)

---

## Feature Extraction

The final XGBoost model uses six image quality features:

* Signal-to-Noise Ratio (SNR)
* Entropy
* Laplacian Variance
* GLCM Contrast
* GLCM Energy
* GLCM Homogeneity

> **Note:** Mean, Variance, and Standard Deviation were excluded because they provide information already represented through SNR. The Bias artifact class was merged with the Original class due to overlapping feature characteristics.

---

## Machine Learning Model

**Algorithm:** XGBoost Classifier

### Training

* Dataset: OASIS
* Four artifact classes:

  * Original
  * Blur
  * Motion
  * Noise

### Testing

* OASIS test split
* IXI dataset

### Performance

* Accuracy: **99.5%**

---

## Dashboard Features

The Streamlit dashboard provides:

* MRI scan upload
* Kaggle dataset download
* Local folder processing
* Batch scan analysis
* Artifact prediction
* QC score generation
* PASS / REVIEW / FAIL classification
* Feature visualization
* Analytics dashboard
* Downloadable QC reports

---

## Technology Stack

### Frontend & Backend

* Streamlit

### Machine Learning

* XGBoost

### Programming Language

* Python

### Libraries

* NumPy
* Pandas
* NiBabel
* Plotly
* Scikit-learn
* Joblib

---

## Project Structure

```
├── app.py
├── feature_extraction.py
├── saved_models/
│   └── xgb_artifact_classifier.joblib
├── final_artifact_features.csv
├── requirements.txt
├── README.md
└── assets/
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/your-repository.git
cd your-repository
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

---

## Using the Dashboard

The application supports three input methods:

* Upload a ZIP file containing MRI scans
* Select a local MRI dataset folder
* Download a dataset directly from Kaggle

After processing, the dashboard displays:

* Predicted artifact class
* Prediction confidence
* QC score
* PASS / REVIEW / FAIL status
* Interactive analytics

---

## Limitations

* Bias-field artifacts were excluded due to overlap with the Original class.
* Synthetic artifacts may not fully represent real clinical imaging conditions.
* Model performance depends on dataset diversity.
* Additional validation on larger clinical datasets is recommended.

---

## Future Enhancements

* Expand training with larger MRI datasets.
* Improve synthetic artifact simulation.
* Explore deep learning-based quality assessment.
* Integrate with hospital imaging workflows.
* Generate explainable AI-based QC reports.

---

## References

1. Marcus, D. S., et al. *Open Access Series of Imaging Studies (OASIS).* Journal of Cognitive Neuroscience, 2007.

2. Chen, T., & Guestrin, C. *XGBoost: A Scalable Tree Boosting System.* KDD, 2016.

3. Pérez-García, F., Sparks, R., & Ourselin, S. *TorchIO: A Python Library for Medical Image Processing.* Computer Methods and Programs in Biomedicine, 2021.

4. IXI Dataset (Kaggle)

---

---

## License

This project was developed for academic and research purposes.

# MRI QC Dashboard

A Streamlit dashboard for batch quality-control of MRI scans, built around
the feature/artifact pipeline from `Model1.ipynb`.

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL Streamlit prints (usually `http://localhost:8501`).


