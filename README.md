# NIR Spectroscopy Moisture Prediction

Predicting biomass moisture content from Near-Infrared (NIR) spectra using chemometric and machine learning models.

## Overview

This project uses NIR spectroscopy data from 126 biomass samples (773 spectra with replicate scans) to predict moisture content. The spectra cover wavenumbers from ~4000 to ~12000 cm-1, with 1037 spectral features per sample. Moisture values range from 18% to 73%.

## Approach

1. **Exploratory Data Analysis** - Inspected spectral profiles, moisture distribution, and replicate structure
2. **Outlier Detection** - PCA-based Hotelling T2 analysis to identify and remove anomalous spectra (39 outliers + 1 faulty sample removed)
3. **Preprocessing Comparison** - Tested 12 combinations of scatter correction (Raw, SNV, MSC) and smoothing/derivatives (None, SG-smooth, SG-1st derivative, SG-2nd derivative)
4. **Modeling** - Compared three regression models with nested grouped 5-fold cross-validation:
   - **PLSR** (Partial Least Squares Regression)
   - **SVR** (Support Vector Regression with RBF kernel)
   - **ANN** (Multi-Layer Perceptron)
5. **External Validation** - Held out 13 physical samples for independent testing
6. **Repeatability & Prediction Intervals** - Assessed within-sample prediction stability and 95% prediction intervals

## Key Results

| Model | CV RMSE | CV R2 | Holdout RMSE | Holdout R2 |
|-------|---------|-------|--------------|------------|
| **PLSR** | **2.63** | **0.965** | **2.67** | **0.955** |
| SVR | 3.06 | 0.953 | 3.20 | 0.935 |
| ANN | 3.93 | 0.922 | 3.51 | 0.922 |

- **Best preprocessing**: SNV (Standard Normal Variate) without additional smoothing
- **Best model**: PLSR with 20 components, achieving the lowest error and highest repeatability
- **95% prediction interval**: +/- 5.2 percentage points for PLSR

## Tech Stack

- Python, NumPy, Pandas, Matplotlib, SciPy
- scikit-learn (PLSRegression, SVR, MLPRegressor, PCA, GroupKFold)

## How to Run

Open and run `NIR_Moisture_Prediction.ipynb` in Jupyter Notebook or Google Colab. The dataset is automatically downloaded from Google Drive.
