# NIR Spectroscopy Moisture Prediction

Predicting biomass moisture content from Near-Infrared (NIR) spectra using chemometric and machine learning models.

## Overview

This project uses NIR spectroscopy data from biomass samples to predict moisture content. The challenge involves high-dimensional spectral data (1037 wavenumber features per sample), replicate scans per physical sample, and the need for appropriate preprocessing to extract the moisture signal from noisy spectra.

## Dataset

- **773 spectra** from **126 physical samples** (most samples had exactly 6 replicate scans)
- **1037 spectral features** (wavenumbers from ~4000 to ~12000 cm-1)
- **Target**: moisture content (range: 17.98% to 73.07%, mean: 45.92%)
- **No missing values** in the dataset
- 1 sample was labelled as "fault" in the sample ID

## Process

### 1. Exploratory Data Analysis
- Loaded the NIR dataset (1040 columns: Sample ID, scan number, 1037 wavenumbers, moisture)
- Plotted raw NIR spectra and confirmed the typical absorbance profile with visible variation between samples
- Checked the moisture distribution - fairly even spread across the full range, giving good target coverage
- Analyzed replicate structure: most samples had exactly 6 scans, 1 faulty sample had only 1 scan
- Plotted the **mean spectrum** and identified two key water-related absorption bands:
  - **O-H first overtone** (~7100 cm-1)
  - **O-H combination band** (~5200 cm-1)
- Since both bands sat inside the 4000-7500 cm-1 range, this region was selected for modeling (454 features after trimming)

### 2. Outlier Detection
- Standardized the spectra and ran **PCA** (10 components) to compress them into key dimensions
- Computed **Hotelling T2 scores** using the first 5 PCs to measure how far each spectrum sat from the center
- Set a 95% confidence cutoff using the F-distribution
- **39 spectra flagged as outliers** above the threshold
- Removed outliers + 1 fault-labelled sample
- **733 spectra from 122 physical samples** remained for analysis

### 3. External Holdout Split
- **Before any preprocessing or modeling**, set aside 13 physical samples as a completely external test set
- Train/validation: 651 spectra from 109 samples
- Holdout: 82 spectra from 13 samples

### 4. Preprocessing Comparison
Tested **12 combinations** of scatter correction and smoothing/derivative methods:

**Scatter correction** (corrects baseline shifts between samples):
- Raw (no correction)
- **SNV** (Standard Normal Variate) - centers and scales each spectrum individually
- **MSC** (Multiplicative Scatter Correction) - corrects against a reference spectrum

**Smoothing / Derivatives** (Savitzky-Golay filter, window=15, polyorder=3):
- None
- Smoothing (reduces noise without changing shape)
- 1st derivative (sharpens peaks, removes baseline offset)
- 2nd derivative (sharpens further but amplifies noise)

All preprocessing was applied only to the **4000-7500 cm-1 moisture-relevant region**.

SNV and MSC made spectra line up much better by removing baseline shifts. Derivatives sharpened peaks but the 2nd derivative also brought in more noise.

### 5. Modeling
All models used **nested grouped 5-fold cross-validation**:
- **Outer loop**: evaluates model performance
- **Inner loop**: tunes hyperparameters
- **Grouping**: ensures replicates of the same physical sample never appear in both train and test folds

**PLSR (Partial Least Squares Regression)**:
- Tested on all 12 preprocessing combinations
- Inner CV selected optimal number of components (1-20)
- **Best preprocessing: SNV + none** (no additional smoothing)

**SVR (Support Vector Regression)**:
- RBF kernel with StandardScaler
- Grid search over C (0.1, 1, 10, 100) and epsilon (0.1, 0.5)
- Applied only to the best preprocessing (SNV + none)

**ANN (Artificial Neural Network)**:
- sklearn MLPRegressor with early stopping
- Tested 3 architectures: (256,64), (128,64,32), (256,128,64)
- Applied only to the best preprocessing (SNV + none)

### 6. Holdout Evaluation
- Retrained each model on the full train/validation set with the best hyperparameters
- Evaluated on the 13 held-out physical samples (82 spectra)

### 7. Repeatability Analysis
- For each holdout sample, calculated the **standard deviation of replicate predictions**
- Low SD means the model gives consistent results regardless of which scan is used

### 8. Prediction Intervals
- Measured the spread of cross-validation residuals
- Built **95% prediction intervals** using 1.96 * residual SD
- Verified that most holdout measurements fell inside the band

## Results

### Cross-Validation Performance (best preprocessing: SNV + none)

| Model | RMSECV | R2 | RPD |
|-------|--------|-----|-----|
| **PLSR** | **2.63** | **0.965** | **5.35** |
| SVR | 3.06 | 0.953 | 4.60 |
| ANN | 3.93 | 0.922 | 3.58 |

### Holdout Performance

| Model | RMSE | R2 | RPD |
|-------|------|-----|-----|
| **PLSR** | **2.67** | **0.955** | **4.70** |
| SVR | 3.20 | 0.935 | 3.91 |
| ANN | 3.51 | 0.922 | 3.57 |

### Preprocessing Ranking (PLSR, all 12 combinations)

| Preprocessing | RMSECV | R2 |
|--------------|--------|-----|
| **SNV + none** | **2.63** | **0.965** |
| SNV + SG-smooth | 2.65 | 0.965 |
| MSC + none | 2.73 | 0.962 |
| MSC + SG-smooth | 2.75 | 0.962 |
| SNV + SG-d2 | 2.83 | 0.960 |
| SNV + SG-d1 | 2.87 | 0.959 |
| MSC + SG-d1 | 2.95 | 0.956 |
| MSC + SG-d2 | 2.95 | 0.956 |
| Raw + SG-smooth | 3.01 | 0.954 |
| Raw + none | 3.03 | 0.954 |
| Raw + SG-d2 | 3.19 | 0.949 |
| Raw + SG-d1 | 3.25 | 0.947 |

### Repeatability (within-sample SD on holdout)

| Model | Mean SD | Median SD | Max SD |
|-------|---------|-----------|--------|
| **PLSR** | **1.50** | **1.52** | **2.42** |
| SVR | 1.58 | 1.09 | 6.14 |
| ANN | 1.75 | 1.54 | 3.48 |

### Prediction Intervals

| Model | Residual SD | 95% PI half-width |
|-------|------------|-------------------|
| **PLSR** | **2.64** | **+/- 5.16** |
| SVR | 3.05 | +/- 5.97 |
| ANN | 3.93 | +/- 7.70 |

## Key Findings

- **PLSR was the best model overall** - lowest error, highest R2, best repeatability, and narrowest prediction intervals in both cross-validation and holdout
- **SNV preprocessing was the most effective** scatter correction. Additional smoothing or derivatives did not help and sometimes hurt performance
- **SVR did not improve over PLSR** despite being able to capture nonlinear patterns, suggesting the spectra-moisture relationship was already close to linear
- **ANN was the weakest model** - with only ~650 training spectra, there was not enough data for the network to learn anything beyond what PLSR already captured
- **Grouped cross-validation was essential** - without it, replicates of the same sample would leak into both train and test, giving artificially inflated results
- For PLSR, a prediction of 45% moisture means the true value is likely between ~40% and ~50% (95% PI of +/- 5.2 percentage points)

## Tech Stack

- Python, NumPy, Pandas, Matplotlib, SciPy
- scikit-learn (PLSRegression, SVR, MLPRegressor, PCA, GroupKFold, StandardScaler)

## How to Run

Open and run `NIR_Moisture_Prediction.ipynb` in Jupyter Notebook or Google Colab. The dataset is automatically downloaded from Google Drive.
