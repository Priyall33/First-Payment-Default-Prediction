# First-Payment-Default-Prediction

Supervised binary classification project predicting whether a borrower will default on their first loan payment, using Logistic Regression, SVM, and XGBoost with class imbalance handling and model explainability.

## Overview

First Payment Default (FPD) is one of the strongest early indicators of loan risk. This project builds a classification model to identify high-risk borrowers before they default, using a highly imbalanced dataset where only 2.89% of borrowers defaulted. The challenge is not just building a model — it's building one that can reliably detect the rare positive class without being overwhelmed by the majority.

## Dataset

- **Source:** Kaggle
- **Size:** 38,985 rows × 85 features
- **Target:** Binary — 1 = defaulted on first payment, 0 = did not default
- **Class imbalance:** 97.11% non-default vs 2.89% default

## Project Structure

├── Priyal_Vyas.ipynb   
├── README.md            

## Methodology

### 1. Exploratory Data Analysis
- Identified missing values across multiple features (27–40% missing in some columns)
- Dropped features with >50% missing values and the ID column
- Analyzed skewness — several features showed extreme positive skew (up to 124x)
- Confirmed severe class imbalance (97% vs 3%)
- Calculated correlations with target — no single feature strongly predicts default alone

### 2. Preprocessing
- **Imputation:** Median imputation for missing values (fit on training data only to prevent leakage)
- **Scaling:** StandardScaler applied after imputation
- **Data split:** Stratified 70/15/15 train/validation/test split to preserve class ratios
- **Feature selection:** L1 regularization (Lasso) reduced 81 features to 68 most informative ones
- **VIF analysis:** Identified multicollinear features for awareness

### 3. Class Imbalance Handling
Two strategies compared:
- **SMOTE** — synthetic oversampling of minority class
- **Class weighting** — penalizing misclassification of minority class more heavily

| Strategy | PR-AUC |
|---|---|
| SMOTE | 0.0448 |
| Class Weighting | 0.0377 |

SMOTE produced better PR-AUC and was selected for final modeling.

### 4. Models & Results

| Model | ROC-AUC | PR-AUC | F1 |
|---|---|---|---|
| Logistic Regression | 0.664 | 0.070 | 0.085 |
| Linear SVM | 0.652 | 0.066 | 0.081 |
| XGBoost | 0.616 | 0.051 | 0.094 |

All three models were tuned using RandomizedSearchCV with StratifiedKFold cross-validation.

### 5. Threshold Tuning
Default classification threshold (0.5) is suboptimal for imbalanced data. An F1-maximizing threshold search was applied across the range 0.05–0.95, selecting **threshold = 0.47** which improved F1 to **0.097**.

### 6. Model Explainability
Feature importance extracted for the final XGBoost model, identifying the top 10 most influential features driving default predictions.

## Key Findings

- No single feature strongly predicts default — the signal is distributed across many features
- Logistic Regression and SVM outperformed XGBoost on recall, catching more defaults
- XGBoost was more conservative — higher precision but lower recall
- Threshold tuning provided a meaningful improvement over the default 0.5 cutoff
- PR-AUC is the most appropriate metric given the severe class imbalance

## Tools & Libraries

- Python, Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn (LogisticRegression, LinearSVC, StandardScaler, SimpleImputer, SelectKBest, RandomizedSearchCV, StratifiedKFold)
- XGBoost
- Imbalanced-learn (SMOTE, RandomOverSampler, RandomUnderSampler)
- SHAP

## How to Run

1. Clone the repository
2. Upload `kaggle_dataset.csv` to your Google Drive
3. Open the notebook in Google Colab
4. Update the file path in the first cell to match your Drive location
5. Run all cells in order
