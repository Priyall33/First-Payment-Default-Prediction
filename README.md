# First Payment Default Prediction

Supervised binary classification project predicting whether a borrower will default on their first loan payment, using Logistic Regression, SVM, and XGBoost with class imbalance handling, hyperparameter tuning, threshold optimization, and SHAP explainability.

## Overview

First Payment Default (FPD) is one of the strongest early indicators of loan risk. This project builds a classification model to identify high-risk borrowers before they default, using a highly imbalanced dataset where only 2.89% of borrowers defaulted. The challenge is not just building a model — it's building one that can reliably detect the rare positive class without being overwhelmed by the majority.

## Dataset

- **Source:** Kaggle
- **Size:** 38,985 rows × 85 features
- **Target:** Binary — 1 = defaulted on first payment, 0 = did not default
- **Class imbalance:** 97.11% non-default vs 2.89% default

## Project Structure

├── First_payment_default.ipynb    
├── README.md                    

## Methodology

### 1. Exploratory Data Analysis
- Identified missing values across multiple features (27–40% missing in some columns)
- Dropped features with >50% missing values and removed the ID column
- Analyzed skewness across all features — several showed extreme positive skew (up to 124x)
- Computed Standardized Mean Difference (SMD) to identify features that differ most between defaulters and non-defaulters
- Computed VIF (Variance Inflation Factor) to detect multicollinearity
- Confirmed severe class imbalance (97% vs 3%) — accuracy alone is not a meaningful metric

### 2. Preprocessing
- **Imputation:** Median imputation for missing values (fit on training data only to prevent data leakage)
- **Scaling:** StandardScaler applied after imputation
- **Data split:** Stratified 70/15/15 train/validation/test split preserving class ratios across all sets
- **Feature selection:** L1 regularization (Lasso) reduced 81 features to 68 most informative ones, removing noise and improving generalization

### 3. Class Imbalance Handling
Two strategies compared on the same model:

| Strategy | PR-AUC |
|---|---|
| SMOTE (synthetic oversampling) | 0.0448 |
| Class weighting | 0.0377 |

SMOTE produced better PR-AUC and was used for final model training.

### 4. Models & Results

Three models trained and evaluated on the validation set:

| Model | ROC-AUC | PR-AUC | F1 |
|---|---|---|---|
| Logistic Regression | 0.664 | 0.070 | 0.085 |
| Linear SVM | 0.652 | 0.066 | 0.081 |
| XGBoost | 0.616 | 0.051 | 0.094 |

Each model includes a written interpretation of its strengths and weaknesses for this specific problem.

### 5. Hyperparameter Tuning
All three models tuned using **RandomizedSearchCV** with **StratifiedKFold (3-fold)** cross-validation, optimizing for F1 score:

- **Logistic Regression:** tuned `C` — best value `0.001`
- **Linear SVM:** tuned `C` — best value `0.01`
- **XGBoost:** tuned `n_estimators`, `max_depth`, `learning_rate` — best: 400 trees, depth 4, lr 0.05

### 6. Threshold Tuning
The default 0.5 classification threshold is suboptimal for imbalanced datasets. An F1-maximizing threshold search was applied across 0.05–0.95, selecting **threshold = 0.47**, which improved the final F1 score to **0.097**.

### 7. Model Explainability
Two levels of explainability implemented:

- **Feature Importance** — extracted directly from the XGBoost model to identify the top 10 most influential features
- **SHAP (SHapley Additive exPlanations)** — TreeExplainer applied to a 1,000-sample subset, producing a summary plot showing both feature importance and the direction of each feature's impact on predictions

### 8. Final Test Evaluation
The final model was evaluated on the held-out test set (never seen during training or tuning), with results reported for both the baseline and improved XGBoost model.

## Key Findings

- No single feature strongly predicts default — the predictive signal is distributed across many features
- Logistic Regression and SVM outperformed XGBoost on recall, catching more defaults
- XGBoost was more conservative — higher precision but missed more defaults
- Threshold tuning provided a meaningful improvement over the default 0.5 cutoff
- PR-AUC is the most appropriate metric given the severe class imbalance — accuracy is misleading here
- SHAP analysis revealed which features push predictions toward default vs non-default

## Tools & Libraries

- Python, Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn (LogisticRegression, LinearSVC, StandardScaler, SimpleImputer, RandomizedSearchCV, StratifiedKFold)
- XGBoost
- Imbalanced-learn (SMOTE)
- SHAP
- Statsmodels (VIF)

## How to Run

1. Clone the repository
2. Upload `kaggle_dataset.csv` to your Google Drive
3. Open `First_payment_default.ipynb` in Google Colab
4. Update the file path in the first cell to match your Drive location
5. Run all cells in order

> **Note:** Cell 56 references `best_xgb_improved` — ensure this variable is defined earlier in your session before running the final evaluation cell.
