# Customer Churn Prediction — Project Context

## Purpose
Learning project for predicting bank customer churn using Python and machine learning. The original work used a public Kaggle bank churn dataset saved locally as `BC.csv`.

This file preserves the useful technical and learning context from the original ChatGPT conversation so that the chat itself does not need to remain as the long-term archive.

## Environment
- Windows
- Visual Studio Code
- Python 3.12
- pandas
- matplotlib
- seaborn
- scikit-learn
- imbalanced-learn
- python-docx
- Project path used during the work: `C:\Users\deeb\kagga\BCCD`
- Virtual environment: `.venv`

## Dataset
The dataset contained 10,000 bank customers and 12 columns:

`customer_id, credit_score, country, gender, age, tenure, balance, products_number, credit_card, active_member, estimated_salary, churn`

Target:
- `churn = 0`: customer stayed
- `churn = 1`: customer left

Class distribution:
- Stayed: 79.63%
- Left: 20.37%
- Total churned customers: 2,037

This class imbalance became an important part of the project.

## Initial data loading

File used: `analys.py`

```python
import pandas as pd

data = pd.read_csv('BC.csv')
print(data.head())
```

The dataset loaded successfully.

## Exploratory Data Analysis
Charts were created with matplotlib and seaborn to compare churn behavior using:
- credit score
- country
- gender

Example structure:

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('BC.csv')

sns.set(style="whitegrid")

plt.figure(figsize=(10, 6))
sns.histplot(data=data, x='credit_score', hue='churn', kde=True)
plt.show()

plt.figure(figsize=(10, 6))
sns.countplot(data=data, x='country', hue='churn')
plt.show()

plt.figure(figsize=(10, 6))
sns.countplot(data=data, x='gender', hue='churn')
plt.show()
```

## Model 1 — Logistic Regression

File: `logistic_regression_churn.py`

Features:
- credit_score
- age
- tenure
- balance
- products_number
- credit_card
- active_member
- estimated_salary

Workflow:
1. Load BC.csv.
2. Separate X and y.
3. 80/20 train/test split.
4. StandardScaler.
5. LogisticRegression.
6. Evaluate predictions.

A column-name error was encountered because the first code used `num_of_products`, `has_cr_card`, and `is_active_member`. Printing `data.columns` revealed the correct names: `products_number`, `credit_card`, and `active_member`.

### Baseline Logistic Regression results
- Accuracy: 0.8095
- churn precision: 0.56
- churn recall: 0.15
- churn F1: 0.24
- Confusion matrix: [[1559, 48], [333, 60]]
- Recorded ROC-AUC: 0.5614

Important lesson: the 81% accuracy was misleading because the dataset was imbalanced. Only 60 of 393 churners in the test set were detected, while 333 were missed.

## Handling class imbalance — SMOTE

File: `SMOTE.PY`

SMOTE was applied only to the training data.

```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
```

### Logistic Regression + SMOTE
- Accuracy: 0.7030
- churn precision: 0.37
- churn recall: 0.71
- churn F1: 0.48
- Confusion matrix: [[1128, 479], [115, 278]]
- Recorded ROC-AUC: 0.7047

SMOTE lowered overall accuracy but greatly improved detection of customers who actually churned. Recall increased from 0.15 to 0.71.

Results were also exported to `churn_model_results.docx`.

## Logistic Regression hyperparameter search

GridSearchCV tested:
- C: 0.01, 0.1, 1, 10, 100
- solver: liblinear, lbfgs, newton-cg, sag, saga
- 5-fold cross-validation
- scoring: roc_auc

The tuned model produced essentially the same results as Logistic Regression + SMOTE, so tuning did not materially improve it.

## Model 2 — Random Forest

Random Forest was trained after SMOTE.

### Baseline Random Forest results
- Accuracy: 0.8215
- churn precision: 0.55
- churn recall: 0.54
- churn F1: 0.54
- Confusion matrix: [[1431, 176], [181, 212]]
- Recorded ROC-AUC: 0.7150

This was substantially more balanced than the original Logistic Regression model.

Results were exported to `churn_model_random_forest_results.docx`.

## Random Forest optimization

File: `random_forest_optimized.py`

Grid:

```python
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}
```

This creates 216 parameter combinations. With 5-fold cross-validation, GridSearchCV performs 1,080 Random Forest fits, explaining the very long runtime.

The full-data run took approximately 1 hour 35 minutes.

### Optimized Random Forest — full dataset
- Accuracy: 0.8310
- churn precision: 0.58
- churn recall: 0.52
- churn F1: 0.55
- Confusion matrix: [[1456, 151], [187, 206]]
- Recorded ROC-AUC: 0.7151

Compared with baseline Random Forest:
- Accuracy: 0.8215 → 0.8310
- churn precision: 0.55 → 0.58
- churn recall: 0.54 → 0.52
- churn F1: 0.54 → 0.55
- recorded AUC: 0.7150 → 0.7151

Optimization therefore mostly traded a little recall for precision/accuracy rather than producing a major improvement.

## Runtime and troubleshooting lessons

### Python / pip
At one point PowerShell did not recognize `pip`. Python itself was available using the full Python executable path. In future, prefer the active virtual environment and `python -m pip install ...`.

### GridSearch runtime
The large Random Forest grid was not frozen. Ctrl+C produced `KeyboardInterrupt` inside `grid_search.fit(...)`, confirming that it was still training.

Better future approaches:
- smaller parameter grid
- RandomizedSearchCV
- `n_jobs=-1`
- fewer CV folds when justified

### Word export
After the long optimized Random Forest run, saving:
`churn_model_random_forest_optimized_results.docx`
failed with:
`PermissionError: [Errno 13] Permission denied`

The model training had already completed; only the Word save failed. The likely cause was that the DOCX file was open/locked.

## 1,000-row sample experiment
The dataset was also split into ten sequential CSV files of 1,000 rows each.

One 1,000-client experiment produced:
- Accuracy: 0.8000
- churn precision: 0.55
- churn recall: 0.52
- churn F1: 0.53
- Confusion matrix: [[137, 19], [21, 23]]
- Recorded ROC-AUC: 0.7005

Important methodological correction for future work: a sample size of 1,000 does not automatically guarantee representativeness. Sequential slicing can introduce bias if the source data has ordering. Prefer randomized and, where appropriate, stratified sampling that preserves the churn distribution.

## Evaluation lessons
Metrics studied:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- ROC-AUC

For an imbalanced churn problem, accuracy alone should not determine model quality. The costs of false negatives and false positives matter.

For future versions, focus especially on:
- churn-class precision
- churn-class recall
- F1
- PR-AUC
- business cost of false positives vs false negatives
- probability threshold selection

### Methodological correction for future runs
The historical code calculated ROC-AUC using hard class predictions:
`roc_auc_score(y_test, y_pred)`.

A better calculation is based on predicted probabilities:

```python
y_prob = model.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_prob)
```

This correction should be used in future experiments. The historical AUC numbers above are retained only as a record of what was originally reported.

## Learning progression preserved from the chat
The project covered:
1. Obtaining a real dataset.
2. Loading CSV data with pandas.
3. Inspecting columns and churn distribution.
4. Filtering churned/staying customers.
5. Exploratory visualization.
6. Train/test splitting.
7. Scaling.
8. Logistic Regression.
9. Understanding class imbalance.
10. SMOTE.
11. Classification metrics.
12. GridSearchCV.
13. Random Forest.
14. Random Forest hyperparameter optimization.
15. Runtime/performance tradeoffs.
16. Exporting experiment results to Word.
17. Troubleshooting Python, packages, column names, runtime, and file locks.

## Working preference learned during this project
When continuing this project, code instructions should explicitly state whether the user should:
- create a new file,
- add code to an existing file, or
- replace the existing file,
and should always name the target file.

The project is intended as a learning exercise, so explanations should accompany the code rather than only supplying commands.

## Recommended next stage
Do not rerun the old experiment blindly. A future continuation should first establish a cleaner evaluation pipeline:
- stratified train/test split
- preprocessing pipeline
- SMOTE only inside training/CV workflow
- probability-based ROC-AUC
- PR-AUC
- threshold analysis
- reproducible model comparison
- optionally add categorical features such as country and gender using encoding

Then additional models can be compared under the same evaluation protocol.

---
Archived from the original Customer Churn Prediction ChatGPT learning conversation during chat cleanup on 2026-09-22.
