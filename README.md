# Diabetes Prediction Using SVM and Random Forest

This project uses supervised machine learning to predict whether a patient has diabetes. The analysis is presented in [`prediction.ipynb`](prediction.ipynb) and compares Support Vector Machine (SVM) models with a Random Forest classifier.

## Project Overview

The notebook analyzes a dataset containing 768 patient records, eight medical features, and a binary `Outcome` target:

- `0`: No diabetes
- `1`: Diabetes

The main goals are to:

1. Explore the dataset and identify data-quality issues.
2. Replace invalid zero values with median values.
3. Remove selected extreme outliers.
4. Train and evaluate SVM and Random Forest models.
5. Tune model hyperparameters using `GridSearchCV`.
6. Compare the models using classification metrics and evaluation curves.

## Dataset Features

| Feature | Description |
| --- | --- |
| `Pregnancies` | Number of pregnancies |
| `Glucose` | Glucose concentration |
| `BloodPressure` | Diastolic blood pressure |
| `SkinThickness` | Triceps skin-fold thickness |
| `Insulin` | Two-hour serum insulin level |
| `BMI` | Body mass index |
| `DiabetesPedigreeFunction` | Hereditary diabetes-risk score |
| `Age` | Patient age |
| `Outcome` | Diabetes classification target |

The notebook first checks for a local dataset at the path defined in the code. If it is not found, it loads the CSV from the configured GitHub URL.

## Methodology

- Exploratory data analysis using distributions, box plots, class counts, and correlation heatmaps.
- Invalid zero values in `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, and `BMI` are treated as missing values and replaced with column medians.
- Selected high-value outliers are removed using thresholds defined in the notebook.
- The data is split into training and testing sets using an 80:20 stratified split with `random_state=42`.
- SVM features are standardized with `StandardScaler`.
- Class imbalance is addressed with `class_weight='balanced'`.
- Hyperparameters are selected with 10-fold `GridSearchCV`.

## Models

### Support Vector Machine

The notebook evaluates two kernels:

- RBF kernel
- Polynomial kernel

The RBF kernel performs better than the polynomial kernel, particularly for recall on the diabetes class.

### Random Forest

The Random Forest model is tuned using:

- `n_estimators`
- `max_depth`
- `min_samples_split`
- `min_samples_leaf`
- `max_features`

## Results

The reported results are based on the notebook's fixed train-test split and may vary if the data or split changes.

| Model | Test Accuracy | Diabetes Recall | F1-Score |
| --- | ---: | ---: | ---: |
| SVM with tuned RBF kernel | approximately 77.6% | approximately 75% | approximately 70% |
| **Tuned Random Forest** | **approximately 78.9%** | **approximately 79%** | **approximately 72%** |

The tuned Random Forest is selected as the best model because it provides the strongest balance between precision and recall and detects more diabetic cases in the reported test results. `Glucose` is identified as the most influential feature.

## Requirements

Python 3.9 or newer is recommended. Install the required packages with:

```bash
pip install jupyter pandas numpy matplotlib seaborn scikit-learn
```

## Running the Notebook

1. Open the project folder in VS Code or Jupyter.
2. Open `prediction.ipynb`.
3. Select a Python kernel with the required packages installed.
4. Run the cells from top to bottom.

An internet connection is required when the local CSV file is unavailable because the notebook then loads the dataset from GitHub.

## Files

- [`prediction.ipynb`](prediction.ipynb): EDA, preprocessing, model training, tuning, and evaluation.
- [`prediction.pdf`](prediction.pdf): PDF export of the analysis.

## Limitations

This is an educational machine learning project, not a medical diagnostic system. The dataset is relatively small, and the results have not been externally validated. Predictions should not be used for clinical decisions without appropriate medical review, independent validation, and regulatory approval.

## Future Work

- Validate the models on an independent dataset.
- Compare additional algorithms such as XGBoost.
- Add feature engineering and calibration analysis.
- Evaluate threshold selection with a focus on minimizing false negatives.