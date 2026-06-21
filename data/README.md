# Data Folder

This folder documents the dataset used in the project.

## Dataset Source

This project uses the **Credit Risk Dataset** from Kaggle.

Dataset URL:

```text
https://www.kaggle.com/datasets/laotse/credit-risk-dataset
```

The dataset contains borrower, loan, and credit-history information used to predict whether a borrower is likely to default.

---

## Important Note About Raw Data

The raw dataset file is **not included** in this repository.

Reasons:

* The dataset is hosted externally on Kaggle.
* Dataset licensing and redistribution rules should be respected.
* Keeping raw data out of the repository makes the project cleaner and lighter.
* The notebook downloads the dataset directly from Kaggle when needed.

---

## Expected Dataset File

The notebook expects the following file after download:

```text
credit-risk-dataset/credit_risk_dataset.csv
```

The notebook downloads the dataset using the `opendatasets` library.

---

## How the Dataset Is Loaded

The project notebook uses this Kaggle dataset URL:

```python
DATASET_URL = "https://www.kaggle.com/datasets/laotse/credit-risk-dataset"
```

After downloading, the data is loaded into pandas for analysis and modeling.

---

## Dataset Structure

Each row represents a loan applicant or loan application.

The main feature groups are:

| Feature Group            | Example Columns                                                                  |
| ------------------------ | -------------------------------------------------------------------------------- |
| Borrower characteristics | `person_age`, `person_income`, `person_emp_length`, `person_home_ownership`      |
| Loan characteristics     | `loan_amnt`, `loan_int_rate`, `loan_intent`, `loan_grade`, `loan_percent_income` |
| Credit history           | `cb_person_default_on_file`, `cb_person_cred_hist_length`                        |
| Target variable          | `loan_status`                                                                    |

---

## Target Variable

The target variable is:

```text
loan_status
```

Target definition:

| Value | Meaning    |
| ----: | ---------- |
|   `0` | No Default |
|   `1` | Default    |

This makes the project a supervised binary classification problem.

---

## Data Quality Notes

The raw dataset contains several issues that are handled in the notebook:

* Missing values in `person_emp_length`
* Missing values in `loan_int_rate`
* Unrealistic or extreme values in variables such as age, employment length, and income
* Class imbalance between default and non-default cases

The notebook applies a transparent preprocessing strategy, including:

* Missing-value indicators
* Median imputation
* Loan-grade-based interest-rate imputation
* Outlier treatment
* Income capping
* Recalculation of `loan_percent_income`

---

## Reproducibility

To reproduce the project:

1. Open the notebook:

```text
notebooks/credit_risk_analysis_and_prediction.ipynb
```

2. Run all cells.

3. Authenticate with Kaggle if requested.

4. The notebook will download and load the dataset automatically.

---

## Disclaimer

The dataset is used for educational and portfolio purposes only.

The model developed in this project should not be used for real-world lending decisions without additional validation, governance, fairness auditing, compliance review, and monitoring.
