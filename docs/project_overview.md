# Project Overview

## Credit Risk Analysis and Prediction Using Machine Learning

This project develops a supervised machine learning solution to predict loan default risk using borrower, loan, and credit-history data.

The project is designed as a **credit-risk decision-support case study**. It combines machine learning with business-cost reasoning and responsible AI considerations.

---

## Business Objective

The business objective is to help a financial institution identify high-risk loan applicants before approval while avoiding unnecessary rejection of potentially good borrowers.

The main business question is:

> Can machine learning support credit-risk teams by predicting loan default probability and prioritizing applications for manual review?

---

## Machine Learning Objective

The machine learning objective is to predict the target variable:

| Target Variable | Meaning                     |
| --------------- | --------------------------- |
| `loan_status`   | Whether a borrower defaults |

Target classes:

| Value | Meaning    |
| ----: | ---------- |
|   `0` | No Default |
|   `1` | Default    |

This is a supervised binary classification problem.

---

## Dataset

The project uses the **Credit Risk Dataset** from Kaggle.

Dataset source:

```text
https://www.kaggle.com/datasets/laotse/credit-risk-dataset
```

The dataset contains borrower, loan, and credit-history variables, including:

* borrower age
* borrower income
* home ownership
* employment length
* loan intent
* loan grade
* loan amount
* interest rate
* loan percent income
* previous default history
* credit history length

The raw dataset is not included in this repository. The notebook downloads the dataset directly from Kaggle using `opendatasets`.

---

## Project Workflow

The project follows an end-to-end machine learning workflow:

1. Business problem framing
2. Dataset loading and understanding
3. Exploratory data analysis
4. Data quality assessment
5. Missing value treatment
6. Outlier policy and cleaning
7. Feature engineering and preprocessing
8. Logistic Regression baseline modeling
9. Random Forest modeling
10. Cross-validation
11. Hyperparameter tuning
12. Threshold analysis
13. Business cost simulation
14. Final model selection
15. Feature importance interpretation
16. Responsible AI and subgroup diagnostic
17. Final recommendations

---

## Main Models

The project compares:

| Model                  | Purpose                        |
| ---------------------- | ------------------------------ |
| Logistic Regression    | Interpretable baseline model   |
| Random Forest Baseline | Main non-linear model          |
| Tuned Random Forest    | Hyperparameter-optimized model |

Model evaluation includes:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC
* PR-AUC
* Confusion matrix
* False positives
* False negatives
* Business cost simulation

---

## Final Selected Model

The final selected model is:

| Item               | Result                                            |
| ------------------ | ------------------------------------------------- |
| Model              | Random Forest Baseline                            |
| Decision threshold | 0.30                                              |
| Selection reason   | Lowest simulated business cost                    |
| Intended use       | Decision support and manual review prioritization |

Final model performance:

| Metric          | Result |
| --------------- | -----: |
| Accuracy        | 0.8985 |
| Precision       | 0.7470 |
| Recall          | 0.7954 |
| F1-score        | 0.7704 |
| ROC-AUC         | 0.9305 |
| PR-AUC          | 0.8810 |
| False Positives |    383 |
| False Negatives |    291 |

---

## Business Cost Simulation

The project uses a simplified cost-sensitive evaluation.

Assumed costs:

| Error Type     |          Cost |
| -------------- | ------------: |
| False Negative | 10 cost units |
| False Positive |   1 cost unit |

The no-model baseline assumes approving all borrowers.

| Scenario              | Total Business Cost |
| --------------------- | ------------------: |
| Approve all borrowers |              14,220 |
| Final selected model  |               3,293 |

The final model reduces simulated business cost by:

* **10,927 cost units**
* **76.84% compared with approving all borrowers**

---

## Key Risk Drivers

The most important model features are:

1. `loan_percent_income`
2. `person_income`
3. `loan_int_rate`
4. `loan_grade`
5. `person_home_ownership`

These features align with credit-risk logic because they represent affordability, repayment capacity, institutional risk pricing, and borrower profile.

---

## Responsible AI Considerations

Credit-risk modeling is a high-impact use case because predictions may influence access to financial services.

This project includes responsible AI considerations such as:

* human-in-the-loop decision support
* avoiding fully automated rejection
* subgroup performance diagnostics
* discussion of proxy variables
* threshold impact review
* model limitation documentation
* recommendation for future fairness audits

The model should not be used for real-world lending decisions without further validation, compliance review, governance, fairness auditing, and monitoring.

---

## Conclusion

This project demonstrates that effective credit-risk modeling requires more than high accuracy.

A useful credit-risk model should combine:

* predictive performance
* threshold optimization
* business cost reasoning
* interpretability
* responsible AI safeguards

The final selected Random Forest model at threshold 0.30 provides the best business value under the chosen cost assumptions and is recommended as a decision-support model for manual credit-risk review.
