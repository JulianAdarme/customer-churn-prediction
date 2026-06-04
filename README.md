# Customer Churn Prediction
**IBM Telco Dataset · Logistic Regression · scikit-learn**

---

## What this is

A telecom company loses revenue every time a customer cancels their subscription. The problem is that by the time they cancel, it's too late to do anything about it. This project builds a model to flag customers *before* they leave, so a retention team can act early.

The dataset is the IBM Telco Customer Churn dataset, which includes contract type, billing information, service usage, and payment method for ~7,000 customers.

---

## Results

| Metric | Value |
|--------|-------|
| ROC-AUC | 0.83 |
| Churn Recall (final) | 75–79% |
| Decision Threshold | 0.3 |

At the selected threshold, roughly 3 out of 4 customers who would have churned are correctly flagged before they leave.

---

## The approach

**1. EDA**
The three strongest signals for churn turned out to be contract type, monthly charges, and tech support. Month-to-month customers churn at ~43%, compared to 11% for one-year and 3% for two-year contracts. That gap alone drives most of the model's predictive power.

**2. Preprocessing**
Features were selected based on business relevance rather than throwing everything at the model. Demographic variables (gender, SeniorCitizen, Dependents) showed low correlation with churn and were dropped. Categorical variables were encoded using one-hot encoding for nominal features (InternetService, PaymentMethod) and ordinal encoding for contract type, which has a natural order.

**3. Handling class imbalance**
Churn represents ~26% of the dataset. A default logistic regression model ignores this and only achieves 47% recall on the churn class — meaning it misses more than half the customers who actually leave. Two adjustments were applied:
- `class_weight={0:1, 1:2}` to penalize missed churners more heavily during training
- Threshold tuning from 0.5 down to 0.3, evaluated across five candidate values

**4. Threshold selection**
The default 0.5 threshold is calibrated for balanced classes. At 0.3, churn recall reaches 75% with precision at 0.55 — meaning that for every 100 customers flagged as at-risk, about 55 will actually churn. In most retention contexts, the cost of missing a churner (lost revenue) exceeds the cost of a false alarm (an unnecessary discount or call), so this tradeoff is intentional.

---

## Key patterns

| Feature | What the data shows |
|---------|---------------------|
| Contract type | Month-to-month → 43% churn rate |
| Monthly charges | Churners have a higher median billing amount |
| Tech support | No tech support is associated with higher churn |
| Payment method | Electronic check users churn more often |

---

## Project structure

```
CustomerChurn.ipynb     # Full notebook: EDA → preprocessing → model → evaluation
Customer_Churn.csv      # IBM Telco dataset
```

---

## Limitations and next steps

- `TotalCharges` and some service features (`OnlineSecurity`, `OnlineBackup`) were excluded in this version. A follow-up should clean and retain them.
- Only logistic regression was tested. Tree-based models like XGBoost would likely improve recall without sacrificing as much precision, and handle non-linear feature interactions better.
- The model was trained on a static snapshot. In production, churn patterns shift over time and the model would need periodic retraining on recent data.
- A Streamlit interface would make the model usable by non-technical stakeholders for individual customer lookups.

---

## Stack

`pandas` · `scikit-learn` · `seaborn` · `matplotlib`
