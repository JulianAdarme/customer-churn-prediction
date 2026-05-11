# customer-churn-prediction
EN- Customer churn prediction using logistic regression with threshold tuning and class imbalance handling.
SP- Prediccón de abando de clientes usando regresión logistica con umbral de decisión y manejo de imbalance en datos.

## Overview

I built this project to explore how machine learning can be used to identify customers likely to leave a service.

One of the main challenges in this dataset was class imbalance, so I experimented with class weighting and different decision thresholds to improve churn detection instead of focusing only on overall accuracy.

---

## 📊 Dataset

IBM Telco Customer Churn dataset, including customer demographics, services, and account information.

The data set includes information about:

- Customers who left within the last month – the column is called Churn
- Services that each customer has signed up for – phone, multiple lines, internet, online security, online backup, device protection, tech support, and streaming TV and movies
- Customer account information – how long they’ve been a customer, contract, payment method, paperless billing, monthly charges, and total charges
- Demographic info about customers – gender, age range, and if they have partners and dependents

---

## Approach

- Cleaned and prepared customer data for modeling
- Encoded categorical variables
- Trained a Logistic Regression model
- Used class weighting to reduce bias toward non-churn customers
- Tested multiple decision thresholds to improve recall

---

## Results

Using a decision threshold of 0.4 produced the best balance between recall and overall model stability.

Main results:
- Recall: 79%
- ROC-AUC: 0.83

The model became significantly better at identifying customers likely to churn, even at the cost of lower precision.

---

## 🧠 Key Insight

- Customers with month-to-month contracts showed noticeably higher churn rates
- Customers using additional support services tended to stay longer
- Lowering the decision threshold improved churn detection considerably

---

## Conclusion

This project helped me understand the trade-off between precision and recall in imbalanced classification problems.

It also showed how threshold tuning can sometimes be more important than improving raw accuracy.

It prioritizes recall to reduce the number of missed churn cases, aligning with business impact.

The model can be further improved with additional data and features.
