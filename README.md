# customer-churn-prediction
EN- Customer churn prediction using logistic regression with threshold tuning and class imbalance handling.
SP- Prediccón de abando de clientes usando regresión logistica con umbral de decisión y manejo de imbalance en datos.

## 📌 Overview

This project focuses on predicting customer churn using machine learning techniques. The goal is to identify customers at risk of leaving in order to support retention strategies.

---

## 📊 Dataset

IBM Telco Customer Churn dataset, including customer demographics, services, and account information.

The data set includes information about:

- Customers who left within the last month – the column is called Churn
- Services that each customer has signed up for – phone, multiple lines, internet, online security, online backup, device protection, tech support, and streaming TV and movies
- Customer account information – how long they’ve been a customer, contract, payment method, paperless billing, monthly charges, and total charges
- Demographic info about customers – gender, age range, and if they have partners and dependents

---

## ⚙️ Approach

- Data cleaning and preprocessing  
- Feature encoding  
- Logistic Regression model  
- Handling class imbalance with `class_weight`  
- Threshold tuning to improve recall  

---

## 📈 Results

- ROC-AUC: ~0.83  
- Improved churn detection through threshold adjustment  
- Balanced trade-off between precision and recall  

---

## 🧠 Key Insight

- Customers with month-to-month contracts show higher churn rates
- Higher monthly charges are associated with increased churn risk
- Lack of additional services (e.g., tech support) may increase churn probability

---

## 🚀 Conclusion

The model can be used as a base for implementing customer retention strategies focused on high-risk users.

It prioritizes recall to reduce the number of missed churn cases, aligning with business impact.

The model can be further improved with additional data and features.
