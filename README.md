# Customer Churn Prediction – English
**IBM Telco Dataset · Logistic Regression · scikit-learn**

---

## Table of Contents

- [What this is](#what-this-is)
- [Results](#results)
- [The Approach](#the-approach)
- [Key Patterns](#key-patterns)
- [Project Structure](#project-structure)
- [Limitations and Next Steps](#limitations-and-next-steps)
- [Stack](#stack)

---

## What this is

A telecom company loses revenue every time a customer cancels their subscription. The problem is that by the time they cancel, it's too late to do anything about it. This project builds a model to flag customers *before* they leave, so a retention team can act early.

The dataset is the IBM Telco Customer Churn dataset, which includes contract type, billing information, service usage, and payment method for ~7,000 customers.

## Results

| Metric | Value |
|--------|-------|
| ROC-AUC | 0.83 |
| Churn Recall (final model) | 84% |
| Churn Precision (final model) | 0.47 |
| Decision Threshold | 0.3 |

At the selected configuration — class weighting plus a 0.3 decision threshold — roughly 5 out of 6 customers who would have churned are correctly flagged before they leave, at the cost of a higher false-alarm rate: for every 100 customers flagged as at-risk, about 47 will actually churn.

## The Approach

**1. EDA**
The three strongest signals for churn turned out to be contract type, monthly charges, and tech support. Month-to-month customers churn far more often than customers on one- or two-year contracts. That gap alone drives most of the model's predictive power.

**2. Preprocessing**
Features were selected based on business relevance rather than throwing everything at the model. Demographic variables (gender, SeniorCitizen, Dependents) showed low correlation with churn and were dropped. Categorical variables were encoded using one-hot encoding for nominal features (InternetService, PaymentMethod) and ordinal encoding for contract type, which has a natural order.

**3. Handling class imbalance**
Churn represents ~26% of the dataset. A default logistic regression model ignores this and only achieves 47% recall on the churn class — meaning it misses more than half the customers who actually leave. Two adjustments were applied:
- `class_weight={0:1, 1:2}` to penalize missed churners more heavily during training
- Threshold tuning, evaluated across five candidate values (0.2, 0.3, 0.4, 0.55, 0.6)

**4. Threshold selection**
The default 0.5 threshold is calibrated for balanced classes. With `class_weight` applied and the threshold lowered to 0.3, churn recall reaches 84%, with precision at 0.47 — meaning that for every 100 customers flagged as at-risk, about 47 will actually churn. In most retention contexts, the cost of missing a churner (lost revenue) exceeds the cost of a false alarm (an unnecessary discount or call), so prioritizing recall over precision is intentional. This is a real trade-off, not a free improvement: nearly half of the flagged customers would not actually have churned, and a retention team acting on this list should budget for that.

## Key Patterns

| Feature | What the data shows |
|---------|---------------------|
| Contract type | Month-to-month customers churn at a much higher rate than one- or two-year contracts |
| Monthly charges | Churners have a higher median billing amount |
| Tech support | No tech support is associated with higher churn |
| Payment method | Electronic check users churn more often |

## Project Structure

```
CustomerChurn.ipynb     # Full notebook: EDA → preprocessing → model → evaluation
Customer_Churn.csv      # IBM Telco dataset
README.md
```

## Limitations and Next Steps

- `TotalCharges` and some service features (`OnlineSecurity`, `OnlineBackup`) were excluded in this version. A follow-up should clean and retain them.
- Only logistic regression was tested. Tree-based models like XGBoost would likely improve recall without sacrificing as much precision, and handle non-linear feature interactions better.
- The precision–recall trade-off at the chosen threshold is significant (0.47 precision) — before using this in production, it's worth testing intermediate thresholds between 0.3 and 0.4, or a cost-based threshold selection, to see whether a better recall/precision balance is achievable.
- The model was trained on a static snapshot. In production, churn patterns shift over time and the model would need periodic retraining on recent data.
- A Streamlit interface would make the model usable by non-technical stakeholders for individual customer lookups.

## Stack

`pandas` · `scikit-learn` · `seaborn` · `matplotlib`

---
---

# Predicción de Deserción de Clientes (Customer Churn) – Español
**IBM Telco Dataset · Regresión Logística · scikit-learn**

---

## Tabla de Contenidos

- [De qué se trata](#de-qué-se-trata)
- [Resultados](#resultados)
- [El enfoque](#el-enfoque)
- [Patrones clave](#patrones-clave)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Limitaciones y próximos pasos](#limitaciones-y-próximos-pasos)
- [Stack](#stack-1)

---

## De qué se trata

Una empresa de telecomunicaciones pierde ingresos cada vez que un cliente cancela su suscripción. El problema es que, para cuando cancela, ya es tarde para hacer algo al respecto. Este proyecto construye un modelo para identificar a los clientes en riesgo *antes* de que se vayan, de modo que un equipo de retención pueda actuar a tiempo.

El dataset utilizado es el IBM Telco Customer Churn, que incluye tipo de contrato, información de facturación, uso de servicios y método de pago para ~7,000 clientes.

## Resultados

| Métrica | Valor |
|--------|-------|
| ROC-AUC | 0.83 |
| Recall de deserción (modelo final) | 84% |
| Precisión de deserción (modelo final) | 0.47 |
| Umbral de decisión | 0.3 |

Con la configuración seleccionada —ponderación de clases más un umbral de decisión de 0.3— se identifica correctamente a aproximadamente 5 de cada 6 clientes que efectivamente habrían desertado, a costa de una tasa de falsas alarmas más alta: de cada 100 clientes marcados como en riesgo, cerca de 47 desertarán realmente.

## El enfoque

**1. EDA**
Las tres señales más fuertes de deserción resultaron ser el tipo de contrato, el monto mensual facturado y el soporte técnico. Los clientes con contrato mes a mes desertan con mucha más frecuencia que quienes tienen contratos de uno o dos años. Esa diferencia por sí sola explica la mayor parte del poder predictivo del modelo.

**2. Preprocesamiento**
Las variables se seleccionaron según su relevancia de negocio, en lugar de incluir todas las disponibles. Las variables demográficas (género, `SeniorCitizen`, `Dependents`) mostraron baja correlación con la deserción y fueron descartadas. Las variables categóricas se codificaron con one-hot encoding para variables nominales (`InternetService`, `PaymentMethod`) y con codificación ordinal para el tipo de contrato, que sí tiene un orden natural.

**3. Manejo del desbalance de clases**
La deserción representa ~26% del dataset. Un modelo de regresión logística por defecto ignora este desbalance y solo alcanza un 47% de recall en la clase de deserción, es decir, no detecta a más de la mitad de los clientes que efectivamente se van. Se aplicaron dos ajustes:
- `class_weight={0:1, 1:2}`, para penalizar más fuertemente los casos de deserción no detectados durante el entrenamiento
- Ajuste del umbral de decisión, evaluado en cinco valores candidatos (0.2, 0.3, 0.4, 0.55 y 0.6)

**4. Selección del umbral**
El umbral por defecto de 0.5 está calibrado para clases balanceadas. Al aplicar `class_weight` y bajar el umbral a 0.3, el recall de deserción llega a 84%, con una precisión de 0.47 — es decir, de cada 100 clientes marcados como en riesgo, unos 47 desertarán realmente. En la mayoría de los contextos de retención, el costo de no detectar a un cliente que se va (ingresos perdidos) supera al costo de una falsa alarma (una llamada o descuento innecesario), por lo que priorizar el recall sobre la precisión es una decisión intencional. Este es un trade-off real, no una mejora gratuita: casi la mitad de los clientes marcados no habría desertado en realidad, y un equipo de retención que actúe sobre esta lista debe contemplar ese costo.

## Patrones clave

| Variable | Qué muestran los datos |
|---------|---------------------|
| Tipo de contrato | Los clientes con contrato mes a mes desertan a una tasa mucho más alta que los de contratos de uno o dos años |
| Monto mensual | Los clientes que desertan tienen una mediana de facturación más alta |
| Soporte técnico | La ausencia de soporte técnico está asociada a una mayor deserción |
| Método de pago | Los usuarios que pagan con cheque electrónico desertan con más frecuencia |

## Estructura del Proyecto

```
CustomerChurn.ipynb     # Notebook completo: EDA → preprocesamiento → modelo → evaluación
Customer_Churn.csv      # Dataset IBM Telco
README.md
```

## Limitaciones y próximos pasos

- `TotalCharges` y algunas variables de servicio (`OnlineSecurity`, `OnlineBackup`) fueron excluidas en esta versión. Una siguiente iteración debería limpiarlas y conservarlas.
- Solo se evaluó regresión logística. Modelos basados en árboles como XGBoost probablemente mejorarían el recall sin sacrificar tanto la precisión, y capturarían mejor las interacciones no lineales entre variables.
- El trade-off entre precisión y recall en el umbral elegido es considerable (precisión de 0.47) — antes de usar este modelo en producción, vale la pena probar umbrales intermedios entre 0.3 y 0.4, o una selección de umbral basada en costos, para ver si es posible lograr un mejor balance.
- El modelo se entrenó sobre una foto fija de los datos. En producción, los patrones de deserción cambian con el tiempo y el modelo necesitaría reentrenamiento periódico.
- Una interfaz en Streamlit permitiría que usuarios no técnicos consulten el riesgo de deserción de clientes individuales de forma interactiva.

## Stack

`pandas` · `scikit-learn` · `seaborn` · `matplotlib`
