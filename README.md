# 🚌 Public Bus Operations & Delay Prediction

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine_Learning-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data_Wrangling-150458?style=flat&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Status](https://img.shields.io/badge/Project_Status-Completed-success)](#)

An end-to-end Machine Learning pipeline designed to optimize urban transit efficiency by classifying trip delay severity and accurately regressing total bus trip durations. Developed as a capstone project during the **EYouth / DECI Data Program**.

---

## 📌 Executive Summary

Public transportation systems often suffer from unpredictable transit times. This project delivers a double-headed machine learning framework:
1. **Classification Task:** Predicts delay severity levels to alert transit dispatchers.
2. **Regression Task:** Estimates exact total trip duration for enhanced route planning.

The solution emphasizes domain-specific risk mitigation—specifically reducing false negatives for severe delay incidents to prevent system bottlenecks.

---

## 📊 Dataset & Preprocessing Pipeline

The model was built on a dataset of **10,000 operational bus trip records**.

* **Data Cleaning & Quality:**
  * Handled missing values using context-aware Imputation strategies (Median for continuous skew distributions, Mode for categorical variables).
  * Identified and removed duplicate trip records and extreme sensor outliers.
* **Feature Engineering & Transformation:**
  * Encoded temporal features (peak/off-peak hours, day-of-week).
  * Standardized continuous numerical variables using `StandardScaler`.
  * One-Hot Encoded categorical variables for line IDs and weather conditions.

---

## ⚙️ Model Architecture & Key Results

### 1. Delay Severity Classification
* **Target:** Categorical Delay Severity Levels (`On Time`, `Moderate Delay`, `Severe Delay`).
* **Selected Model:** Support Vector Machine (SVM) classifier tuned with hyperparameter search.
* **Evaluation Metrics:**
  * **F1-Macro Score:** `0.715`
  * **Critical Risk Reduction:** Reduced the rate of misclassifying **Severe Delays** as *On Time* to under **0.9%**, prioritizing operational safety and early warning.

### 2. Trip Duration Regression
* **Target:** Total Trip Duration in minutes.
* **Selected Models:** Linear Regression & K-Nearest Neighbors (KNN) Regressor.
* **Evaluation Metrics:**
  * **Variance Explained ($R^2$):** Reached **`0.924`** on held-out test data.
  * Demonstrated low Root Mean Squared Error (RMSE) across diverse traffic scenarios.

---

## 🛠️ Tech Stack & Tooling

* **Core Programming:** Python
* **Data Processing & Analytics:** Pandas, NumPy
* **Machine Learning Framework:** Scikit-Learn
* **Model Persistence:** Joblib (`.joblib` artifact exports)
* **Visualization:** Matplotlib, Seaborn

---

## 📁 Repository Structure

```text
.
├── bus_operations_project.ipynb     # Jupyter Notebook with EDA, modeling & evaluation
├── best_bus_delay_classifier.joblib  # Trained SVM Classification Model
├── best_bus_duration_regressor.joblib # Trained Duration Regression Model
├── EYOUTH_bus_operations_report.pdf  # Comprehensive technical & business report
└── README.md                        # Documentation
