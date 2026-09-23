# 📊 XGBoost Customer Churn Prediction & Tuning Pipeline

An end-to-end Machine Learning classification project designed to predict customer churn risk using an optimized **XGBoost Classifier** with **5-Fold GridSearchCV**, complete with an interactive **Streamlit** web application.

---

## 📌 Project Overview
Customer retention is critical for subscription and banking services. This project implements a high-performance machine learning pipeline to identify customers at risk of churning, enabling proactive retention strategies.

### 🌟 Key Highlights
* **Optimal Hyperparameter Tuning:** Used 5-Fold Cross-Validation (`GridSearchCV`) across 16 parameter combinations to select the best tree depth and learning rate.
* **Overfitting Prevention:** Fine-tuned `subsample` and `max_depth` to ensure model generalization on unseen data.
* **Production Deployment:** Built and tested a real-time web application using **Streamlit** for interactive inference.

---

## 📈 Model Performance & Metrics

| Metric | Score |
| :--- | :--- |
| **Accuracy** | **98.0%** |
| **ROC-AUC** | **0.9980** |
| **Precision (Churn Class)** | High |
| **Recall (Churn Class)** | High |

### Best Hyperparameters Found:
* `n_estimators`: 100
* `max_depth`: 3
* `learning_rate`: 0.1
* `subsample`: 0.8

---

## 🛠️ Tech Stack & Libraries
* **Language:** Python 3.13
* **Machine Learning:** XGBoost, Scikit-Learn
* **Data Handling:** Pandas, NumPy
* **Serialization:** Joblib
* **Web Framework:** Streamlit
