<div align="center">

<img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
<img src="https://img.shields.io/badge/XGBoost-FF6600?style=for-the-badge&logo=xgboost&logoColor=white"/>
<img src="https://img.shields.io/badge/Status-Complete-brightgreen?style=for-the-badge"/>

# 📉 Customer Churn Classifier

**Predict customer churn before it happens — and save the revenue.**

*An end-to-end machine learning pipeline built for real-world business impact.*

</div>

---

## 🧩 Problem Statement

Every month, subscription businesses lose customers silently. By the time churn is noticed, it's too late. This project answers one critical question:

> **"Which customers are most likely to leave — and why?"**

Using historical customer behavior, this classifier flags high-risk customers with a churn probability score, giving retention teams the window they need to act.

---

## 📊 Model Performance

<div align="center">

| Model | Test AUC | CV AUC (5-fold) | Accuracy |
|---|---|---|---|
| ✅ Logistic Regression | **0.704** | 0.675 | 72% |
| XGBoost | 0.687 | 0.691 | 72% |

</div>

![Results](outputs/results.png)

> 📌 **Logistic Regression** was selected as the final model based on Test AUC performance.

---

## 🚀 Quickstart

```bash
# Clone the repo
git clone https://github.com/zain-cs/customer-churn-classifier.git
cd customer-churn-classifier

# Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Run the full pipeline
python src/generate_data.py  # Step 1 — Generate data
python src/train.py          # Step 2 — Train & evaluate
python src/predict.py        # Step 3 — Predict new customers
```

---

## 🗂️ Project Structure