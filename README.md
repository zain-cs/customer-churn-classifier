v---

## Quickstart

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/customer-churn-classifier.git
cd customer-churn-classifier
```

### 2. Create and activate virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Generate the dataset
```bash
python src/generate_data.py
```

### 5. Train the models
```bash
python src/train.py
```

### 6. Predict on new customers
```bash
python src/predict.py
```

---

## Features

| Feature | Description |
|---|---|
| `Tenure` | Number of months the customer has been with the company |
| `Contract` | Contract type — Month-to-month, One year, Two year |
| `MonthlyCharges` | Current monthly bill amount |
| `SupportCalls` | Number of support tickets raised |
| `NumProducts` | Number of services the customer is subscribed to |
| `InternetService` | Type of internet service — DSL, Fiber optic, or None |
| `PaymentMethod` | How the customer pays their bill |

---

## Key Insights

- **Contract type** is the strongest predictor — month-to-month customers churn at 3x the rate of annual customers
- **New customers churn most** — the majority of churners leave within their first year
- **High support calls** signal dissatisfaction and strongly predict churn
- **Bundled customers stay longer** — subscribing to multiple products reduces churn significantly

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.10+ | Core language |
| pandas & NumPy | Data manipulation |
| scikit-learn | Logistic Regression, preprocessing, evaluation |
| XGBoost | Gradient boosting classifier |
| matplotlib & seaborn | Visualization |

---

## Future Improvements

- [ ] Add SHAP values for per-customer explainability
- [ ] Build a Streamlit dashboard for business users
- [ ] Connect to real Telco dataset from Kaggle
- [ ] Hyperparameter tuning with Optuna
- [ ] Deploy as a REST API with FastAPI

---

## License

MIT License — free to use, modify, and distribute.