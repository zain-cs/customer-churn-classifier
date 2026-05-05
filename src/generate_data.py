import pandas as pd
import numpy as np

np.random.seed(42)
n = 1000

# --- Generate raw features ---
tenure = np.random.randint(1, 72, n)
monthly_charges = np.round(np.random.uniform(20, 120, n), 2)
total_charges = np.round(tenure * monthly_charges * np.random.uniform(0.9, 1.1, n), 2)
num_products = np.random.randint(1, 5, n)
support_calls = np.random.randint(0, 10, n)
contract = np.random.choice(['Month-to-month', 'One year', 'Two year'], n, p=[0.55, 0.25, 0.20])
payment_method = np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], n)
internet_service = np.random.choice(['DSL', 'Fiber optic', 'No'], n, p=[0.35, 0.45, 0.20])
gender = np.random.choice(['Male', 'Female'], n)
senior_citizen = np.random.choice([0, 1], n, p=[0.84, 0.16])
partner = np.random.choice(['Yes', 'No'], n)
dependents = np.random.choice(['Yes', 'No'], n, p=[0.3, 0.7])

# --- Calculate churn probability based on business logic ---
churn_prob = (
    0.05
    + (tenure < 12) * 0.20          # new customers churn more
    + (monthly_charges > 80) * 0.15  # expensive plan = more likely to leave
    + (support_calls > 5) * 0.15     # frustrated customers leave
    + (contract == 'Month-to-month') * 0.20  # no commitment = easy to leave
    + (internet_service == 'Fiber optic') * 0.05
    + (payment_method == 'Electronic check') * 0.05
    + senior_citizen * 0.05
    - (num_products >= 3) * 0.10     # bundled = stickier
    - (partner == 'Yes') * 0.05
)
churn_prob = np.clip(churn_prob, 0.02, 0.95)
churn = (np.random.rand(n) < churn_prob).astype(int)

# --- Build the DataFrame ---
df = pd.DataFrame({
    'CustomerID': [f'CUST-{i:04d}' for i in range(n)],
    'Gender': gender,
    'SeniorCitizen': senior_citizen,
    'Partner': partner,
    'Dependents': dependents,
    'Tenure': tenure,
    'InternetService': internet_service,
    'Contract': contract,
    'PaymentMethod': payment_method,
    'MonthlyCharges': monthly_charges,
    'TotalCharges': total_charges,
    'NumProducts': num_products,
    'SupportCalls': support_calls,
    'Churn': churn
})

df.to_csv('data/churn_data.csv', index=False)
print(f"Dataset created! {n} rows | Churn rate: {churn.mean():.1%}")
print(df.head())