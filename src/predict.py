import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the saved model
model = pickle.load(open('models/churn_model.pkl', 'rb'))

# --- Define new customers to predict ---
new_customers = pd.DataFrame([
    {
        # High risk customer — new, expensive plan, many complaints
        'Gender': 'Male', 'SeniorCitizen': 0, 'Partner': 'Yes',
        'Dependents': 'No', 'Tenure': 2, 'InternetService': 'Fiber optic',
        'Contract': 'Month-to-month', 'PaymentMethod': 'Electronic check',
        'MonthlyCharges': 95.0, 'TotalCharges': 190.0,
        'NumProducts': 1, 'SupportCalls': 7
    },
    {
        # Low risk customer — loyal, cheap plan, few complaints
        'Gender': 'Female', 'SeniorCitizen': 0, 'Partner': 'Yes',
        'Dependents': 'Yes', 'Tenure': 48, 'InternetService': 'DSL',
        'Contract': 'Two year', 'PaymentMethod': 'Bank transfer',
        'MonthlyCharges': 55.0, 'TotalCharges': 2640.0,
        'NumProducts': 3, 'SupportCalls': 1
    }
])

# Encode text columns (same as training)
cat_cols = ['Gender', 'Partner', 'Dependents', 'InternetService', 'Contract', 'PaymentMethod']
le = LabelEncoder()
for col in cat_cols:
    new_customers[col] = le.fit_transform(new_customers[col])

# Make predictions
probs = model.predict_proba(new_customers)[:, 1]
preds = model.predict(new_customers)

# Show results
print("\n📋 Churn Prediction Results")
print("─" * 40)
for i, (pred, prob) in enumerate(zip(preds, probs)):
    label = "⚠️  WILL CHURN" if pred == 1 else "✅ Will Stay"
    print(f"Customer {i+1}: {label} (churn probability: {prob:.1%})")