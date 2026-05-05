import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os, pickle, warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, ConfusionMatrixDisplay)
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

os.makedirs('models', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

# ── 1. Load Data ──────────────────────────────────────────────────────────────
df = pd.read_csv('data/churn_data.csv')
print(f"Loaded {len(df)} rows | Churn rate: {df['Churn'].mean():.1%}\n")

# ── 2. Preprocess ─────────────────────────────────────────────────────────────
df = df.drop(columns=['CustomerID'])

# Convert text columns to numbers
cat_cols = ['Gender', 'Partner', 'Dependents', 'InternetService', 'Contract', 'PaymentMethod']
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

X = df.drop(columns=['Churn'])  # features
y = df['Churn']                 # target (0 = stay, 1 = churn)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows\n")

# ── 3. Train Models ───────────────────────────────────────────────────────────
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'XGBoost': XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                              eval_metric='logloss', random_state=42)
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    auc   = roc_auc_score(y_test, proba)
    cv    = cross_val_score(model, X, y, cv=5, scoring='roc_auc').mean()
    results[name] = {'model': model, 'preds': preds, 'proba': proba,
                     'auc': auc, 'cv_auc': cv}
    print(f"{'─'*40}")
    print(f"{name}")
    print(f"  Test AUC : {auc:.4f}")
    print(f"  CV  AUC  : {cv:.4f}")
    print(classification_report(y_test, preds, target_names=['Stay', 'Churn']))

# ── 4. Save Best Model ────────────────────────────────────────────────────────
best_name = max(results, key=lambda k: results[k]['auc'])
best = results[best_name]
pickle.dump(best['model'], open('models/churn_model.pkl', 'wb'))
print(f"\n✅ Best model: {best_name} (AUC={best['auc']:.4f}) saved!")

# ── 5. Plot Results ───────────────────────────────────────────────────────────
sns.set_theme(style='whitegrid', palette='muted')
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Customer Churn Classifier — Results', fontsize=15, fontweight='bold')

# ROC Curve
ax = axes[0]
for name, r in results.items():
    fpr, tpr, _ = roc_curve(y_test, r['proba'])
    ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC={r['auc']:.3f})")
ax.plot([0,1],[0,1],'k--', lw=1)
ax.set(xlabel='False Positive Rate', ylabel='True Positive Rate', title='ROC Curve')
ax.legend(fontsize=9)

# Confusion Matrix
ax = axes[1]
cm = confusion_matrix(y_test, best['preds'])
ConfusionMatrixDisplay(cm, display_labels=['Stay','Churn']).plot(ax=ax, colorbar=False, cmap='Blues')
ax.set_title(f'Confusion Matrix — {best_name}')

# Feature Importance
ax = axes[2]
xgb_model = results['XGBoost']['model']
feat_imp = pd.Series(xgb_model.feature_importances_, index=X.columns).sort_values(ascending=True)
feat_imp.tail(10).plot(kind='barh', ax=ax, color='steelblue')
ax.set(title='Top Feature Importances (XGBoost)', xlabel='Importance')

plt.tight_layout()
plt.savefig('outputs/results.png', dpi=150, bbox_inches='tight')
print("📊 Plot saved to outputs/results.png")
plt.show()