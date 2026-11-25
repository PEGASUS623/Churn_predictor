# train.py
import pandas as pd
import numpy as np
import joblib
import sklearn

from sklearn.model_selection import (train_test_split, GridSearchCV,
                                     StratifiedKFold, cross_val_predict,
                                     cross_val_score)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, precision_recall_curve)

# ---------- 1. Load and clean ----------
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
df = df.drop('customerID', axis=1)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna()
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# ---------- 2. Feature engineering ----------
service_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                'TechSupport', 'StreamingTV', 'StreamingMovies']
df['NumServices'] = (df[service_cols] == 'Yes').sum(axis=1)
df['AutoPay'] = df['PaymentMethod'].str.contains('automatic').astype(int)

# TotalCharges is roughly tenure x MonthlyCharges (multicollinearity)
df = df.drop('TotalCharges', axis=1)

# ---------- 3. Features / target, split first ----------
X = df.drop('Churn', axis=1)
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

# ---------- 4. Preprocessing ----------
num_cols = ['tenure', 'MonthlyCharges', 'NumServices']
cat_cols = [c for c in X_train.columns if c not in num_cols + ['AutoPay']]

pre = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(drop='if_binary', handle_unknown='ignore'), cat_cols)
], remainder='passthrough')   # AutoPay passes through as 0/1

# ---------- 5. Logistic regression + grid search ----------
lr = Pipeline([('pre', pre),
               ('clf', LogisticRegression(max_iter=2000))])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid = GridSearchCV(lr, {'clf__C': [0.01, 0.1, 1, 10, 100]},
                    scoring='roc_auc', cv=cv)
grid.fit(X_train, y_train)
best_lr = grid.best_estimator_
print("Best C:", grid.best_params_, "CV AUC:", round(grid.best_score_, 4))

# ---------- 6. Compare with Gradient Boosting ----------
gb = Pipeline([('pre', pre),
               ('clf', GradientBoostingClassifier(random_state=42))])
gb_auc = cross_val_score(gb, X_train, y_train, cv=cv, scoring='roc_auc').mean()
print("Gradient Boosting CV AUC:", round(gb_auc, 4))

# Keep logistic regression unless GB is clearly better (interpretability)
final_model = best_lr if grid.best_score_ >= gb_auc - 0.005 else gb
print("Final model:", "Logistic Regression" if final_model is best_lr else "Gradient Boosting")

# ---------- 7. Choose threshold on out-of-fold TRAIN predictions ----------
oof_prob = cross_val_predict(final_model, X_train, y_train, cv=cv,
                             method='predict_proba')[:, 1]
prec, rec, thr = precision_recall_curve(y_train, oof_prob)

beta = 1   # use 2 to favour recall, 0.5 to favour precision
f = (1 + beta**2) * prec * rec / (beta**2 * prec + rec + 1e-9)
best_t = float(thr[np.argmax(f[:-1])])
print("Chosen threshold:", round(best_t, 3))

# ---------- 8. Fit and evaluate on the test set (once) ----------
final_model.fit(X_train, y_train)
y_prob = final_model.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= best_t).astype(int)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("ROC-AUC:", round(roc_auc_score(y_test, y_prob), 4))
# ---------- 10. Save for the app ----------
joblib.dump(final_model, 'churn_model.joblib')
joblib.dump(best_t, 'threshold.joblib')
print("\nSaved churn_model.joblib and threshold.joblib")
print("scikit-learn version used:", sklearn.__version__)   # put this in requirements.txt