# churn_rf_pipeline.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import __version__ as sklearn_version
from packaging import version

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)

# -------------------------------
# 1) Load data
# -------------------------------
df = pd.read_csv(r"C:\Users\mayur\Downloads\Customer_Churn_Project\vw_ChurnData.csv")

# Keep only 'Stayed' and 'Churned'
df = df[df["Customer_Status"].isin(["Stayed", "Churned"])].copy()

# Create binary target: 1 = Churned, 0 = Stayed
df["churn"] = (df["Customer_Status"] == "Churned").astype(int)

# Drop leakage/ID columns
drop_cols = ["Customer_ID", "Customer_Status", "Churn_Category", "Churn_Reason"]
drop_cols = [c for c in drop_cols if c in df.columns]
X = df.drop(columns=drop_cols + ["churn"])
y = df["churn"]

# -------------------------------
# 2) Feature grouping
# -------------------------------
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in X.columns if c not in num_cols]

# -------------------------------
# 3) Preprocessing
# -------------------------------
num_tf = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

# Handle sklearn version compatibility for OneHotEncoder
if version.parse(sklearn_version) >= version.parse("1.2"):
    onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
else:
    onehot = OneHotEncoder(handle_unknown="ignore", sparse=False)

cat_tf = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", onehot)
])

preprocessor = ColumnTransformer([
    ("num", num_tf, num_cols),
    ("cat", cat_tf, cat_cols)
])

# -------------------------------
# 4) Train-test split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# -------------------------------
# 5) Model
# -------------------------------
rf = RandomForestClassifier(
    n_estimators=400,
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1
)

clf = Pipeline([
    ("preprocess", preprocessor),
    ("model", rf)
])

# -------------------------------
# 6) Train
# -------------------------------
clf.fit(X_train, y_train)

# -------------------------------
# 7) Evaluate
# -------------------------------
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

roc_auc = roc_auc_score(y_test, y_proba)
print(f"ROC AUC: {roc_auc:.3f}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ROC curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# -------------------------------
# 8) Feature importance
# -------------------------------
model = clf.named_steps["model"]

# Get one-hot feature names
ohe = clf.named_steps["preprocess"].named_transformers_["cat"].named_steps["onehot"]
ohe_features = ohe.get_feature_names_out(cat_cols)
all_features = num_cols + list(ohe_features)

importances = model.feature_importances_
feat_imp = pd.DataFrame({
    "feature": all_features,
    "importance": importances
}).sort_values(by="importance", ascending=False)

print("\nTop 10 Features by Importance:\n")
print(feat_imp.head(10))

# -------------------------------
# 9) Save predictions to CSV
# -------------------------------
# Predict for the full dataset
df["Predicted_Prob"] = clf.predict_proba(X)[:, 1]
df["Predicted_Status"] = np.where(df["Predicted_Prob"] >= 0.5, "Churn", "Stay")

df.to_csv(r"C:\Users\mayur\Downloads\Customer_Churn_Project\prediction.csv", index=False)
print("\n✅ Predictions saved as prediction.csv")
import joblib


joblib.dump(clf, "churn_model_rf.joblib")
print("\n✅ Model pipeline saved as churn_model_rf.joblib")
import pandas as pd
import joblib

# -------------------------------
# 1) Load saved model
# -------------------------------
clf = joblib.load("churn_model_rf.joblib")

# -------------------------------
# 2) Load new data
# -------------------------------
new_data = pd.read_csv("vw_JoinData.csv")

# Keep Customer_ID for tracking if available
drop_cols = ["Customer_ID", "Customer_Status", "Churn_Category", "Churn_Reason"]
drop_cols = [c for c in drop_cols if c in new_data.columns]

X_new = new_data.drop(columns=drop_cols)

# -------------------------------
# 3) Make predictions
# -------------------------------
new_data["Predicted_Prob"] = clf.predict_proba(X_new)[:, 1]
new_data["Predicted_Status"] = (new_data["Predicted_Prob"] >= 0.5).astype(int)  # 1 = Churn, 0 = Stay

# -------------------------------
# 4) Filter only churned customers
# -------------------------------
churned_customers = new_data[new_data["Predicted_Status"] == 1]

# -------------------------------
# 5) Save output
# -------------------------------
churned_customers.to_csv(r"C:\Users\mayur\Downloads\Customer_Churn_Project/vw_JoinData_predictions.csv", index=False)
print("✅ Only churned customers saved as vw_JoinData_churned.csv")
