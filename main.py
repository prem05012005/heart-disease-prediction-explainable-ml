from visualize import plot_contributions

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from explain import explain_prediction

# -----------------------------
# STEP 1: Load data
# -----------------------------
data = pd.read_csv("data/heart_disease_uci.csv")

# -----------------------------
# STEP 2: Drop useless / high-missing columns
# -----------------------------
drop_cols = ["id", "ca", "thal", "slope"]
data = data.drop(columns=drop_cols)

# -----------------------------
# STEP 3: Handle missing numeric values
# -----------------------------
numeric_cols = ["trestbps", "chol", "thalch", "oldpeak"]
for col in numeric_cols:
    data[col] = data[col].fillna(data[col].median())

# -----------------------------
# STEP 4: Encode categorical columns
# -----------------------------
# Sex
data["sex"] = data["sex"].map({"Male": 1, "Female": 0})

# Chest pain
le_cp = LabelEncoder()
data["cp"] = le_cp.fit_transform(data["cp"])

# Resting ECG
le_ecg = LabelEncoder()
data["restecg"] = le_ecg.fit_transform(data["restecg"])

# Dataset source (will drop later)
le_dataset = LabelEncoder()
data["dataset"] = le_dataset.fit_transform(data["dataset"])

# -----------------------------
# STEP 5: Fill binary columns
# -----------------------------
binary_cols = ["fbs", "exang"]
for col in binary_cols:
    data[col] = data[col].fillna(data[col].mode()[0])

# -----------------------------
# STEP 6: Create binary target
# -----------------------------
data["target"] = data["num"].apply(lambda x: 1 if x > 0 else 0)
data = data.drop(columns=["num"])

# -----------------------------
# STEP 7: Split features & target
# Remove 'dataset' (non-medical feature)
# -----------------------------
X = data.drop(columns=["target", "dataset"])
y = data["target"]

# -----------------------------
# STEP 8: Stratified train-test split (IMPORTANT)
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# -----------------------------
# STEP 9: Feature scaling
# -----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# STEP 10: Train Logistic Regression (tuned)
# -----------------------------
model = LogisticRegression(
    max_iter=3000,
    solver="liblinear",
    C=1.5
)
model.fit(X_train_scaled, y_train)

# -----------------------------
# STEP 11: Accuracy evaluation
# -----------------------------
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print("Heart Disease Prediction Model Trained Successfully")
print("Model Accuracy:", round(accuracy * 100, 2), "%")

# -----------------------------
# STEP 12: Explain prediction (dataset patient)
# -----------------------------
feature_names = X.columns

base_pred, contributions = explain_prediction(
    model, scaler, X, feature_names, sample_index=0
)

print("\nPrediction Probability:", round(base_pred, 3))
print("\nFeature Contributions:")
for k, v in sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True):
    print(f"{k:15s}: {round(v, 4)}")

# -----------------------------
# STEP 13: Manual patient input
# -----------------------------
patient_input = {
    "age": 55,
    "sex": 1,
    "cp": 2,
    "trestbps": 140,
    "chol": 250,
    "fbs": 1,
    "restecg": 1,
    "thalch": 150,
    "exang": 0,
    "oldpeak": 2.3
}

patient_df = pd.DataFrame([patient_input])
patient_scaled = scaler.transform(patient_df)

probability = model.predict_proba(patient_scaled)[0][1]
prediction = 1 if probability >= 0.5 else 0

print("\n--- Manual Patient Prediction ---")
print("Heart Disease Probability:", round(probability, 3))
print("Final Prediction:", "YES" if prediction == 1 else "NO")

# -----------------------------
# STEP 14: Visualization
# -----------------------------
plot_contributions(contributions, top_n=5)
