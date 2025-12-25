import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder

# -----------------------------
# Load and preprocess data
# -----------------------------
data = pd.read_csv("data/heart_disease_uci.csv")

drop_cols = ["id", "ca", "thal", "slope"]
data = data.drop(columns=drop_cols)

numeric_cols = ["trestbps", "chol", "thalch", "oldpeak"]
for col in numeric_cols:
    data[col] = data[col].fillna(data[col].median())

data["sex"] = data["sex"].map({"Male": 1, "Female": 0})

le_cp = LabelEncoder()
data["cp"] = le_cp.fit_transform(data["cp"])

le_ecg = LabelEncoder()
data["restecg"] = le_ecg.fit_transform(data["restecg"])

binary_cols = ["fbs", "exang"]
for col in binary_cols:
    data[col] = data[col].fillna(data[col].mode()[0])

data["target"] = data["num"].apply(lambda x: 1 if x > 0 else 0)
data = data.drop(columns=["num", "dataset"])

X = data.drop(columns=["target"])
y = data["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

model = LogisticRegression(max_iter=3000, solver="liblinear", C=1.5)
model.fit(X_train_scaled, y_train)

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("❤️ Heart Disease Prediction System")

st.markdown("""
### 📝 How to Fill the Inputs

**Chest Pain Type (cp):**
- `0` → Typical angina (heart-related chest pain)
- `1` → Atypical angina
- `2` → Non-anginal pain
- `3` → Asymptomatic (no chest pain)

**Rest ECG (restecg):**
- `0` → Normal
- `1` → ST-T wave abnormality
- `2` → Left ventricular hypertrophy

**Exercise Induced Angina (exang):**
- `0` → No chest pain during exercise
- `1` → Chest pain during exercise

**Fasting Blood Sugar (fbs):**
- `0` → ≤ 120 mg/dL (Normal)
- `1` → > 120 mg/dL (High)
""")

st.write("Enter patient details to predict heart disease risk")

# -----------------------------
# User Inputs
# -----------------------------
age = st.number_input("Age", 1, 120, 55)
sex = st.selectbox("Sex", ["Male", "Female"])
cp = st.number_input("Chest Pain Type (0–3)", 0, 3, 2)
trestbps = st.number_input("Resting Blood Pressure", 80, 200, 140)
chol = st.number_input("Cholesterol", 100, 400, 250)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])
restecg = st.number_input("Rest ECG (0–2)", 0, 2, 1)
thalch = st.number_input("Max Heart Rate", 60, 220, 150)
exang = st.selectbox("Exercise Induced Angina", [0, 1])
oldpeak = st.number_input("Oldpeak", 0.0, 6.0, 2.3)

# -----------------------------
# Visualization function
# -----------------------------
def plot_contributions_streamlit(contributions):
    sorted_items = sorted(
        contributions.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:5]

    features = [f for f, _ in sorted_items]
    values = [v for _, v in sorted_items]
    colors = ["red" if v > 0 else "green" for v in values]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(features, values, color=colors)
    ax.set_xlabel("Contribution to Prediction")
    ax.set_title("Top Feature Contributions")
    ax.invert_yaxis()

    st.pyplot(fig)

# -----------------------------
# Prediction + Explainability
# -----------------------------
if st.button("Predict"):
    input_data = pd.DataFrame([{
        "age": age,
        "sex": 1 if sex == "Male" else 0,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalch": thalch,
        "exang": exang,
        "oldpeak": oldpeak
    }])

    input_scaled = scaler.transform(input_data)
    prob = model.predict_proba(input_scaled)[0][1]
    prediction = "YES" if prob >= 0.5 else "NO"

    st.subheader("Prediction Result")
    st.write("Heart Disease Probability:", round(prob, 3))
    st.write("Final Prediction:", prediction)

    # -------- Explainability --------
    contributions = {}
    for col in input_data.columns:
        modified = input_data.copy()
        modified[col] = X[col].mean()
        modified_scaled = scaler.transform(modified)
        new_prob = model.predict_proba(modified_scaled)[0][1]
        contributions[col] = prob - new_prob

    st.subheader("Feature Contribution Explanation")
    plot_contributions_streamlit(contributions)

    st.markdown("""
### 📘 How to Read This Chart
- **Red bars** → Increase heart disease risk  
- **Green bars** → Reduce heart disease risk  
- Longer bars → Stronger influence  
""")

    # -------- Human-friendly explanation --------
    st.markdown("### 🧠 What These Top Factors Mean")

    feature_descriptions = {
        "oldpeak": "ST depression during exercise. Higher values indicate more heart stress.",
        "cp": "Type of chest pain experienced by the patient.",
        "exang": "Chest pain occurring during physical exercise.",
        "sex": "Biological sex (statistical risk factor).",
        "chol": "Cholesterol level in the blood."
    }

    top_features = sorted(
        contributions.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:5]

    for i, (feature, value) in enumerate(top_features, start=1):
        impact = "increases" if value > 0 else "reduces"
        strength = "strongly" if abs(value) > 0.15 else "moderately"

        st.write(
            f"**{i}. {feature.upper()}** – {feature_descriptions.get(feature, '')} "
            f"This factor **{strength} {impact}** the predicted heart disease risk."
        )

    st.markdown("""
⚠️ **Note:**  
This explanation is personalized for the entered values and is meant for educational purposes only.
It does not replace professional medical advice.
""")
