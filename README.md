# ❤️ Heart Disease Prediction – Explainable Machine Learning

## 📌 Project Overview
This project predicts the likelihood of heart disease using clinical patient data.
Unlike black-box models, it also explains **why** a prediction was made by identifying
which features increased or reduced the risk.

The goal is to build a **trustworthy and interpretable medical ML system**.

---

## 🚀 Key Features
- Heart disease prediction (Yes / No)
- Probability-based risk estimation
- Feature contribution explainability
- Visualization of risk factors
- Interactive Streamlit web application

---

## 🧠 Machine Learning Model
- **Algorithm:** Logistic Regression  
- **Accuracy:** ~80.87% on unseen test data  
- **Why Logistic Regression?**
  - Suitable for binary classification
  - Produces stable probability estimates
  - Highly interpretable (important in healthcare)

---

## 🔍 Explainability Approach
For a given patient:
- The model predicts heart disease probability
- Each feature’s contribution is calculated by measuring
  how the prediction changes when that feature is neutralized
- Positive contribution → increases risk  
- Negative contribution → reduces risk  

This makes the model’s decision transparent and understandable.

---

## 📊 Visualization
- Bar chart showing top contributing features
- Red bars → increase heart disease risk
- Green bars → reduce heart disease risk

---

## 🛠 Tech Stack
- Python  
- Pandas, NumPy  
- Scikit-learn  
- Matplotlib  
- Streamlit  

---

## ▶️ How to Run Locally

### 1️⃣ Clone the repository
```bash
git clone https://github.com/prem05012005/heart-disease-prediction-explainable-ml.git
cd heart-disease-prediction-explainable-ml
2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Run the Streamlit app
streamlit run app.py
