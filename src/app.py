import os
import pandas as pd
import joblib
import streamlit as st

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "model")

MODEL_PATH = os.path.join(MODEL_DIR, "churn_prediction_model.pkl")
LE_GENDER_PATH = os.path.join(MODEL_DIR, "Gender_encoder.pkl")
LE_GEOGRAPHY_PATH = os.path.join(MODEL_DIR, "Geography_encoder.pkl")

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Please run the training script first.")
    return joblib.load(MODEL_PATH)

# Load model and encoders
model = load_model()
le_gender = joblib.load(LE_GENDER_PATH)
le_geography = joblib.load(LE_GEOGRAPHY_PATH)

st.title("Bank Churn Prediction App")
st.write("Enter customer details below:")

# Input fields
credit_score = st.number_input("Credit Score", min_value=0, value=650)
age = st.number_input("Age", min_value=18, max_value=100, value=42)
tenure = st.number_input("Tenure (Years)", min_value=0, value=3)
balance = st.number_input("Balance", min_value=0.0, value=60000.0, step=1000.0)
num_of_products = st.number_input("Number of Products", min_value=1, max_value=10, value=2)
has_cr_card = st.selectbox("Has Credit Card?", ["No", "Yes"])
is_active_member = st.selectbox("Is Active Member?", ["No", "Yes"])
estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=50000.0, step=1000.0)
gender = st.selectbox("Gender", ["Male", "Female"])
geography = st.selectbox("Geography", ["France", "Spain", "Germany"])

# Create input dataframe in same column order as training
input_data = pd.DataFrame({
    "CreditScore": [credit_score],
    "Geography": [geography],
    "Gender": [gender],
    "Age": [age],
    "Tenure": [tenure],
    "Balance": [balance],
    "NumOfProducts": [num_of_products],
    "HasCrCard": [1 if has_cr_card == "Yes" else 0],
    "IsActiveMember": [1 if is_active_member == "Yes" else 0],
    "EstimatedSalary": [estimated_salary]
})

# Encode categorical features
input_data["Gender"] = le_gender.transform(input_data["Gender"])
input_data["Geography"] = le_geography.transform(input_data["Geography"])

if st.button("Predict Churn"):
    pred = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]  # probability of churn
    if pred == 1:
        st.error(f"This customer is likely to CHURN. (Confidence: {prob*100:.2f}%)")
    else:
        st.success(f"This customer is likely to STAY. (Confidence: {(1-prob)*100:.2f}%)")