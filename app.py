import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📊", layout="centered")

st.title("📊 Customer Churn Prediction System")
st.write("Enter customer attributes below to evaluate the churn risk using our optimized XGBoost model.")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_model():
    paths = [
        os.path.join(BASE_DIR, "outputs", "xgboost_best_model.pkl"),
        os.path.join(BASE_DIR, "xgboost_best_model.pkl")
    ]
    for path in paths:
        if os.path.exists(path):
            return joblib.load(path)
    return None

model = load_model()

if model is None:
    st.error("⚠️ File template not found. `xgboost_best_model.pkl`!")
    st.info( "Make sure to place the file `xgboost_best_model.pkl` Inside a folder `outputs` Or next to a file `app.py` directly.")
    st.stop()

st.subheader("Customer Profile Details")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", min_value=18, max_value=80, value=38)
    balance = st.number_input("Account Balance ($)", min_value=0.0, max_value=200000.0, value=45000.0, step=1000.0)
    credit_score = st.slider("Credit Score", min_value=300, max_value=850, value=650)

with col2:
    num_products = st.selectbox("Number of Products", options=[1, 2, 3, 4], index=0)
    is_active = st.selectbox("Is Active Member?", options=[1, 0], format_func=lambda x: "Yes" if x == 1 else "No")

st.write("---")

if st.button("Predict Churn Risk", type="primary", use_container_width=True):
    input_data = pd.DataFrame({
        "Age": [age],
        "Balance": [balance],
        "CreditScore": [credit_score],
        "NumOfProducts": [num_products],
        "IsActiveMember": [is_active]
    })
    
    prediction = model.predict(input_data)[0]
    churn_probability = model.predict_proba(input_data)[0][1]
    
    st.subheader("Prediction Result")
    if prediction == 1:
        st.error(f"⚠️ **High Churn Risk!**\n\nThe customer has a **{churn_probability * 100:.1f}%** probability of leaving.")
    else:
        st.success(f"✅ **Low Churn Risk.**\n\nThe customer has a **{(1 - churn_probability) * 100:.1f}%** probability of staying.")