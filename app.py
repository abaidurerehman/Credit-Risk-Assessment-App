import streamlit as st
import numpy as np
import joblib
import os

# -------------------------------
# Page Config & Style
st.set_page_config(page_title="Credit Risk App", layout="centered")

# Custom CSS for stylish divs and layout
st.markdown("""
    <style>
        body {
            background-color: #f0f2f6;
        }

        .main-container {
            background-color: #ffffff;
            padding: 2rem;
            border-radius: 20px;
            box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
            max-width: 900px;
            margin: auto;
        }

        .header {
            text-align: center;
            padding-bottom: 10px;
        }

        .stButton>button {
            background-color: #0066cc;
            color: white;
            padding: 10px 20px;
            border-radius: 10px;
            font-size: 16px;
            font-weight: bold;
        }

        .result-box {
            margin-top: 2rem;
            padding: 1.5rem;
            border-radius: 15px;
            font-size: 18px;
            text-align: center;
        }

        .success-box {
            background-color: #e0f8e9;
            color: #006600;
        }

        .error-box {
            background-color: #fdecea;
            color: #990000;
        }
    </style>
""", unsafe_allow_html=True)

# -------------------------------
# Load Model and Scaler
MODEL_PATH = "xgb_model.pkl"
SCALER_PATH = "scaler.pkl"

if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    st.error("❌ Model or Scaler not found. Make sure 'xgb_model.pkl' and 'scaler.pkl' exist.")
    st.stop()

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# -------------------------------
# Main UI Container
st.markdown('<div class="main-container">', unsafe_allow_html=True)

# Header
st.markdown('<div class="header"><h2>💳 Credit Risk Assessment App</h2><p>Evaluate customer credit risk with an ML model.</p></div>', unsafe_allow_html=True)

# Input Form
with st.form("credit_form"):
    col1, col2 = st.columns(2)

    with col1:
        RevolvingUtilizationOfUnsecuredLines = st.number_input("Revolving Utilization", 0.0, 10.0, 0.5, format="%.4f")
        age = st.number_input("Age", 18, 120, 35)
        NumberOfTime30_59DaysPastDueNotWorse = st.number_input("30-59 Days Past Due", 0, 98, 0)
        DebtRatio = st.number_input("Debt Ratio", 0.0, 10000.0, 100.0, format="%.2f")
        MonthlyIncome = st.number_input("Monthly Income", 0.0, 200000.0, 5000.0, format="%.2f")

    with col2:
        NumberOfOpenCreditLinesAndLoans = st.number_input("Open Credit Lines", 0, 60, 10)
        NumberOfTimes90DaysLate = st.number_input("90 Days Late", 0, 98, 0)
        NumberRealEstateLoansOrLines = st.number_input("Real Estate Loans", 0, 20, 1)
        NumberOfTime60_89DaysPastDueNotWorse = st.number_input("60-89 Days Past Due", 0, 98, 0)
        NumberOfDependents = st.number_input("Dependents", 0, 20, 1)

    submit = st.form_submit_button("📊 Predict Credit Risk")

# -------------------------------
# Prediction Result
if submit:
    try:
        input_data = np.array([[
            RevolvingUtilizationOfUnsecuredLines, age,
            NumberOfTime30_59DaysPastDueNotWorse, DebtRatio,
            MonthlyIncome, NumberOfOpenCreditLinesAndLoans,
            NumberOfTimes90DaysLate, NumberRealEstateLoansOrLines,
            NumberOfTime60_89DaysPastDueNotWorse, NumberOfDependents
        ]])

        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]

        if prediction == 1:
            st.markdown(
                f'<div class="result-box error-box">🚨 <strong>High Credit Risk Detected!</strong><br>Probability: {probability:.2%}</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="result-box success-box">✅ <strong>Low Credit Risk</strong><br>Probability: {probability:.2%}</div>',
                unsafe_allow_html=True
            )

    except Exception as e:
        st.exception(f"Prediction failed due to: {str(e)}")

st.markdown('</div>', unsafe_allow_html=True)  # Close main-container
