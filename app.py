import streamlit as st
import pandas as pd
import joblib

preprocessor = joblib.load("heart_preprocessor.pkl")
model = joblib.load("heart_naive_bayes_model.pkl")

st.set_page_config(page_title="Heart Disease Prediction", layout="centered")

st.title("❤️ Heart Disease Prediction App")
st.write("Enter patient details to predict heart disease risk.")

with st.form("heart_form"):

    age = st.number_input("Age", min_value=1, max_value=120, value=50)
    sex = st.selectbox("Sex", ["M", "F"])
    chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "ASY", "TA"])
    resting_bp = st.number_input("Resting Blood Pressure", min_value=80, max_value=250, value=130)
    cholesterol = st.number_input("Cholesterol", min_value=100, max_value=600, value=200)
    fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
    resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
    max_hr = st.number_input("Max Heart Rate", min_value=60, max_value=220, value=150)
    exercise_angina = st.selectbox("Exercise Induced Angina", ["Yes", "No"])
    exercise_angina = 1 if exercise_angina == "Yes" else 0

    oldpeak = st.number_input("Oldpeak (ST Depression)", min_value=-3.0, max_value=7.0, value=1.0)
    st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

    submit = st.form_submit_button("Predict")

if submit:

    input_df = pd.DataFrame([{
        "Age": age,
        "Sex": sex,
        "ChestPainType": chest_pain,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_bs,
        "RestingECG": resting_ecg,
        "MaxHR": max_hr,
        "ExerciseAngina": exercise_angina,
        "Oldpeak": oldpeak,
        "ST_Slope": st_slope
    }])

    X_proc = preprocessor.transform(input_df)

    prediction = model.predict(X_proc)[0]
    probability = model.predict_proba(X_proc)[0][1]

    st.subheader("🔍 Prediction Result")

    if prediction == 1:
        st.error(f"⚠️ High Risk of Heart Disease\n\nProbability: {probability:.2%}")
    else:
        st.success(f"✅ Low Risk of Heart Disease\n\nProbability: {probability:.2%}")
