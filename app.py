import streamlit as st
import pandas as pd
import pickle

# Load saved model and preprocessor
with open("best_salary_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("preprocessor.pkl", "rb") as f:
    preprocessor = pickle.load(f)

st.title("💼 Employee Salary Prediction")

st.markdown("Fill in the employee details to estimate salary:")

# Input form
age = st.number_input("Age", min_value=18, max_value=100, step=1)
gender = st.selectbox("Gender", ["Male", "Female", "Other"])
education = st.selectbox("Education Level", [
    "High School", "Bachelor's", 
    "Master's", "phD"
])
job_title = st.text_input("Job Title", placeholder="e.g. Data Scientist")
experience = st.slider("Years of Experience", min_value=0, max_value=40, step=1)

if st.button("Predict Salary"):
    input_data = pd.DataFrame([{
        "Age": age,
        "Gender": gender,
        "Education Level": education,
        "Job Title": job_title,
        "Years of Experience": experience
    }])

    # Preprocess and predict
    X_processed = preprocessor.transform(input_data)
    predicted_salary = model.predict(X_processed)[0]
    
    st.success(f"💰 Estimated Salary: ${predicted_salary:,.2f}")
