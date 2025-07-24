import streamlit as st
import pandas as pd
import pickle
import plotly.graph_objects as go

# Page config
st.set_page_config(page_title="Salary Prediction", page_icon="💼", layout="centered")

# Load model and preprocessor
with open("best_salary_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("preprocessor.pkl", "rb") as f:
    preprocessor = pickle.load(f)

# Header
st.title("💼 Employee Salary Predictor")
st.markdown("This app predicts the **estimated salary** of an employee based on their details.")

st.divider()

# Sidebar for input
st.sidebar.header("📋 Input Employee Details")

age = st.sidebar.number_input("Age", min_value=18, max_value=100, step=1, value=30)

gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"])

education = st.sidebar.selectbox("Education Level", [
    "High School", "Bachelor's", "Master's", "phD"
])


job_title_options = sorted([
    'Account Manager', 'Accountant', 'Administrative Assistant', 'Business Analyst',
    'Business Development Manager', 'Business Intelligence Analyst', 'CEO',
    'Copywriter', 'Creative Director', 'Customer Service Manager',
    'Customer Service Rep', 'Customer Service Representative', 'Customer Success Rep',
    'Data Analyst', 'Data Entry Clerk', 'Data Scientist', 'Delivery Driver',
    'Digital Marketing Manager', 'Digital Marketing Specialist', 'Director',
    'Director of Marketing', 'Director of Operations', 'Event Coordinator',
    'Financial Analyst', 'Financial Manager', 'Graphic Designer', 'Help Desk Analyst',
    'HR Generalist', 'HR Manager', 'Human Resources Director', 'Human Resources Manager',
    'IT Manager', 'IT Support', 'Junior Accountant', 'Junior Developer',
    'Junior Sales Associate', 'Juniour HR Coordinator', 'Juniour HR Generalist',
    'Marketing Analyst', 'Marketing Coordinator', 'Marketing Director',
    'Marketing Manager', 'Marketing Specialist', 'Network Engineer',
    'Operations Director', 'Operations Manager', 'Product Designer',
    'Product Manager', 'Project Engineer', 'Project Manager', 'Receptionist',
    'Recruiter', 'Research Director', 'Sales Associate', 'Sales Director',
    'Sales Executive', 'Sales Manager', 'Senior Consultant', 'Senior Data Scientist',
    'Senior Engineer', 'Senior Financial Analyst', 'Senior Manager',
    'Senior Project Engineer', 'Senior Scientist', 'Senior Software Engineer',
    'Social Media Man', 'Social Media Manager',
    'Social Media Specialist', 'Software Developer', 'Software Engineer',
    'Software Engineer Manager', 'Software Manager', 'Strategy Consultant',
    'System Administrator', 'Technical Support Specialist', 'Technical Writer',
    'UX Designer', 'UX Researcher', 'VP of Finance', 'VP of Operations',
    'Web Developer'
])


job_title = st.sidebar.selectbox("Job Title", job_title_options)


experience = st.sidebar.slider("Years of Experience", 0, 40, 5)


# Predict button
if st.sidebar.button("📈 Predict Salary"):
    input_data = pd.DataFrame([{
        "Age": age,
        "Gender": gender,
        "Education Level": education,
        "Job Title": job_title,
        "Years of Experience": experience
    }])

    try:
        # Preprocess and predict
        X_processed = preprocessor.transform(input_data)
        predicted_salary = model.predict(X_processed)[0]

        # Conditional color
        if predicted_salary >= 150000:
            color = "#1b5e20"  # dark green
            remark = "🏆 Top-tier salary! Well deserved."
        elif predicted_salary >= 80000:
            color = "#2e7d32"  # medium green
            remark = "💼 Competitive salary range!"
        else:
            color = "#f9a825"  # amber
            remark = "📈 Room to grow! Keep building skills."

        # Define salary gauge chart
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=predicted_salary,
            # title={'text': "Salary Gauge", 'font': {'size': 24}},
            gauge={
                'axis': {'range': [0, 200000], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "darkgreen"},
                'steps': [
                    {'range': [0, 50000], 'color': '#ffecb3'},
                    {'range': [50000, 100000], 'color': '#c8e6c9'},
                    {'range': [100000, 150000], 'color': '#a5d6a7'},
                    {'range': [150000, 200000], 'color': '#81c784'},
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': predicted_salary
                }
            }
        ))

        # Stylish card output
        # st.markdown("---")
        st.markdown(
            f"""
            <div style="background-color: #f1f8e9; padding: 25px; border-radius: 12px;
                        text-align: center; box-shadow: 0 6px 12px rgba(0,0,0,0.1);">
                <h2 style="color: {color}; margin-bottom: 10px;">💰 Estimated Annual Salary</h2>
                <p style="font-size: 40px; font-weight: bold; color: {color}; margin: 0;">
                    ${predicted_salary:,.2f}
                </p>
                <p style="font-size: 18px; color: #555; margin-top: 10px;">{remark}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Replace balloons with emoji
        st.balloons()
        # Gauge chart
        st.plotly_chart(gauge, use_container_width=True)
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")



# # Optional footer
# st.markdown("---")
# st.caption("🔍 *Model trained using historical employee data and deployed with Streamlit.*")
