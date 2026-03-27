import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load model & columns
model = pickle.load(open('model.pkl', 'rb'))
columns = pickle.load(open('columns.pkl', 'rb'))

st.title("🎯 Job Acceptance Prediction System")

st.write("Enter Candidate Details:")

# ---------- INPUTS ----------

technical_score = st.slider("Technical Score", 0, 100, 70)
communication_score = st.slider("Communication Score", 0, 100, 65)
skills_match = st.slider("Skills Match %", 0, 100, 70)
experience = st.slider("Years of Experience", 0, 10, 2)
certifications = st.slider("Certifications Count", 0, 10, 2)

company_tier = st.selectbox("Company Tier", ["Tier 1", "Tier 2", "Tier 3"])
job_role_match = st.selectbox("Job Role Match", ["Matched", "Not Matched"])
relocation = st.selectbox("Relocation Willingness", ["Yes", "No"])

# ---------- FEATURE ENGINEERING ----------

# Experience category
if experience <= 1:
    exp_cat = "Fresher"
elif experience <= 3:
    exp_cat = "Junior"
else:
    exp_cat = "Senior"

# Skills level
if skills_match <= 50:
    skills_level = "Low"
elif skills_match <= 75:
    skills_level = "Medium"
else:
    skills_level = "High"

# Interview performance
if technical_score <= 60:
    performance = "Poor"
elif technical_score <= 75:
    performance = "Average"
else:
    performance = "Good"

# Placement score
placement_score = (
    0.4 * technical_score +
    0.3 * communication_score +
    0.3 * skills_match
)

# ---------- CREATE INPUT DATAFRAME ----------

input_dict = {
    'technical_score': technical_score,
    'communication_score': communication_score,
    'skills_match_percentage': skills_match,
    'years_of_experience': experience,
    'certifications_count': certifications,
    'placement_score': placement_score
}

input_df = pd.DataFrame([input_dict])

# Add categorical columns
input_df['experience_category'] = exp_cat
input_df['skills_level'] = skills_level
input_df['interview_performance'] = performance
input_df['company_tier'] = company_tier
input_df['job_role_match'] = job_role_match
input_df['relocation_willingness'] = relocation

# ---------- ENCODING ----------

input_df = pd.get_dummies(input_df)

# Match training columns
input_df = input_df.reindex(columns=columns, fill_value=0)

# ---------- PREDICTION ----------

if st.button("Predict"):
    
    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.success(f"✅ Candidate likely to ACCEPT the job (Probability: {prob:.2f})")
    else:
        st.error(f"❌ Candidate likely to REJECT the job (Probability: {prob:.2f})")