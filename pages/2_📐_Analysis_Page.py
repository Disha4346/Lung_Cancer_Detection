import streamlit as st
import joblib
import numpy as np

# Load saved model and tools
model = joblib.load("lung_cancer_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoders.pkl")

st.title("📊 Lung Cancer Survival Prediction")

st.markdown("""
Please enter the patient details below. The model will predict the **likelihood of survival** based on past medical records.
""")

# Initialize session state for inputs
for field in [
    "age", "gender", "country", "cancer_stage", "family_history", "smoking_status",
    "bmi", "cholesterol_level", "hypertension", "asthma", "cirrhosis", "other_cancer",
    "treatment_type", "treatment_duration"
]:
    if field not in st.session_state:
        st.session_state[field] = None

# Input fields using session_state
st.session_state.age = st.number_input("Age", min_value=0, max_value=120, value=st.session_state.age or 50)
st.session_state.gender = st.selectbox("Gender", label_encoders["gender"].classes_, index=0)
st.session_state.country = st.selectbox("Country", label_encoders["country"].classes_, index=0)
st.session_state.cancer_stage = st.selectbox("Cancer Stage", label_encoders["cancer_stage"].classes_, index=0)
st.session_state.family_history = st.selectbox("Family History", label_encoders["family_history"].classes_, index=0)
st.session_state.smoking_status = st.selectbox("Smoking Status", label_encoders["smoking_status"].classes_, index=0)
st.session_state.bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=st.session_state.bmi or 25.0)
st.session_state.cholesterol_level = st.number_input("Cholesterol Level", min_value=50, max_value=300, value=st.session_state.cholesterol_level or 180)
# Ensure binary features are initialized as integers
st.session_state.hypertension = st.session_state.hypertension if isinstance(st.session_state.hypertension, int) else 0
st.session_state.asthma = st.session_state.asthma if isinstance(st.session_state.asthma, int) else 0
st.session_state.cirrhosis = st.session_state.cirrhosis if isinstance(st.session_state.cirrhosis, int) else 0
st.session_state.other_cancer = st.session_state.other_cancer if isinstance(st.session_state.other_cancer, int) else 0

st.session_state.hypertension = st.selectbox("Hypertension", [0, 1], index=[0, 1].index(st.session_state.hypertension))
st.session_state.asthma = st.selectbox("Asthma", [0, 1], index=[0, 1].index(st.session_state.asthma))
st.session_state.cirrhosis = st.selectbox("Cirrhosis", [0, 1], index=[0, 1].index(st.session_state.cirrhosis))
st.session_state.other_cancer = st.selectbox("Other Cancer History", [0, 1], index=[0, 1].index(st.session_state.other_cancer))
st.session_state.treatment_type = st.selectbox("Treatment Type", label_encoders["treatment_type"].classes_, index=0)
st.session_state.treatment_duration = st.number_input("Treatment Duration (in days)", min_value=0, max_value=5000, value=st.session_state.treatment_duration or 300)

# When user clicks "Predict"
if st.button("🔍 Predict Survival"):
    try:
        # Encode categorical features using label encoders
        gender_enc = label_encoders["gender"].transform([st.session_state.gender])[0]
        country_enc = label_encoders["country"].transform([st.session_state.country])[0]
        stage_enc = label_encoders["cancer_stage"].transform([st.session_state.cancer_stage])[0]
        history_enc = label_encoders["family_history"].transform([st.session_state.family_history])[0]
        smoking_enc = label_encoders["smoking_status"].transform([st.session_state.smoking_status])[0]
        treatment_enc = label_encoders["treatment_type"].transform([st.session_state.treatment_type])[0]

        # Construct feature array
        input_data = np.array([[
            st.session_state.age,
            gender_enc,
            country_enc,
            stage_enc,
            history_enc,
            smoking_enc,
            st.session_state.bmi,
            st.session_state.cholesterol_level,
            st.session_state.hypertension,
            st.session_state.asthma,
            st.session_state.cirrhosis,
            st.session_state.other_cancer,
            treatment_enc,
            st.session_state.treatment_duration
        ]])

        # Scale input
        input_scaled = scaler.transform(input_data)

        # Predict
        prediction = model.predict(input_scaled)[0]
        prediction_proba = model.predict_proba(input_scaled)[0][1]

        if prediction == 1:
            st.success(f"✅ The model predicts that the patient is likely to SURVIVE. Confidence: {prediction_proba:.2%}")
        else:
            st.error(f"⚠️ The model predicts that the patient is NOT likely to survive. Confidence: {(1 - prediction_proba):.2%}")

    except Exception as e:
        st.error(f"🚨 Prediction failed: {e}")
