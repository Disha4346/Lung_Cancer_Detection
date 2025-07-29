import streamlit as st

# Set page configuration
st.set_page_config(
    page_title="Lung Cancer Prediction App",
    page_icon="🫁",
    layout="centered"
)

# Title and intro
st.title("🫁 Lung Cancer Prediction App")

st.markdown("""
Welcome to the **Lung Cancer Survival Prediction App**.

This application helps you:
- Understand lung cancer and its treatment
- Predict patient survival probability based on medical records

Navigate using the sidebar:
- 📘 Info Page: Learn about the disease
- 📊 Analysis Page: Enter patient data and get predictions

---

> ℹ️ This app is built for educational and demonstration purposes.
""")

st.markdown("### 🎥 Watch This Video to Learn More About Lung Cancer")

# Embed YouTube video
st.video("https://youtu.be/FE0gxMkD_3A?si=YE-vKEiVIO8eb9ry")
