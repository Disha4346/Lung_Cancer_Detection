import streamlit as st

st.title("📘 Lung Cancer Overview")

st.markdown("""
**Lung Cancer** is one of the leading causes of cancer-related deaths globally. Early diagnosis and personalized treatment can significantly improve patient outcomes.

This app uses a machine learning model to **predict the likelihood of a patient surviving** based on medical and demographic information.

### Why This App?
- Helps doctors assess treatment impact
- Useful for health researchers and planners
- Gives patients and families insight (non-medical advice)

---
**Model Used**: Random Forest  
**Input Features**: Age, Gender, BMI, Smoking Status, Treatment Duration, etc.

ℹ️ For educational and exploratory purposes only.
""")
