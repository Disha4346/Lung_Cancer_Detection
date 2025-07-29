# 🫁 Lung Cancer Survival Prediction App

This is a **Streamlit-based web application** that predicts whether a patient is likely to survive lung cancer based on medical history and clinical attributes. It uses a trained machine learning model and interactive input fields for real-time predictions.

---

## 📂 Dataset

The dataset used for training is a structured CSV file containing:

- Patient demographic data (age, gender, country)
- Clinical information (cancer stage, BMI, cholesterol level, etc.)
- Comorbidities (hypertension, asthma, cirrhosis, other cancer)
- Treatment information (type, duration)
- Target: `survived` (0 = No, 1 = Yes)

---

## 🚀 Features

- ✅ Interactive UI using **Streamlit**
- 📊 Two pages:  
  - **Info Page** – Learn about lung cancer and view an educational YouTube video  
  - **Analysis Page** – Input patient data and get survival predictions  
- 🎯 Trained ML model (Random Forest Classifier)
- 🧠 Categorical encoding and scaling handled using pre-saved encoders
- 📁 Session state initialization for persistent inputs

---

## 📁 Folder Structure
lung_cancer_project/
├── app.py
├── dataset_med.csv
├── train_model.py
├── lung_cancer_model.pkl
├── scaler.pkl
├── label_encoders.pkl
├── requirements.txt
└── pages/
├── 1_Info_Page.py
└── 2_Analysis_Page.py


---

## 🛠️ Technologies Used

- **Python 3.10+**
- **Pandas**, **NumPy**
- **Scikit-learn**
- **Streamlit**
- **Joblib**
- **Matplotlib**, **Seaborn** (for visualizations)

---

## 🖥️ Run Locally

1. ✅ Clone the repository:
   ```bash
   git clone https://github.com/Disha4346/Lung_Cancer_Prediction.git

2. ✅ Create a virtual environment:
   ```bash
   python -m venv myenv
    source myenv/bin/activate  # For Linux/macOS
    myenv\Scripts\activate     # For Windows
3. ✅ Install dependencies:
   ```bash
   pip install -r requirements.txt
4. ✅ Run the app:
   ```bash
   streamlit run app.py

📹 Video Demonstration
Watch this quick explainer video:

👩‍⚕️ Disclaimer
This tool is intended for educational and demonstration purposes only.
It is not a substitute for professional medical advice, diagnosis, or treatment.




