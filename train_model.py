import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ========== Step 1: Load Dataset ==========
print("🔍 Loading dataset...")
df = pd.read_csv("dataset_med.csv")

print("\n📊 Preview of dataset:")
print(df.head())
print("\nℹ️ Dataset info:")
print(df.info())
print("\n❓ Null values before processing:")
print(df.isnull().sum())

# ========== Step 2: Parse Dates ==========
print("\n🗓️ Parsing dates (format: %d-%m-%Y)...")
df['diagnosis_date'] = pd.to_datetime(df['diagnosis_date'], format='%d-%m-%Y', errors='coerce')
df['end_treatment_date'] = pd.to_datetime(df['end_treatment_date'], format='%d-%m-%Y', errors='coerce')

# ========== Step 3: Feature Engineering ==========
df['treatment_duration'] = (df['end_treatment_date'] - df['diagnosis_date']).dt.days

# Drop unnecessary columns
df.drop(columns=['id', 'diagnosis_date', 'end_treatment_date'], inplace=True)

# Check missing values before drop
print("\n🔎 Missing values before dropna():")
print(df.isnull().sum())

# Drop rows with missing data
df.dropna(inplace=True)
print(f"\n✅ Shape after dropna(): {df.shape}")

# ========== Step 4: Encode Categorical Variables ==========
print("\n🔁 Encoding categorical columns...")
categorical_cols = df.select_dtypes(include='object').columns
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# ========== Step 5: Split Data ==========
X = df.drop(columns=['survived'])
y = df['survived']

print(f"\n📊 Feature matrix shape: {X.shape}, Target vector shape: {y.shape}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ========== Step 6: Feature Scaling ==========
print("\n📐 Scaling features...")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ========== Step 7: Train Model ==========
print("\n🚀 Training Random Forest Classifier...")
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# ========== Step 8: Evaluate Model ==========
y_pred = model.predict(X_test)

print("\n📈 Model Performance:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# ========== Step 9: Plot Feature Importances ==========
importances = model.feature_importances_
features = df.drop(columns=['survived']).columns

plt.figure(figsize=(10,6))
sns.barplot(x=importances, y=features)
plt.title("Feature Importance")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.show()

# ========== Step 10: Save Artifacts ==========
print("\n💾 Saving model, scaler, and label encoders...")
joblib.dump(model, "lung_cancer_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")
print("✅ All files saved successfully!")
