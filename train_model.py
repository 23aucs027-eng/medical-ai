import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

# -----------------------------------------------------
# 1. Load Dataset
# -----------------------------------------------------
DATA_PATH = "data/DiseaseAndSymptoms.csv"

df = pd.read_csv(DATA_PATH)

# Clean column names
symptom_cols = [col for col in df.columns if "Symptom" in col]

# -----------------------------------------------------
# 2. Combine all Symptom_1…Symptom_17 into a list
# -----------------------------------------------------
def merge_symptoms(row):
    symptoms = []
    for col in symptom_cols:
        value = row[col]
        if isinstance(value, str):
            symptoms.append(value.strip().lower())
    return symptoms

df["symptoms"] = df.apply(merge_symptoms, axis=1)

# Only keep disease + merged symptoms columns
df = df[["Disease", "symptoms"]]

print(df.head())

# -----------------------------------------------------
# 3. One-Hot Encode Symptoms
# -----------------------------------------------------
mlb = MultiLabelBinarizer()
X = mlb.fit_transform(df["symptoms"])
y = df["Disease"]

print(f"[INFO] Total unique symptoms: {len(mlb.classes_)}")

# -----------------------------------------------------
# 4. Train RandomForest Model
# -----------------------------------------------------
model = RandomForestClassifier(
    n_estimators=400,
    max_depth=None,
    random_state=42
)

model.fit(X, y)

print("[INFO] Model training complete.")

# -----------------------------------------------------
# 5. Save Model & Encoders
# -----------------------------------------------------
os.makedirs("model", exist_ok=True)

with open("model/rf_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("model/mlb.pkl", "wb") as f:
    pickle.dump(mlb, f)

with open("model/symptom_list.pkl", "wb") as f:
    pickle.dump(mlb.classes_.tolist(), f)

print("[INFO] All model files saved successfully!")
