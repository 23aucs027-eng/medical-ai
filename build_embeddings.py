# build_embeddings.py
import pandas as pd
import pickle
import os
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np

# ---------------------------------------------------------
# 1. Load datasets
# ---------------------------------------------------------
desc_df = pd.read_csv("data/symptom_Description.csv")
prec_sym_df = pd.read_csv("data/symptom_precaution.csv")
prec_dis_df = pd.read_csv("data/Disease precaution.csv")

def clean(x):
    if pd.isna(x):
        return ""
    return str(x).strip()

knowledge_base = []

# ---------------------------------------------------------
# 2. Process symptom_Description.csv
# Expected columns: Disease, Description
# ---------------------------------------------------------
if set(["Disease", "Description"]).issubset(desc_df.columns):
    for _, row in desc_df.iterrows():
        disease = clean(row["Disease"])
        desc = clean(row["Description"])
        if desc:
            knowledge_base.append(f"Disease: {disease}. Description: {desc}")
else:
    raise ValueError("symptom_Description.csv must contain 'Disease' and 'Description' columns")

# ---------------------------------------------------------
# 3. Process symptom_precaution.csv (your real structure)
# Columns: Disease, Precaution_1...Precaution_4
# ---------------------------------------------------------
prec_cols = [c for c in prec_sym_df.columns if c.startswith("Precaution")]

for _, row in prec_sym_df.iterrows():
    disease = clean(row["Disease"])
    precautions = [clean(row[c]) for c in prec_cols if clean(row[c])]
    if precautions:
        text = f"Disease: {disease}. Precautions: {' | '.join(precautions)}"
        knowledge_base.append(text)

# ---------------------------------------------------------
# 4. Process Disease precaution.csv (same structure)
# Columns: Disease, Precaution1...Precaution4
# ---------------------------------------------------------
prec_cols = [c for c in prec_dis_df.columns if c.lower().startswith("precaution")]

for _, row in prec_dis_df.iterrows():
    disease = clean(row["Disease"])
    precautions = [clean(row[c]) for c in prec_cols if clean(row[c])]
    if precautions:
        text = f"Disease: {disease}. Additional precautions: {' | '.join(precautions)}"
        knowledge_base.append(text)

# ---------------------------------------------------------
# 5. Remove duplicates
# ---------------------------------------------------------
knowledge_base = list(dict.fromkeys(knowledge_base))

print(f"[INFO] Total knowledge chunks prepared: {len(knowledge_base)}")

# ---------------------------------------------------------
# 6. Build embeddings using MiniLM
# ---------------------------------------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(knowledge_base, convert_to_numpy=True, show_progress_bar=True)

# ---------------------------------------------------------
# 7. Build FAISS index
# ---------------------------------------------------------
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings)

# ---------------------------------------------------------
# 8. Save everything
# ---------------------------------------------------------
os.makedirs("embeddings", exist_ok=True)

faiss.write_index(index, "embeddings/medical_faiss.index")

with open("embeddings/medical_chunks.pkl", "wb") as f:
    pickle.dump(knowledge_base, f)

print("[INFO] Embeddings + FAISS index saved successfully!")
