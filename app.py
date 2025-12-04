# app.py
import streamlit as st
import pickle, os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

MODEL_DIR = "model"
EMB_DIR = "embeddings"

@st.cache_resource
def load_rf_model():
    with open(os.path.join(MODEL_DIR,"rf_model.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(MODEL_DIR,"mlb.pkl"), "rb") as f:
        mlb = pickle.load(f)
    with open(os.path.join(MODEL_DIR,"symptom_list.pkl"), "rb") as f:
        symptom_list = pickle.load(f)
    return model, mlb, symptom_list

@st.cache_resource
def load_embeddings():
    idx = faiss.read_index(os.path.join(EMB_DIR,"medical_faiss.index"))
    with open(os.path.join(EMB_DIR,"medical_chunks.pkl"), "rb") as f:
        chunks = pickle.load(f)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return idx, chunks, model

def predict_disease(selected_symptoms, model, mlb, symptom_list, topk=3):
    # create binary vector aligned with mlb.classes_
    vec = np.zeros(len(mlb.classes_), dtype=int)
    # map provided symptom names to indices - try lowercasing match
    class_list = [s.lower() for s in mlb.classes_]
    for s in selected_symptoms:
        s_norm = s.strip().lower()
        if s_norm in class_list:
            idx = class_list.index(s_norm)
            vec[idx] = 1
        else:
            # try fuzzy matching: simple substring match
            for i,c in enumerate(class_list):
                if s_norm in c or c in s_norm:
                    vec[i] = 1
    probs = None
    try:
        probs = model.predict_proba([vec])[0]
    except:
        # some classifiers might not have predict_proba; fallback to predict and 1.0
        pred = model.predict([vec])[0]
        return [(pred, 1.0)]
    # get topk indices
    top_idx = np.argsort(probs)[::-1][:topk]
    results = [(model.classes_[i], float(probs[i])) for i in top_idx]
    return results

def semantic_search(query, idx, chunks, embed_model, k=3):
    q_emb = embed_model.encode([query], convert_to_numpy=True)
    D,I = idx.search(q_emb, k)
    answers = []
    for i,dist in zip(I[0], D[0]):
        if i < len(chunks):
            answers.append({"text":chunks[i], "score": float(dist)})
    return answers

# Streamlit UI
st.set_page_config(page_title="Medical AI Assistant", layout="wide")
st.title("🩺 Smart Medical AI Assistant")

tabs = st.tabs(["Disease Predictor", "Medical Chatbot"])

with tabs[0]:
    st.header("Disease Predictor")
    model, mlb, symptom_list = load_rf_model()
    st.write("Select symptoms (you can multi-select).")
    selected = st.multiselect("Symptoms", options=symptom_list, default=[])
    if st.button("Predict"):
        if not selected:
            st.warning("Please select at least one symptom.")
        else:
            results = predict_disease(selected, model, mlb, symptom_list, topk=5)
            st.subheader("Top predictions")
            for disease, prob in results:
                st.write(f"- **{disease}** — probability: {prob:.3f}")

with tabs[1]:
    st.header("Medical Chatbot (Knowledge Search)")
    idx, chunks, embed_model = load_embeddings()
    query = st.text_input("Ask a medical question (about symptoms, precautions, treatment)...")
    k = st.slider("Top-k results", 1, 6, 3)
    if st.button("Search"):
        if not query:
            st.warning("Please type a question.")
        else:
            answers = semantic_search(query, idx, chunks, embed_model, k=k)
            st.subheader("Top answers from knowledge base")
            for a in answers:
                st.write(a["text"])
                st.caption(f"distance: {a['score']:.3f}")

st.markdown("---")
