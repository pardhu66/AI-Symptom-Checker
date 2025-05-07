from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import re
import difflib
from xgboost import XGBClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

# Load datasets
symptom_df = pd.read_csv("final_dataset2.csv")
remedy_df = pd.read_csv("remedy_dataset.csv")
overview_df = pd.read_csv("disease_overviews_final.csv")
remedy_df['Disease_clean'] = remedy_df['Disease'].str.lower().str.strip()
overview_df['Disease_clean'] = overview_df['Disease'].str.lower().str.strip()

X = symptom_df['Symptoms']
y = symptom_df['Disease']
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Model
model = Pipeline([
    ('tfidf', TfidfVectorizer(
        tokenizer=lambda x: [s.strip() for s in x.split(',')],
        max_features=800,
        ngram_range=(1, 1),
        min_df=5
    )),
    ('clf', XGBClassifier(
        n_estimators=80,
        max_depth=7,
        learning_rate=0.2,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric='mlogloss',
        random_state=42,
        n_jobs=-1,
        verbosity=0
    ))
])
model.fit(X, y_encoded)

# SBERT
all_symptoms = sorted(set(symptom_df['Symptoms'].str.lower().str.split(',').explode().str.strip()))
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
symptom_vecs = sbert_model.encode(all_symptoms)

def get_combined_symptom_matches(user_input, top_n=10):
    user_input = re.sub(r"[^\w\s]", " ", user_input.lower())
    tokens = user_input.split()
    matched_symptoms = set()
    for token in tokens:
        close_matches = difflib.get_close_matches(token, all_symptoms, n=1, cutoff=0.85)
        if close_matches:
            matched_symptoms.add(close_matches[0])

    if len(matched_symptoms) < top_n:
        user_vec = sbert_model.encode([user_input])
        similarities = cosine_similarity(user_vec, symptom_vecs)[0]
        top_indices = similarities.argsort()[::-1]
        for i in top_indices:
            candidate = all_symptoms[i]
            if candidate not in matched_symptoms:
                matched_symptoms.add(candidate)
            if len(matched_symptoms) >= top_n:
                break

    return list(matched_symptoms)[:top_n]

def predict_diseases(selected_symptoms, top_n=5):
    input_symptom_str = ', '.join(selected_symptoms)
    probas = model.predict_proba([input_symptom_str])[0]
    top_indices = np.argsort(probas)[-top_n:][::-1]
    results = []
    for i in top_indices:
        disease = le.inverse_transform([model.named_steps['clf'].classes_[i]])[0]
        prob = f"{probas[i]*100:.1f}%"
        results.append({"disease": disease, "probability": prob})
    return results

def get_disease_info(disease_name):
    name = disease_name.lower().strip()
    match1 = difflib.get_close_matches(name, remedy_df['Disease_clean'].tolist(), n=1, cutoff=0.85)
    match2 = difflib.get_close_matches(name, overview_df['Disease_clean'].tolist(), n=1, cutoff=0.85)

    remedy = precaution = overview = "Not available"
    if match1:
        row = remedy_df[remedy_df['Disease_clean'] == match1[0]].iloc[0]
        remedy = row.iloc[1]
        precaution = row.iloc[2]
    if match2:
        row = overview_df[overview_df['Disease_clean'] == match2[0]].iloc[0]
        overview = row['Overview']
    return {
        "remedy": remedy,
        "precaution": precaution,
        "overview": overview
    }

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/get_symptoms', methods=['POST'])
def get_symptoms():
    user_input = request.json.get("input", "")
    matched = get_combined_symptom_matches(user_input)
    return jsonify(matched)

@app.route('/predict', methods=['POST'])
def predict():
    selected = request.json.get("selectedSymptoms", [])
    if not selected:
        return jsonify([])
    return jsonify(predict_diseases(selected))

@app.route('/details', methods=['POST'])
def details():
    disease = request.json.get("disease", "")
    return jsonify(get_disease_info(disease))
import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Use Render's default port
    app.run(host='0.0.0.0', port=port, debug=True)
