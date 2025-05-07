# 🌿 AI-Based Symptom Tracker and Herbal Remedy Recommendation

## 📌 Project Overview

AI-Based Symptom Tracker and Herbal Remedy Recommendation is an intelligent web-based healthcare assistant that predicts probable diseases from free-text symptom input and offers natural, side-effect-free herbal remedies and preventive advice. This system uses NLP, machine learning (XGBoost), and Sentence-BERT (SBERT) embeddings to process and understand user inputs and provide actionable, explainable insights.

Built with real patient data, it supports intelligent symptom interpretation (including spelling mistakes and synonyms), disease classification, and herbal remedy mapping — empowering users to self-screen responsibly and naturally.

---

## 📋 Table of Contents

* Features
* Datasets Used
* System Architecture
* Model Details
* Installation Guide
* Usage Instructions
* Screenshots
* Future Scope
* Limitations
* Contributors
* License

---

## ✅ Features

### 🧠 SBERT-Based Symptom Matching

* Understands symptom descriptions even in natural, informal language
* Matches to top 10 dataset symptoms using semantic embeddings and fuzzy matching
* Accepts spelling mistakes (e.g., "fevar" → "fever")

### 🔍 Disease Prediction with XGBoost

* Predicts top 5 likely diseases based on selected symptoms
* Confidence scores shown for interpretability

### 🌱 Herbal Remedies + Precaution Mapping

* Displays matched remedies and precautions for each predicted disease
* Integrates fuzzy disease name matching with a curated herbal dataset

### 📚 Disease Overview Integration

* Pulls disease summaries from a separate overview file for medical context
* Helps users understand what the condition means

### 💻 Interactive Web Interface

* Symptom input field and smart suggestion system
* Dynamic UI to click disease and reveal remedy, overview, and precautions
* Styled with a medical background for professional look

---

## 📊 Dataset Summary

### 1. `final_dataset2.csv`

* 13,000+ rows
* 408 unique symptoms, 133 diseases

### 2. `remedy_dataset.csv`

* Herbal remedy and precaution mapping for each disease

### 3. `disease_overviews_final.csv`

* High-level disease summaries used for educational explanations

---

## 🧱 System Architecture

1. **User Input**: Free-text symptom description
2. **SBERT + Fuzzy Matcher**: Suggest top 10 relevant symptoms
3. **Symptom Selection**: User picks from matches
4. **Model**: XGBoost classifier using TF-IDF vectors of selected symptoms
5. **Prediction**: Top 5 diseases with confidence levels
6. **Remedy Matching**: Maps remedies, precautions, and overview using fuzzy logic
7. **Output**: Displayed interactively on the web app

---

## 🤖 Model Details

* **Vectorizer**: TF-IDF (unigrams)
* **Classifier**: XGBoost (n\_estimators=80, max\_depth=7)
* **Semantic Engine**: SentenceTransformer ('all-MiniLM-L6-v2')
* **Fuzzy Match**: `difflib.get_close_matches()`

---

## ⚙️ Installation Guide

```bash
# Clone repository
$ git clone https://github.com/yourusername/symptom-tracker.git
$ cd symptom-tracker

# Create virtual environment
$ python3 -m venv venv
$ source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
$ pip install -r requirements.txt

# Run Flask app
$ python app.py
```

Ensure your folder structure:

```
symptom-tracker/
├── app.py
├── templates/
│   └── index.html
├── final_dataset2.csv
├── remedy_dataset.csv
├── disease_overviews_final.csv
```

---

## 🧪 Usage

1. Enter your symptoms in a natural sentence (e.g., "My head hurts and I feel feverish")
2. Select the best-matched symptoms
3. Click “Predict Diseases”
4. View the top 5 diseases
5. Click any disease to view:

   * 🔍 Overview
   * 🌿 Herbal Remedy
   * ⚠️ Precaution

---

## 🖼️ Screenshots

* Smart Symptom Matching Interface
* Disease Prediction with Confidence
* Herbal Remedy + Overview Panel

(Attach screenshots if needed)

---

## 🚀 Future Scope

* Add multilingual symptom inputs
* Integrate real-time medical API (e.g., MedlinePlus)
* Patient demographics for personalized results
* Integrate advanced LLMs for context-aware explanations
* Export/share results securely
* Deploy to Hugging Face or Streamlit Cloud

---

## 🚧 Limitations

* No patient history or age/gender input
* Accuracy may vary with ambiguous phrasing
* Herbal remedies are general, not personalized
* Does not replace professional medical advice

---

## 👨‍💻 Contributors

Pardha Saradhi Reddy Golamaru

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgements

* UCI symptom-disease dataset
* SentenceTransformers
* XGBoost team
* Ayurvedic and Home Remedy community databases
* Flask and Bootstrap UI libraries

> 💡 **Pro Tip:** If you found this project useful, star it ⭐ on GitHub and share your feedback!
