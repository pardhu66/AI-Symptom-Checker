# 🌿 AI-Based Symptom Tracker and Herbal Remedy Recommendation

## 📌 Project Overview
AI-Based Symptom Tracker and Herbal Remedy Recommendation is a personalized healthcare assistant that predicts probable diseases from free-text symptom descriptions and suggests natural herbal remedies with safety precautions. It leverages a powerful combination of **machine learning**, **semantic search (SBERT)**, and **web-based interactivity** to support users in self-screening without requiring deep medical knowledge.

Unlike static rule-based systems, this platform interprets *natural language inputs*, corrects spelling errors, and returns interpretable predictions with high accuracy. Herbal remedies and educational disease overviews help users take the first informed step toward recovery.

---

## 📋 Table of Contents
- Features
- Previous Works in the Field
- Datasets Used
- System Architecture
- Model Details
- Installation Guide
- Usage Instructions
- Screenshots
- Future Scope
- Limitations
- Contributors
- License
- Acknowledgements

---

## ✅ Features
### 🧠 SBERT-Based Symptom Matching
- Understands colloquial, informal, or noisy symptom inputs
- Handles spelling errors (e.g., "fevar" → "fever")
- Uses Sentence-BERT + fuzzy logic to match top-10 symptoms from dataset

### 🔍 Disease Prediction with XGBoost
- Predicts **Top 5 diseases** ranked by probability
- Learns symptom combinations and underlying patterns (not just matching)

### 🌱 Herbal Remedies & Precautions
- Maps herbal treatment and safety advice from a curated dataset
- Matches diseases using fuzzy score-based matching to improve accuracy

### 📚 Disease Overview Integration
- Adds human-readable explanations for each predicted disease
- Helps educate users, reducing fear or confusion about conditions

### 💻 Interactive Web UI
- Dark-mode responsive web design
- Highlights symptom suggestions with toggles
- Interactive cards for disease results and expandable remedy sections

---

## 🔍 Previous Works in the Field
Previous symptom-disease prediction systems have typically relied on either:
- **Rule-based matching** (e.g., keyword maps)
- **Simple classifiers without contextual understanding**
- **Black-box systems without interpretability**

Examples include:
- **MySymptoms** app – uses fixed logic, no ML, no synonym/spelling support
- **WebMD Symptom Checker** – guided form inputs, lacks free-text understanding
- **Early research** – accuracy limited, cannot process natural language

**What makes our system better?**
✅ Accepts free-text symptom input (like a real conversation)
✅ Handles synonyms and spelling mistakes using SBERT + fuzzy logic
✅ Interpretable ML with confidence scores (XGBoost)
✅ Adds herbal remedies, overview, and precautions — a holistic layer not seen in most existing works

---

## 📊 Dataset Summary
### 1. `final_dataset2.csv`
- 13,000+ rows
- 408 unique symptoms
- 133 distinct diseases

### 2. `remedy_dataset.csv`
- Maps diseases to herbal treatments and related precautions

### 3. `disease_overviews_final.csv`
- Medical overviews for each disease to support education

---

## 🧱 System Architecture
```
User Input (Free Text)
      ↓
Symptom Matching Engine (SBERT + Fuzzy Matcher)
      ↓
Symptom Selector (User selects from Top-10)
      ↓
Model (TF-IDF + XGBoost)
      ↓
Top-5 Disease Predictions
      ↓
Herbal Remedy Matcher + Disease Overview Fetch
      ↓
Interactive Display on Web Interface
```
![ChatGPT Image May 6, 2025, 08_57_44 PM](https://github.com/user-attachments/assets/9357c92c-5251-442f-9bda-66942c727027)

---

## 🤖 Model Details
- **Vectorization**: TF-IDF 
- **Classifier**: XGBoost
  - `n_estimators=80`, `max_depth=7`, `learning_rate=0.2`
- **Semantic Matcher**: `SentenceTransformer('all-MiniLM-L6-v2')`
- **Spelling/Similarity Handling**: `difflib.get_close_matches()`

---

## ⚙️ Installation Guide
```bash
# Clone repository
$ git clone https://github.com/pardhu66/AI-Symptom-Checker.git
$ cd symptom-tracker

# Create virtual environment
$ python3 -m venv venv
$ source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
$ pip install -r requirements.txt

# Run the application
$ python app.py
```

---

## 🧪 Usage Instructions
1. Enter symptoms as a sentence (e.g., “I have chills and sore throat”)
2. Select matched symptoms from intelligent suggestions
3. Click “Predict Diseases” to view top 5 predicted conditions
4. Click a disease to reveal:
   - 📘 Overview
   - 🌿 Herbal Remedy
   - ⚠️ Safety Precaution

---

## 🖼️ Screenshots
- ✅ Smart Symptom Matcher Interface
- ✅ Top 5 Disease Predictor
- ✅ Remedy + Precaution + Overview Panels

---

## 🚀 Future Scope
- 🌍 Add multilingual symptom processing (e.g., Spanish, Hindi)
- 🤝 Integrate real-time medical APIs (e.g., MedlinePlus, WHO)
- 📱 Add mobile-first interface & voice input for accessibility
- 🔬 Integrate user demographics (age, gender, history) for personalization
- 🧠 Integrate LLMs (GPT-style) to enhance interpretability
- 🗃️ Secure cloud storage of inputs and result history

---

## 🚧 Limitations
- ❌ No personal history (age/gender/duration not factored)
- ❌ Herbal remedies are general, not patient-specific
- ❌ Cannot currently disambiguate identical symptom overlaps in rare diseases
- ❌ Not designed to replace professional medical advice
- ❌ Model accuracy can degrade on edge-case inputs or vague terms

---

## 👨‍💻 Contributor
**Pardha Saradhi Reddy Golamaru**

---

## 📄 License
This project is licensed under the [MIT License](LICENSE).

---

## 🙏 Acknowledgements
- UCI Machine Learning Repository
- Sentence-BERT & Hugging Face
- Open-source herbal remedy datasets
- XGBoost & scikit-learn
- Bootstrap 5 and Flask web framework
