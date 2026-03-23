# MoodReads 📚✨

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red?logo=streamlit)
![Tests](https://img.shields.io/badge/tests-5%20passed-brightgreen)
![License](https://img.shields.io/badge/license-MIT-green)

> Book recommendation engine based on mood — powered by TF-IDF + Sentence Transformers.

## 🎯 What it does

Given a mood (e.g. "fantasia", "intenso"), MoodReads recommends books from a Goodreads
dataset using two approaches:

- **TF-IDF + Cosine Similarity** — fast, keyword-based baseline
- **Sentence Transformers** — semantic embeddings (all-MiniLM-L6-v2) for deeper context matching

## 🗂 Dataset
Goodreads Books Dataset: ~1k books with title, description, genres, ratings, cover images.

## 🚀 How to run

```bash
git clone https://github.com/anaXtreia/mood-reads
cd moodreads
pip install -r requirements.txt

# 1. clean dataset
python -m src.data_cleaning

# 2. launch app
streamlit run app.py
```

## 🧪 Tests
```bash
pytest tests/ -v
```

## 📐 Architecture
```
CSV → data_cleaning.py → clean dataset
                              ↓
                    TF-IDF vectorizer  OR  Sentence Transformer
                              ↓
                    Cosine Similarity + Genre Boost
                              ↓
                    Streamlit UI (mood selector, book cards)
```

## 🛠 Tech Stack
- Python 3.11 · Pandas · scikit-learn · Sentence-Transformers
- Streamlit · pytest