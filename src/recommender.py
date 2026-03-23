import pandas as pd
import joblib
from functools import lru_cache
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast


from src.config import CLEAN_DATA_PATH, MOOD_MAP, GENRE_BOOST_WEIGHT, TFIDF_VECTORIZER_PATH, TFIDF_MATRIX_PATH

@lru_cache(maxsize=1)
def load_data() -> pd.DataFrame:
    """Carrega o dataset e faz cache na memória. Converte NaNs para None (JSON friendly)."""
    df = pd.read_csv(CLEAN_DATA_PATH)
    return df.where(pd.notnull(df), None)

@lru_cache(maxsize=1)
def load_tfidf_models():
    """Carrega o vectorizer e a matriz pré-computados."""
    vectorizer = joblib.load(TFIDF_VECTORIZER_PATH)
    matrix = joblib.load(TFIDF_MATRIX_PATH)
    return vectorizer, matrix


def recommend_books(mood: str, n: int = 5) -> list[dict]:
    """Recomenda livros com base num mood usando TF-IDF."""
    df = load_data()
    vectorizer, tfidf_matrix = load_tfidf_models()

    mood_query = MOOD_MAP.get(mood, mood)
    mood_vector = vectorizer.transform([mood_query])
    similarities = cosine_similarity(mood_vector, tfidf_matrix).flatten()

    def genre_boost(row):
        try:
            genres = " ".join(ast.literal_eval(row['genres_list']))
        except (ValueError, SyntaxError):
            genres = ""
        return sum(1 for word in mood_query.split() if word.lower() in genres.lower())
    
    boosts = df.apply(genre_boost, axis=1)
    final_score = similarities + (GENRE_BOOST_WEIGHT * boosts)

    top_indices = final_score.argsort()[-n:][::-1]
    results = df.iloc[top_indices]
    
    recommendations = []
    for _, row in results.iterrows():
        recommendations.append({
            "title": row['title'],
            "author": row['author_name'],
            "rating": row['average_rating'],
            "imageURL": row['imageURL'],
            "amazonLink": row['amazonLink'],
            "why": f"Vibe match: {mood} energy ✨"
        })
    
    return recommendations