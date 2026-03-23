import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from src.recommender import load_data
from src.config import CLEAN_DATA_PATH, MOOD_MAP

@st.cache_resource
def load_semantic_model():
    """
    Carrega o modelo all-MiniLM-L6-v2.
    Pequeno (80MB), rápido, e muito bom para similaridade semântica.
    Só descarrega uma vez — depois fica em cache local.
    """
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_data
def build_embeddings(_model, contents: list[str]) -> np.ndarray:
    """Gera embeddings para todos os livros. Cacheado — só corre uma vez."""
    return _model.encode(contents, show_progress_bar=True)

def recommend_semantic(mood: str, n: int = 5) -> list[dict]:
    """Recomenda usando embeddings semânticos em vez de TF-IDF."""
    df = load_data(CLEAN_DATA_PATH)
    model = load_semantic_model()

    # Embeddings dos livros (cacheados)
    book_embeddings = build_embeddings(model, df['content'].tolist())

    # Embedding do mood query
    mood_query = MOOD_MAP.get(mood, mood)
    mood_embedding = model.encode([mood_query])

    # Similaridade
    scores = cosine_similarity(mood_embedding, book_embeddings).flatten()
    top_indices = scores.argsort()[-n:][::-1]

    return [
        {
            "title": df.iloc[i]['title'],
            "author": df.iloc[i]['author_name'],
            "rating": df.iloc[i]['average_rating'],
            "imageURL": df.iloc[i]['imageURL'],
            "amazonLink": df.iloc[i]['amazonLink'],
            "why": f"Semantic match: {scores[i]:.2f} ✨"
        }
        for i in top_indices
    ]