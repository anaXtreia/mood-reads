import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from functools import lru_cache

from src.recommender import load_data
from src.config import MOOD_MAP, SEMANTIC_EMBEDDINGS_PATH

@lru_cache(maxsize=1)
def load_semantic_model():
    """Carrega o modelo do HuggingFace apenas uma vez para a RAM."""
    return SentenceTransformer('all-MiniLM-L6-v2')

@lru_cache(maxsize=1)
def load_book_embeddings():
    """Carrega os embeddings pré-computados do disco."""
    return np.load(SEMANTIC_EMBEDDINGS_PATH)

def recommend_semantic(mood: str, n: int = 5) -> list[dict]:
    """Recomenda usando os embeddings semânticos pré-computados."""
    df = load_data()
    model = load_semantic_model()
    book_embeddings = load_book_embeddings()

    mood_query = MOOD_MAP.get(mood, mood)
    mood_embedding = model.encode([mood_query])

    scores = cosine_similarity(mood_embedding, book_embeddings).flatten()
    top_indices = scores.argsort()[-n:][::-1]

    recommendations = []
    for i in top_indices:
        row = df.iloc[i]
        recommendations.append({
            "title": row['title'],
            "author": row['author_name'],
            "rating": row['average_rating'],
            "imageURL": row['imageURL'],
            "amazonLink": row['amazonLink'],
            "why": f"Semantic match: {scores[i]:.2f} ✨"
        })

    return recommendations