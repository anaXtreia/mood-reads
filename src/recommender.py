import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast


from src.config import CLEAN_DATA_PATH, MOOD_MAP, GENRE_BOOST_WEIGHT

@st.cache_data
def load_data(path: str = CLEAN_DATA_PATH) -> pd.DataFrame:
    """Carrega o dataset. Cacheado — só lê o CSV uma vez."""
    return pd.read_csv(path)

@st.cache_resource
def build_tfidf(_df: pd.DataFrame):
    """
    Constrói vectorizer e matriz TF-IDF.
    Cacheado — só treina uma vez por sessão.
    O underscore em _df diz ao Streamlit para não tentar serializar o df.
    """
    vectorizer = TfidfVectorizer(stop_words='english')
    matrix = vectorizer.fit_transform(_df['content'])
    return vectorizer, matrix


# --- Carregar dataset pronto ---
df = load_data()

# --- TF-IDF (usar content) ---
vectorizer, tfidf_matrix = build_tfidf(df)

# --- Mood map (keywords por mood) ---
mood_map = MOOD_MAP

# --- Função de recomendação ---
def recommend_books(mood, n=5):
    """
    Recomenda livros com base num mood usando TF-IDF e cosine similarity.

    Args:
        mood: Um de ['leve', 'intenso', 'reflexivo', 'fantasia', 'romântico']
        n: Número de recomendações a devolver

    Returns:
        Lista de dicts com info dos livros, ou string de erro se mood inválido.

    Example:
        >>> recs = recommend_books("fantasia", n=3)
        >>> recs[0]['title']
        'The Name of the Wind'
    """

    if mood not in mood_map:
        return f"Mood '{mood}' não encontrado. Opções: {list(mood_map.keys())}"

    # transformar mood em query
    mood_query = mood_map[mood]

    # vetorizar mood
    mood_vector = vectorizer.transform([mood_query])

    # similaridade básica
    similarities = cosine_similarity(mood_vector, tfidf_matrix).flatten()

    # boost por genres: aumenta score se mood keywords estiverem nos genres
    def genre_boost(row):
        try:
            genres = " ".join(ast.literal_eval(row['genres_list']))
        except (ValueError, SyntaxError):
            genres = ""
        return sum(1 for word in mood_query.split() if word.lower() in genres.lower())
    
    boosts = df.apply(genre_boost, axis=1)
    final_score = similarities + 0.1 * boosts  # peso pequeno para não quebrar TF-IDF

    # top indices
    top_indices = final_score.argsort()[-n:][::-1]

    # montar resultados
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


# --- Teste rápido ---
if __name__ == "__main__":
    mood = "fantasia"
    recs = recommend_books(mood)
    for r in recs:
        print(r)