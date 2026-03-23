from pathlib import Path

# --- Paths ---
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "goodreads_books_dataset.csv"
CLEAN_DATA_PATH = DATA_DIR / "clean_books_ready.csv"

# NOVOS CAMINHOS PARA OS MODELOS
MODELS_DIR = BASE_DIR / "models"
TFIDF_VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.pkl"
TFIDF_MATRIX_PATH = MODELS_DIR / "tfidf_matrix.pkl"
SEMANTIC_EMBEDDINGS_PATH = MODELS_DIR / "book_embeddings.npy"

# --- Modelo ---
TFIDF_STOP_WORDS = "english"
GENRE_BOOST_WEIGHT: float = 0.1
DEFAULT_N_RECOMMENDATIONS: int = 5

# --- Moods ---
MOOD_MAP: dict[str, str] = {
    "leve":       "funny light romance humor feel-good",
    "intenso":    "dark thriller mystery crime suspense",
    "reflexivo":  "philosophy life deep emotional introspective",
    "fantasia":   "magic fantasy adventure dragons",
    "romântico":  "love romance relationship emotional"
}

# --- UI ---
APP_TITLE = "📚 MoodReads"
APP_SUBTITLE = "Escolhe o teu mood e recebe recomendações de livros"