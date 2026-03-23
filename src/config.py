from pathlib import Path

# --- Paths ---
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "goodreads_books_dataset.csv"
CLEAN_DATA_PATH = DATA_DIR / "clean_books_ready.csv"

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