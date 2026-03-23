import pandas as pd
import numpy as np
import joblib
import logging
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer

# Configurar logs para vermos o que está a acontecer
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Definir caminhos (ajusta se a tua estrutura for diferente)
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "clean_books_ready.csv"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True) # Cria a pasta models se não existir

def precompute():
    logger.info("A carregar dataset limpo...")
    df = pd.read_csv(DATA_PATH)
    
    # Substituir nulos para evitar erros nos embeddings/tfidf
    df['content'] = df['content'].fillna("")
    conteudos = df['content'].tolist()

    # --- 1. Pré-computar TF-IDF ---
    logger.info("A treinar modelo TF-IDF...")
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(conteudos)
    
    # Guardar no disco
    joblib.dump(vectorizer, MODELS_DIR / "tfidf_vectorizer.pkl")
    joblib.dump(tfidf_matrix, MODELS_DIR / "tfidf_matrix.pkl")
    logger.info("TF-IDF guardado com sucesso!")

    # --- 2. Pré-computar Embeddings Semânticos ---
    logger.info("A gerar embeddings semânticos (isto pode demorar 1-2 minutos)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    book_embeddings = model.encode(conteudos, show_progress_bar=True)
    
    # Guardar no disco (formato numpy otimizado)
    np.save(MODELS_DIR / "book_embeddings.npy", book_embeddings)
    logger.info("Embeddings semânticos guardados com sucesso!")

if __name__ == "__main__":
    precompute()
    logger.info("Pré-computação concluída. O teu projeto está pronto para ser rápido como um relâmpago! ⚡")