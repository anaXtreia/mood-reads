from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any

from src.recommender import recommend_books
from src.recommender_semantic import recommend_semantic
from src.config import MOOD_MAP, APP_TITLE, APP_SUBTITLE
app = FastAPI(
    title=APP_TITLE,
    description=APP_SUBTITLE,
    version="1.0.0" 
)


class MoodRequest(BaseModel):
    mood: str
    # O Field ajuda a limitar os valores. Aqui dizemos que n_recs deve ser entre 1 e 20.
    n_recs: int = Field(default=5, ge=1, le=20, description="Número de livros a recomendar")
    use_semantic: bool = Field(default=False, description="Usar o modelo semântico mais avançado?")

@app.get("/")
def home():
    return {
        "message": "Bem-vindo à API do MoodReads! 📚",
        "docs_url": "/docs",
        "available_moods": list(MOOD_MAP.keys())
    }

# 3. Endpoint principal de Recomendação
@app.post("/recommend")
def get_recommendations(request: MoodRequest):
    """
    Recebe um mood e devolve uma lista de livros recomendados.
    """
    # Validar se o mood inserido existe no nosso config
    if request.mood not in MOOD_MAP:
        raise HTTPException(
            status_code=400, 
            detail=f"Mood '{request.mood}' não é válido. Opções: {list(MOOD_MAP.keys())}"
        )
    
    try:
        # Chamar a função certa dependendo do toggle 'use_semantic'
        if request.use_semantic:
            recs = recommend_semantic(request.mood, request.n_recs)
        else:
            recs = recommend_books(request.mood, request.n_recs)
        
        # Devolver a resposta num formato estruturado
        return {
            "query_info": {
                "mood_requested": request.mood,
                "semantic_mode": request.use_semantic,
                "results_count": len(recs)
            },
            "recommendations": recs
        }
        
    except Exception as e:
        # Se algo falhar internamente (ex: erro no pandas ou embeddings), devolvemos um erro 500
        raise HTTPException(status_code=500, detail=f"Erro interno no servidor: {str(e)}")