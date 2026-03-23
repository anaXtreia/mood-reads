import pytest
from src.recommender import recommend_books
from src.config import MOOD_MAP

# --- Testes básicos ---

def test_recommend_devolve_n_resultados():
    """Verificar que a função devolve exactamente n livros."""
    recs = recommend_books("fantasia", n=3)
    assert len(recs) == 3

def test_recommend_estrutura_do_dict():
    """Verificar que cada resultado tem os campos obrigatórios."""
    recs = recommend_books("leve", n=1)
    required_keys = {"title", "author", "rating", "imageURL", "amazonLink", "why"}
    assert required_keys.issubset(recs[0].keys())

def test_mood_invalido_devolve_erro():
    """Mood inexistente deve devolver string de erro, não crashar."""
    result = recommend_books("saudade")
    assert isinstance(result, str)

def test_todos_os_moods_funcionam():
    """Cada mood definido no config deve devolver resultados.
    Teste parametrizado — corre uma vez por mood automaticamente."""
    for mood in MOOD_MAP:
        recs = recommend_books(mood, n=1)
        assert isinstance(recs, list), f"Mood '{mood}' falhou"
        assert len(recs) > 0

def test_rating_e_numerico():
    """O rating no resultado deve ser convertível para float."""
    recs = recommend_books("intenso", n=2)
    for book in recs:
        assert isinstance(float(book["rating"]), float)