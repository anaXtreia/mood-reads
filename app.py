import streamlit as st
from src.recommender import recommend_books
from src.recommender_semantic import recommend_semantic
import pandas as pd


from src.config import APP_TITLE, APP_SUBTITLE, MOOD_MAP

# --- Config ---
st.set_page_config(page_title=APP_TITLE, layout="wide")

st.title(APP_TITLE)
st.subheader(APP_SUBTITLE)

# --- Sidebar ---
use_semantic = st.sidebar.toggle(
    "🧠 Modo Semântico (Sentence Transformers)",
    help="Mais lento na primeira execução, mas percebe contexto semântico."
)


mood = st.sidebar.selectbox(
    "Qual é o teu mood hoje?",
    list(MOOD_MAP.keys())
)
n_recs = st.sidebar.slider("Número de recomendações", 1, 10, 5)



# --- Mostrar recomendações ---
if st.button("Recomendar"):
    if use_semantic:
        recs = recommend_semantic(mood, n_recs)
    else:
        recs = recommend_books(mood, n_recs)

    for book in recs:
        image_url = book.get('imageURL')
        amazon_link = book.get('amazonLink')

        # Layout da recomendação
        cols = st.columns([1, 3])
        with cols[0]:
            if image_url:
                st.image(image_url, width=150)
        with cols[1]:
            st.markdown(f"### {book['title']}")
            st.write(f"**Autor:** {book['author']}")
            st.write(f"**Rating:** {book['rating']}")
            st.write(f"*{book['why']}*")
            if amazon_link:
                st.markdown(f"[Comprar no Amazon]({amazon_link})", unsafe_allow_html=True)
        st.markdown("---")

# --- Footer ---
st.markdown("💡 Dataset: Goodreads | Glow-up by your queen 👑")

