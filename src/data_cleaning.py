import pandas as pd
import ast
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

def clean_books_dataset(input_path="data/goodreads_books_dataset.csv",
                        output_path="data/clean_books_ready.csv"):
    """
    Limpa e prepara o dataset de livros para uso em recomendações.
    """

    df = pd.read_csv(input_path)
    logger.info(f"Dataset carregado: {input_path} | shape: {df.shape}")

    # --- Author ---
    df['author_name'] = df['author'].apply(lambda x: ast.literal_eval(x)['name'] if pd.notnull(x) else "Unknown")

    # --- Series ---
    def parse_series(s):
        if pd.isnull(s):
            return pd.Series({"series_title": None, "series_pos": None})
        s_dict = ast.literal_eval(s)
        return pd.Series({"series_title": s_dict.get("title"), "series_pos": s_dict.get("self_position")})
    df[['series_title', 'series_pos']] = df['series'].apply(parse_series)

    # --- Genres ---
    df['genres_list'] = df['genres'].apply(lambda x: ast.literal_eval(x) if pd.notnull(x) else [])

    # --- Ratings and pages ---
    df['average_rating'] = pd.to_numeric(df['rating'], errors='coerce')
    df['ratings_count'] = pd.to_numeric(df['ratings'], errors='coerce')
    df['pages'] = pd.to_numeric(df['pages'], errors='coerce')

    # --- Publication date ---
    df['publicationDate'] = pd.to_datetime(df['publicationDate'], errors='coerce')

    # --- Clean text ---
    df['description'] = df['description'].fillna("").str.replace('\n', ' ').str.strip()
    # Content: juntar title + description + genres (preparação para TF-IDF)
    df['content'] = df['title'] + " " + df['description'] + " " + df['genres_list'].apply(lambda x: " ".join(x))

    # --- Remove duplicados e linhas sem content ---
    df_clean = df.drop_duplicates(subset=['title']).dropna(subset=['content'])

    # --- Selecionar colunas finais ---
    df_clean = df_clean[['title', 'author_name', 'series_title', 'series_pos', 
                         'average_rating', 'ratings_count', 'pages', 
                         'publicationDate', 'genres_list', 'content', 'imageURL', 'amazonLink']]

    # --- Reset index e guardar CSV ---
    df_clean = df_clean.reset_index(drop=True)
    df_clean.to_csv(output_path, index=False)
    logger.info(f"Dataset pronto: {output_path} | shape: {df_clean.shape}")


if __name__ == "__main__":
    clean_books_dataset()