"""
main.py
=======
Point d’entrée du projet. Configure les logs et déroule les différentes étapes.
"""

import logging
from pathlib import Path
import os
import pandas as pd
from dotenv import load_dotenv

from src.scrap_data import (
    fetch_data,
    clean_data,
    get_reviews,
    export_to_excel,
    show_data,
)
from src.analyze_nlp_data import (
    build_and_set_polarity_lexicons,
    build_reviews_df,
    inspect_lexicon_coverage,
    plot_wordclouds_by_rating_with_polarity,
    pos_groups,
    neg_groups,
    cluster_from_reviews 
)

# ----------------------------- Logging global ---------------------------------

def setup_logging(level: int = logging.INFO) -> None:
    """ Configuration du logger """
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        )
    else:
        root.setLevel(level)


if __name__ == "__main__":
    setup_logging(logging.INFO)

    # --- Paramètres d’E/S & credentials ---
    DATA_DIR = Path("data")
    OUTPUT_DIR = DATA_DIR / "output"
    INPUT_JSON_DIR = DATA_DIR / "input_json_data"
    PLACES_XLSX = OUTPUT_DIR / "list_places.xlsx"
    REVIEWS_XLSX = OUTPUT_DIR / "list_reviews.xlsx"

    load_dotenv()
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    SERP_API_KEY = os.getenv("SERP_API_KEY")
    """
    # ========== 1) Ingestion, nettoyage & export des Google places ==========
    df_places = fetch_data(INPUT_JSON_DIR)
    df_places = clean_data(df_places)
    export_to_excel(df_places, PLACES_XLSX)

    # ========== 2) Récupération des Google reviews & export ==========
    scrapped = get_reviews(
         df_places,
         google_api_key=GOOGLE_API_KEY,
         serp_api_key=SERP_API_KEY,
         single_call_google=False, # True pour les tests initiaux
         single_call_serp=False,   # True pour les tests initiaux
    )
    if scrapped:
         export_to_excel(pd.DataFrame(scrapped), REVIEWS_XLSX)
   
    # ========== 3) Visualisation simple des Google reviews (hors NLP) ==========
    show_data(REVIEWS_XLSX, top_n=20)
    """
    # ========== 4) Analyse NLP & Wordclouds ==========
    # Prépare les lexiques de polarité (liste custom + VADER)
    build_and_set_polarity_lexicons(include_vader=True)

    # Construit le Dataframe "gold" (rating/review) depuis le fichier Excel des reviews
    df_reviews = build_reviews_df(REVIEWS_XLSX)

    # Mesure la couverture lexicale (diagnostic)
    inspect_lexicon_coverage(df_reviews, top_k=40)

    # Génère les nuages (unigrams + bigrams) par polarité (basé sur le rating)
    plot_wordclouds_by_rating_with_polarity(df_reviews, max_words=200, use_bigrams=False, commons_min_count=1, pos_groups=pos_groups, neg_groups=neg_groups)
    
    # ========== Test non concluant : exploration cluserisation auto ==========
    clusters_pos, clusters_neg = cluster_from_reviews(
        df_reviews, 
        n_clusters_pos=3, 
        n_clusters_neg=3, 
        top_n=100
    )
