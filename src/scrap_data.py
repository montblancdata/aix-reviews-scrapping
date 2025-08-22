"""
scrap_data.py
==============
Fonctions d’ingestion, de nettoyage, d’appels API (Google / SerpAPI) et d’export.
Ne contient pas de logique NLP (voir analyze_nlp_data.py).
"""

import io
import json
import logging
import os
from pathlib import Path
from typing import Iterable, List, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ----------------------------- Constantes / Config -----------------------------

GOOGLE_PLACE_DETAILS_WEBSERVICE = "https://places.googleapis.com/v1/places"
SERP_API_SEARCHES_WEBSERVICE = "https://serpapi.com/search"

RELEVANT_TYPE_ID = {
    "apartment_rental_agency",
    "bed_and_breakfast",
    "furnished_apartment_building",
    "holiday_apartment_rental",
    "hotel",
    "indoor_lodging",
    "lodge",
    "lodging",
    "self_catering_accommodation",
    "serviced_apartment",
    "tenant_ownership",
    "vacation_appartment",
    "vacation_home_rental_agency",
    "villa",
}

IRRELEVANT_TYPE_IDS = {
    "restaurant",
    "spa",
}

BLACK_LIST_TITLE = {"agence", "conciergerie"}

AIX_LES_BAINS = "Aix-les-Bains"

EMPTY_REGEX = r"^\s*$"


# ----------------------------- Ingestion et Nettoyage --------------------------

def fetch_data(input_dir: Path) -> pd.DataFrame:
    """
    Charge et concatène les résultats JSON en un DataFrame brut.
    - input_dir: répertoire contenant les .json
    Note : les JSON ont été extraits manuellement du playground SerpAPI pour éviter de consommer tous les crédits.
    """
    df = pd.DataFrame()
    files = list(input_dir.glob("*.json"))
    if not files:
        logger.warning("Aucun fichier JSON trouvé dans %s", input_dir)

    for file in files:
        logger.info("Chargement du fichier JSON: %s", file)
        with open(file, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        local_results = data.get("local_results", []) or []
        rows = []
        for r in local_results:
            rows.append(
                {
                    "title": r.get("title", ""),
                    "place_id": r.get("place_id", ""),
                    "rating": r.get("rating", ""),
                    "reviews": r.get("reviews", ""),
                    "type_id": r.get("type_id", ""),
                    "type_ids": r.get("type_ids", ""),
                    "address": r.get("address", ""),
                }
            )
        if rows:
            df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)

    logger.info("Lignes chargées depuis JSON: %d", len(df))
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie le DataFrame des Google places :
    - suppression des doublons
    - filtre type_id dans RELEVANT_TYPE_ID
    - suppression si type_ids contient au moins un autre type que ceux non pertinents (IRRELEVANT_TYPE_IDS)
    - adresse doit contenir AIX_LES_BAINS (na=False)
    - blacklist sur title (mots indésirables)
    - normalisation rating/reviews (vides -> NaN/0)
    """
    if df.empty:
        logger.warning("DataFrame d'entrée vide pour clean_data()")
        return df

    df = df.drop_duplicates(subset=["place_id"])
    df = df[df["type_id"].isin(RELEVANT_TYPE_ID)]
    df = df[df["type_ids"].apply(lambda ids: any(bad not in ids for bad in IRRELEVANT_TYPE_IDS))]
    df = df[df["address"].str.contains(AIX_LES_BAINS, na=False)]
    df = df[~df["title"].apply(lambda t: any(bad in str(t).lower() for bad in BLACK_LIST_TITLE))]

    """
    # avoid FutureWarning: Downcasting behavior in `replace` is deprecated and will be removed in a future version. 
    # To retain the old behavior, explicitly call `result.infer_objects(copy=False)`. 
    # To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)
    """
    pd.set_option("future.no_silent_downcasting", True)
    df["reviews"] = df["reviews"].replace(EMPTY_REGEX, 0, regex=True)
    df["rating"] = df["rating"].replace(EMPTY_REGEX, np.nan, regex=True)

    # Logs de contrôle (niveau DEBUG pour ne pas polluer)
    logger.debug("Aperçu 5 premières lignes:\n%s", df.head().to_string())
    _buf = io.StringIO()
    df.info(buf=_buf, verbose=True)
    logger.debug("Info DataFrame:\n%s", _buf.getvalue())

    return df


# ----------------------------- Appels API Reviews (Google/SerpAPI) --------------------------------

def get_reviews_from_google_api(place_id: str, api_key: str, language: str = "fr") -> List[Tuple]:
    """
    Récupère les reviews via Google Places v1/places.
    Retourne une liste de tuples (author, rating, original_comment, language, french_comment)
    """
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": api_key,
        "X-Goog-FieldMask": "name,displayName,reviews",
    }
    params = {"languageCode": language}
    url = f"{GOOGLE_PLACE_DETAILS_WEBSERVICE}/{place_id}"

    logger.info("Appel Google Places pour place_id=%s", place_id)
    try:
        r = requests.get(url, headers=headers, params=params, timeout=20)
        logger.info("Google status=%s", r.status_code)
        r.raise_for_status()
        data = r.json()
    except requests.RequestException:
        logger.exception("Erreur réseau Google Places (place_id=%s)", place_id)
        return []
    except ValueError:
        logger.error("Réponse non-JSON Google Places (place_id=%s): %s", place_id, r.text[:500])
        return []

    raw = data.get("reviews", []) or []
    out = [
        (
            rv.get("authorAttribution", {}).get("displayName", ""),
            rv.get("rating", -1),
            rv.get("originalText", {}).get("text", ""),
            rv.get("originalText", {}).get("languageCode", ""),
            rv.get("text", {}).get("text", ""),
        )
        for rv in raw
    ]
    logger.info("Reviews récupérées via GoogleAPI: %d (place_id=%s)", len(out), place_id)
    return out


def get_reviews_from_serp_api(place_id: str, serp_api_key: str) -> List[Tuple]:
    """
    Récupère les reviews via SerpAPI avec gestion de la pagination.
    Retourne une liste de tuples (author, rating, original_comment, language, french_comment)
    """
    params = {
        "engine": "google_maps_reviews",
        "hl": "fr",
        "place_id": place_id,
        "api_key": serp_api_key,
    }

    logger.info("Appel SerpAPI pour place_id=%s", place_id)
    all_reviews: List[Tuple] = []
    next_page_token = None

    while True:
        if next_page_token:
            params["next_page_token"] = next_page_token
            logger.debug("Pagination SerpAPI next_page_token=%s", next_page_token)

        r = requests.get(SERP_API_SEARCHES_WEBSERVICE, params=params, timeout=20)
        if r.status_code != 200:
            logger.error("SerpAPI status=%s place_id=%s body=%s", r.status_code, place_id, r.text[:300])
            break

        data = r.json()
        page_reviews = data.get("reviews", []) or []
        mapped = [
            (
                rv.get("user", {}).get("name", ""),
                rv.get("rating", -1),
                rv.get("extracted_snippet", {}).get("original", ""),
                "",
                rv.get("extracted_snippet", {}).get("translated", ""),
            )
            for rv in page_reviews
        ]
        all_reviews.extend(mapped)

        next_page_token = data.get("serpapi_pagination", {}).get("next_page_token")
        if not next_page_token:
            break

    logger.info("Reviews récupérées via SerpAPI: %d (place_id=%s)", len(all_reviews), place_id)
    return all_reviews


def get_reviews(
    places_df: pd.DataFrame,
    google_api_key: str,
    serp_api_key: str,
    single_call_google: bool = True,
    single_call_serp: bool = True,
) -> List[Dict]:
    """
    Récupération des Google reviews.
    Si nombre de Google reviews <= 5 : utilisation de l'API Google
    Si nombre de Google reviews > 5 : utilisation de l'API SerpAPI
    - single_call_google/serp : utile en phase de test pour limiter la consommation des crédits (on teste sur un seul place_id par api)
    Retourne une liste de dict prêts à transformer en DataFrame.
    """
    out: List[Dict] = []
    google_called = False
    serp_called = False

    for row in places_df.itertuples():
        # nombre de reviews <= 5 → Google ; sinon SerpAPI
        provider = "google" if 0 <= row.reviews <= 5 else "serpapi"

        if provider == "google" and single_call_google and google_called:
            logger.info("Google déjà appelé une fois — skip %s", row.title)
            continue
        if provider == "serpapi" and single_call_serp and serp_called:
            logger.info("SerpAPI déjà appelé une fois — skip %s", row.title)
            continue

        if provider == "google":
            google_called = True
            reviews = get_reviews_from_google_api(row.place_id, google_api_key)
        else:
            serp_called = True
            reviews = get_reviews_from_serp_api(row.place_id, serp_api_key)

        for author, rating, original_comment, lang, french_comment in reviews:
            out.append(
                {
                    "Title": row.title,
                    "Type ID": getattr(row, "type_id", None),
                    "Author": author,
                    "Rating": rating,
                    "Original comment": original_comment,
                    "Language": lang,
                    "French comment": french_comment,
                }
            )

    logger.info("Total de reviews collectées: %d", len(out))
    return out


# ----------------------------- Export et Visualisation --------------------------

def export_to_excel(df: pd.DataFrame, path: Path) -> None:
    """Exporte un DataFrame vers Excel (openpyxl)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if df.empty:
        logger.warning("DataFrame vide — rien à exporter vers %s", path)
        return
    df.to_excel(path, index=False, engine="openpyxl")
    logger.info("Export Excel réussi: %s (%d lignes)", path, len(df))


def show_data(excel_path: Path, top_n: int = 20) -> None:
    """
    Visualisation à partir d’un fichier Excel 'list_reviews.xlsx' :
    - (1) Distribution des notes avec barres empilées :
         (bas)  = reviews avec commentaire 
         (haut) = reviews sans commentaire
    - (2) Top N établissements par nombre de reviews
    """
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    if not excel_path.exists():
        logger.error("Fichier introuvable: %s", excel_path)
        return

    df = pd.read_excel(excel_path)
    if df.empty:
        logger.warning("Le fichier de reviews est vide: %s", excel_path)
        return

    required = {"Rating", "Title", "Original comment"}
    if not required.issubset(df.columns):
        logger.error("Colonnes obligatoires manquantes. Attendu: %s | Présent: %s",
                     sorted(required), list(df.columns))
        return

    # Numériser la note
    df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")

    # Colonne booléenne : présence d’un commentaire (Original comment non vide)
    df["_has_comment"] = df["Original comment"].astype(str).str.strip().ne("") & df["Original comment"].notna()

    # Agrégations par note
    grp = df.groupby("Rating", dropna=False)
    total = grp.size()
    with_cmt = grp["_has_comment"].sum(min_count=1).fillna(0).astype(int)
    without_cmt = (total - with_cmt).fillna(0).astype(int)

    # Ordonner l'axe des notes (NaN à la fin si présent)
    ordered_index = sorted([x for x in total.index if pd.notna(x)])
    if total.index.isna().any():
        ordered_index += [np.nan]

    total = total.reindex(ordered_index)
    with_cmt = with_cmt.reindex(ordered_index).fillna(0).astype(int)
    without_cmt = without_cmt.reindex(ordered_index).fillna(0).astype(int)
    x_labels = [str(v) if pd.notna(v) else "NaN" for v in total.index]

    # --- (1) Barres empilées ---
    plt.figure(figsize=(11, 6))
    b1 = plt.bar(x_labels, with_cmt.values, edgecolor="black", label="Avec commentaire")
    b2 = plt.bar(x_labels, without_cmt.values, bottom=with_cmt.values, edgecolor="black", label="Sans commentaire")

    plt.title("Distribution des notes (avec et sans commentaire)")
    plt.xlabel("Note")
    plt.ylabel("Nombre d'avis")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()

    # Labels 
    # Code commenté car visuel "surchargé"
    """
    for rect, val in zip(b1, with_cmt.values):
        if val > 0:
            plt.text(rect.get_x() + rect.get_width()/2, rect.get_y() + rect.get_height()/2,
                     f"{val}", ha="center", va="center", fontsize=9)
    for rect, val, base in zip(b2, without_cmt.values, with_cmt.values):
        if val > 0:
            plt.text(rect.get_x() + rect.get_width()/2, base + val/2,
                     f"{val}", ha="center", va="center", fontsize=9)
    for i, tot in enumerate((with_cmt + without_cmt).values):
        if tot > 0:
            plt.text(i, tot + max(1, tot*0.015), f"Total {tot}", ha="center", va="bottom", fontsize=9)
    """
    plt.tight_layout()
    plt.show()

    # --- (2) Top N établissements ---
    top = df["Title"].value_counts().head(top_n)
    plt.figure(figsize=(12, 8))
    plt.barh(top.index[::-1], top.values[::-1], edgecolor="black")
    plt.title(f"Top {top_n} des établissements par nombre de reviews")
    plt.xlabel("Nombre de reviews")
    plt.ylabel("Établissement")
    plt.grid(axis="x", alpha=0.75)
    plt.tight_layout()
    plt.show()