"""
analyze_nlp_data.py
===================
Pipeline NLP : préparation du texte, fusion des lexiques (custom + VADER FR),
comptages (n-grammes), filtres de polarité et visualisation (wordclouds).
- Prérequis : NLTK 'stopwords' et spaCy 'fr_core_news_sm' doivent être installés (voir setup_env.py si ce n'est pas le cas).
"""

import logging
from collections import Counter
from pathlib import Path
from typing import Iterable, Tuple, Set

import matplotlib.pyplot as plt
import pandas as pd
import spacy
from matplotlib import font_manager
from nltk.corpus import stopwords
from wordcloud import WordCloud
from vaderSentiment_fr.vaderSentiment import SentimentIntensityAnalyzer

logger = logging.getLogger(__name__)

# ----------------------------- Ressources linguistiques ------------------------

# spaCy : tokenizer + tagger + lemmatizer (pas de parser/NER pour perf)
nlp = spacy.load("fr_core_news_sm", disable=["parser", "ner"])

# Stopwords FR + stopwords de contexte (hébergement)
STOPWORDS_FR = set(stopwords.words("french"))
STOPWORDS_METIER = {
    "appartement", "studio", "hôtel", "hotel", "residence", "résidence", "location", "logement", "séjour",
    "appart", "chambre", "logements", "immeuble", "airbnb", "booking", "loc", "curiste", "curistes"
}
STOPWORDS = STOPWORDS_FR | STOPWORDS_METIER

# Lexiques custom base (adaptés au domaine)
POS_LEXICON_CUSTOM = {
    "agréable", "chaleureux", "convivial", "sympa", "parfait", "excellent", "top",
    "impeccable", "super", "nickel", "formidable", "merveilleux", "idéal", "magnifique",
    "recommandé", "paradis", "extra", "génial", "remarquable",
    "spacieux", "lumineux", "moderne", "propre", "calme", "tranquille", "cosy", "chic",
    "confortable", "bien équipé", "fonctionnel", "pratique",
    "accueillant", "disponible", "gentil", "attentionné", "serviable", "à l’écoute",
    "réactif", "discrétion", "politesse", "amabilité",
    "proche", "bien situé", "central", "emplacement idéal", "vue", "panoramique", "lac", "thermes",
    "centre-ville", "commodités", "accessible",
    "correct", "intéressant", "bon marché", "économique", "avantageux",
    "reposant", "détente", "plaisant", "harmonieux", "séjour réussi", "séjour agréable",
}
NEG_LEXICON_CUSTOM = {
    "sale", "poussière", "malpropre", "mal entretenu", "moisi", "dégoutant", "insalubre",
    "odeur", "poubelle", "insectes", "cafards", "taches", "mauvais état",
    "bruyant", "mal isolé", "vétuste", "ancien", "usé", "froid", "chaud", "étroit", "sombre",
    "dégradé", "abîmé", "inconfort", "désagréable",
    "impoli", "froideur", "méchant", "arnaque", "mensonge",
    "injoignable", "indisponible", "retard", "mauvaise foi",
    "loin", "mal situé", "perdu", "inaccessible", "décentré", "isolé", "difficile d’accès",
    "cher", "trop cher", "abusif", "prix élevé", "hors de prix", "décevant",
    "déception", "déçu", "catastrophe", "horrible", "lamentable", "pire", "inadmissible",
    "problème", "panne", "incident", "malchance", "médiocre",
    "wifi lent", "wifi absent", "pas de chauffage", "pas d’eau chaude",
    "literie mauvaise", "matelas dur", "odeur de tabac", "parking compliqué",
}

# Lexiques fusionnés (custom + VADER si dispo)
POS_LEXICON: Set[str] = set()
NEG_LEXICON: Set[str] = set()


# ----------------------------- Lexiques / VADER --------------------------------

def _normalize_terms(terms: Iterable[str]) -> Set[str]:
    """Normalise (minuscule/strip) et supprime les entrées vides."""
    return {str(t).strip().lower() for t in terms if str(t).strip()}


def _load_vader_fr_lexicon() -> Tuple[Set[str], Set[str]]:
    """
    Charge le lexique interne de vaderSentiment-fr via l'instance d'analyseur.
    Retourne (positifs, négatifs). En cas d’indispo, retourne (set(), set()).
    """
    try:
        analyzer = SentimentIntensityAnalyzer()
        lex = getattr(analyzer, "lexicon", None)
        if isinstance(lex, dict) and lex:
            pos = {w for w, s in lex.items() if s > 0}
            neg = {w for w, s in lex.items() if s < 0}
            pos, neg = _normalize_terms(pos), _normalize_terms(neg)

            logger.info("VADER FR détecté via instance: POS=%d, NEG=%d", len(pos), len(neg))
            return pos, neg

        # Fallback module-level
        try:
            from vaderSentiment_fr import vaderSentiment as vs
            module_lex = getattr(vs, "LEXICON", None) or getattr(vs, "lexicon", None)
            if isinstance(module_lex, dict) and module_lex:
                pos = {w for w, s in module_lex.items() if s > 0}
                neg = {w for w, s in module_lex.items() if s < 0}
                pos, neg = _normalize_terms(pos), _normalize_terms(neg)
                logger.info("VADER FR détecté via module: POS=%d, NEG=%d", len(pos), len(neg))
                return pos, neg
        except Exception:
            pass

        logger.warning("VADER FR présent mais lexique introuvable.")
    except Exception as e:
        logger.info("VADER FR non disponible (%s).", e)

    return set(), set()


def build_and_set_polarity_lexicons(
    include_vader: bool = True,
    extra_pos: Iterable[str] = (),
    extra_neg: Iterable[str] = (),
) -> None:
    """
    Construit les lexiques de polarité dans POS_LEXICON / NEG_LEXICON.
    Sources fusionnées : lexique custom + VADER si disponible.
    Supprime les conflits (un même terme dans POS et NEG).
    """
    global POS_LEXICON, NEG_LEXICON

    pos = _normalize_terms(POS_LEXICON_CUSTOM)
    neg = _normalize_terms(NEG_LEXICON_CUSTOM)

    if include_vader:
        vpos, vneg = _load_vader_fr_lexicon()
        pos |= vpos
        neg |= vneg

    pos |= _normalize_terms(extra_pos)
    neg |= _normalize_terms(extra_neg)

    both = pos & neg
    if both:
        logger.info("Conflits POS/NEG (%d) — suppression.", len(both))
        pos -= both
        neg -= both

    POS_LEXICON, NEG_LEXICON = pos, neg
    logger.info("Lexiques fusionnés prêts: POS=%d, NEG=%d", len(POS_LEXICON), len(NEG_LEXICON))


# ----------------------------- Préparation texte / Comptage -------------------

def preprocess_text(text: str) -> list[str]:
    """
    Prétraitement standardisé :
    - minuscule
    - tokenisation spaCy
    - filtrage POS : NOUN + ADJ uniquement
    - lemmatisation
    - suppression stopwords + tokens courts/non alpha
    Retourne une liste de lemmes utiles.
    """
    doc = nlp((text or "").lower())
    kept: list[str] = []
    for tok in doc:
        if tok.is_alpha and len(tok.text) > 1 and tok.pos_ in {"NOUN", "ADJ"}:
            lem = tok.lemma_
            if lem not in STOPWORDS:
                kept.append(lem)
    return kept


def build_counter(reviews: Iterable[str], ngram_range: tuple[int, int] = (1, 1)) -> Counter:
    """
    Construit un compteur de fréquences :
    - unigrams si (1,1)
    - n-grammes si (n,n)
    """
    counts = Counter()
    n = ngram_range[1]
    for review in reviews:
        tokens = preprocess_text(review)
        if n == 1:
            counts.update(tokens)
        else:
            for i in range(len(tokens) - n + 1):
                counts[" ".join(tokens[i : i + n])] += 1
    logger.debug("Top tokens %s : %s", ngram_range, counts.most_common(5))
    return counts


def apply_polarity_filter(counter: Counter, *, keep: str = "pos") -> Counter:
    """
    Filtre un compteur selon une polarité :
    - keep='pos' → retire les termes “négatifs”
    - keep='neg' → retire les termes “positifs”
    Règle n-grammes : bannir si AU MOINS un token appartient au lexique opposé.
    """
    banned = NEG_LEXICON if keep == "pos" else POS_LEXICON
    filtered = Counter()
    for term, freq in counter.items():
        tokens = term.split()
        if any(t in banned for t in tokens):
            continue
        filtered[term] = freq
    return filtered


# ----------------------------- Visualisation / Wordclouds ---------------------

def plot_wordcloud(counter: Counter, title: str, max_words: int = 200) -> None:
    """Affiche un nuage de mots à partir d’un compteur de fréquences."""
    if not counter:
        logger.warning("Pas de données pour %s", title)
        return
    font_path = font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans"))
    wc = WordCloud(
        width=1200,
        height=800,
        background_color="white",
        max_words=max_words,
        collocations=False,
        font_path=font_path,
    ).generate_from_frequencies(dict(counter))

    plt.figure(figsize=(12, 8))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def build_reviews_df(filepath: str | Path) -> pd.DataFrame:
    """
    Construit un DataFrame minimal des avis à partir d’un Excel :
    - rating (float)
    - review (string) = 'French comment' si dispo sinon 'Original comment'
    Filtre les lignes où 'review' est vide.
    """
    df = pd.read_excel(filepath)
    required = {"Rating", "Original comment"}
    if not required.issubset(df.columns):
        raise KeyError(f"Colonnes manquantes. Attendu: {required}, trouvé: {list(df.columns)}")

    if "French comment" not in df.columns:
        df["French comment"] = ""

    def pick(fr, orig):
        fr_s = ("" if pd.isna(fr) else str(fr)).strip()
        return fr_s if fr_s else ("" if pd.isna(orig) else str(orig)).strip()

    reviews = [pick(fr, orig) for fr, orig in zip(df["French comment"], df["Original comment"])]
    out = pd.DataFrame(
        {
            "rating": pd.to_numeric(df["Rating"], errors="coerce"),
            "review": pd.Series(reviews, dtype="string").str.strip(),
        }
    )
    out = out[out["review"].astype(bool)].reset_index(drop=True)
    return out


def inspect_lexicon_coverage(df_reviews: pd.DataFrame, top_k: int = 50) -> dict:
    """
    Mesure la couverture lexicale (types) et renvoie quelques stats + top OOV.
    """
    all_tokens: list[str] = []
    for txt in df_reviews["review"].astype(str):
        all_tokens.extend(preprocess_text(txt))

    total_tokens = len(all_tokens)
    vocab = Counter(all_tokens)
    unique_tokens = len(vocab)

    covered = {w for w in vocab if (w in POS_LEXICON) or (w in NEG_LEXICON)}
    oov = [(w, c) for w, c in vocab.items() if w not in POS_LEXICON and w not in NEG_LEXICON]
    oov.sort(key=lambda x: x[1], reverse=True)

    logger.info("Couverture lexique — tokens totaux: %d, uniques: %d", total_tokens, unique_tokens)
    logger.info(
        "Couverts par lexique: %d (%.1f%% types), OOV: %d (%.1f%% types)",
        len(covered),
        100 * len(covered) / max(1, unique_tokens),
        len(oov),
        100 * len(oov) / max(1, unique_tokens),
    )
    head = oov[:top_k]
    if head:
        logger.info("Top %d OOV (token, freq): %s", top_k, head)
    else:
        logger.info("Pas d'OOV — excellente couverture.")
    return {"vocab_size": unique_tokens, "covered_types": len(covered), "oov_top": head}


def plot_wordclouds_by_rating_with_polarity(
    df_reviews: pd.DataFrame, *, max_words: int = 200, use_bigrams: bool = True
) -> None:
    """
    Génére des nuages de mots basés sur le rating :
    - POSITIF : rating >= 4
    - NEG/NEUTRE : rating <= 3
    Pipeline : preprocess -> ngram -> filtre polarité -> wordcloud
    """
    logger.info("Nuages par rating avec filtre de polarité…")

    pos = df_reviews[df_reviews["rating"] >= 4]
    logger.info("Reviews positives: %d", len(pos))
    if not pos.empty:
        c_pos_uni = build_counter(pos["review"], ngram_range=(1, 1))
        c_pos_uni = apply_polarity_filter(c_pos_uni, keep="pos")
        plot_wordcloud(c_pos_uni, "Nuage — POSITIF (unigrammes, rating ≥ 4)", max_words)
        if use_bigrams:
            c_pos_bi = build_counter(pos["review"], ngram_range=(2, 2))
            c_pos_bi = apply_polarity_filter(c_pos_bi, keep="pos")
            plot_wordcloud(c_pos_bi, "Nuage — POSITIF (bigrammes, rating ≥ 4)", max_words)
    else:
        logger.warning("Aucune review positive.")

    neg = df_reviews[df_reviews["rating"] <= 3]
    logger.info("Reviews négatives/neutres: %d", len(neg))
    if not neg.empty:
        c_neg_uni = build_counter(neg["review"], ngram_range=(1, 1))
        c_neg_uni = apply_polarity_filter(c_neg_uni, keep="neg")
        plot_wordcloud(c_neg_uni, "Nuage — NÉG/NEUTRE (unigrammes, rating ≤ 3)", max_words)
        if use_bigrams:
            c_neg_bi = build_counter(neg["review"], ngram_range=(2, 2))
            c_neg_bi = apply_polarity_filter(c_neg_bi, keep="neg")
            plot_wordcloud(c_neg_bi, "Nuage — NÉG/NEUTRE (bigrammes, rating ≤ 3)", max_words)
    else:
        logger.warning("Aucune review négative/neutre.")