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
from typing import Iterable, Tuple, Set, List

import matplotlib.pyplot as plt
import pandas as pd
import spacy
from matplotlib import font_manager
from nltk.corpus import stopwords
from wordcloud import WordCloud
from vaderSentiment_fr.vaderSentiment import SentimentIntensityAnalyzer

import random
from sklearn.cluster import KMeans
import numpy as np

logger = logging.getLogger(__name__)

# ----------------------------- Ressources linguistiques ------------------------

# spaCy : tokenizer + tagger + lemmatizer (pas de parser/NER pour perf)
nlp = spacy.load("fr_core_news_lg", disable=["parser", "ner"])

# Stopwords FR + stopwords de contexte (hébergement)
STOPWORDS_FR = set(stopwords.words("french"))
STOPWORDS_METIER = {
    "appartement", "studio", "hôtel", "hotel", "residence", "résidence", "location", "logement", "séjour",
    "appart", "chambre", "logements", "immeuble", "airbnb", "booking", "loc", "curiste", "curistes", "cure"
}
STOPWORDS_EXTRACLEANING = {
    "agréable", "bon", "merci",  "sympathique", "impossible"
}
STOPWORDS = STOPWORDS_FR | STOPWORDS_METIER | STOPWORDS_EXTRACLEANING

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
    "petit", "mauvais"
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
        if tok.ent_type_ == "PER" or tok.pos_ == "PROPN":
            continue
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


def drop_common_terms(a: Counter, b: Counter, min_count: int = 2) -> Tuple[Counter, Counter]:
    """
    Supprime des deux compteurs 'a' et 'b' les termes communs fréquents.
    Un terme est considéré commun s'il apparaît dans les DEUX compteurs
    avec une fréquence >= min_count (dans chacun).
    Retourne deux nouveaux compteurs (copies filtrées).
    """
    # Identifier les termes présents des deux côtés avec une fréquence suffisante
    common = {t for t in (a.keys() & b.keys()) if a[t] >= min_count and b[t] >= min_count}
    if not common:
        return a.copy(), b.copy()

    af = Counter({t: c for t, c in a.items() if t not in common})
    bf = Counter({t: c for t, c in b.items() if t not in common})
    logger.info("Suppression de %d termes communs (min_count=%d)", len(common), min_count)
    return af, bf


# ----------------------------- Clustering automatique de mots (spaCy only) ---

def _embed_words_spacy(words: list[str]) -> dict[str, np.ndarray]:
    """Crée des embeddings de mots via spaCy 

    - Pour chaque mot, on récupère `nlp(w).vector`.
    - On filtre les vecteurs nuls (norme = 0)
    - Retourne un dict {mot: vecteur} pour les mots valides.

    """
    out: dict[str, np.ndarray] = {}
    for w in words:
        doc = nlp(w)
        vec = getattr(doc, "vector", None)
        if vec is not None:
            try:
                if np.linalg.norm(vec) > 0:
                    out[w] = vec
            except Exception:
                # Si `vec` n'est pas un array numpy compatible, on ignore silencieusement
                pass
    logger.info("Embeddings spaCy valides: %d / %d", len(out), len(words))
    return out


def cluster_words(counter: Counter, n_clusters: int = 3, top_n: int = 100) -> dict[int, list[str]]:
    """Clusterise les top_n mots d'un Counter en `n_clusters` thèmes via KMeans.

    Étapes:
      1) Sélection des mots les plus fréquents (pour limiter le bruit)
      2) Embeddings spaCy (filtrage des vecteurs nuls)
      3) KMeans pour regrouper en `n_clusters`

    Retour: dict {cluster_id: [mots]}
    """
    if not counter:
        logger.warning("Clusterisation impossible : Counter vide")
        return {}

    # 1) Top mots
    most_common = counter.most_common(max(1, top_n))
    words = [w for w, _ in most_common]
    logger.info("Clusterisation (spaCy only) sur %d mots (top_n=%d, k=%d)", len(words), top_n, n_clusters)

    # 2) Embeddings spaCy
    emb = _embed_words_spacy(words)
    if len(emb) < n_clusters:
        logger.warning(
            "Embeddings spaCy insuffisants (%d vecteurs valides) pour %d clusters — abandon.",
            len(emb), n_clusters,
        )
        return {}

    X = np.vstack(list(emb.values()))
    w_valid = list(emb.keys())

    # 3) KMeans
    try:
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = km.fit_predict(X)
    except Exception as e:
        logger.error("Échec KMeans: %s", e)
        return {}

    # Organisation des résultats
    grouped: dict[int, list[str]] = {}
    for w, lab in zip(w_valid, labels):
        grouped.setdefault(lab, []).append(w)

    for lab, terms in grouped.items():
        logger.info("Cluster %d (%d mots): %s", lab, len(terms), sorted(terms)[:15])
    return grouped


def cluster_from_reviews(
    df_reviews: pd.DataFrame,
    *,
    n_clusters_pos: int = 3,
    n_clusters_neg: int = 3,
    top_n: int = 100,
) -> tuple[dict[int, list[str]], dict[int, list[str]]]:
    """Pipeline pratique pour clusteriser automatiquement positif et négatif.

    - Sépare les reviews en POS (rating ≥ 4) et NEG/NEUTRE (rating ≤ 3)
    - Construit des compteurs unigrammes avec `build_counter`
    - Applique `apply_polarity_filter` pour garder les termes cohérents avec la polarité
    - Lance `cluster_words`

    Retour: (clusters_pos, clusters_neg)
    """
    pos = df_reviews[df_reviews["rating"] >= 4]
    neg = df_reviews[df_reviews["rating"] <= 3]

    c_pos = build_counter(pos["review"], ngram_range=(1, 1)) if not pos.empty else Counter()
    c_pos = apply_polarity_filter(c_pos, keep="pos") if c_pos else c_pos

    c_neg = build_counter(neg["review"], ngram_range=(1, 1)) if not neg.empty else Counter()
    c_neg = apply_polarity_filter(c_neg, keep="neg") if c_neg else c_neg

    logger.info(
        "Clusterisation auto — POS reviews: %d | NEG/NEUTRE reviews: %d (top_n=%d)",
        len(pos), len(neg), top_n,
    )

    clusters_pos = cluster_words(c_pos, n_clusters=n_clusters_pos, top_n=top_n) if c_pos else {}
    clusters_neg = cluster_words(c_neg, n_clusters=n_clusters_neg, top_n=top_n) if c_neg else {}

    if not clusters_pos:
        logger.warning("Aucun cluster POS généré (vecteurs spaCy insuffisants ou données limitées).")
    if not clusters_neg:
        logger.warning("Aucun cluster NEG généré (vecteurs spaCy insuffisants ou données limitées).")

    return clusters_pos, clusters_neg


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


# --- Colorisation par groupes (couleurs fixes pour certains mots, aléatoire pour le reste)

def _make_color_func_groups(groups: List[tuple[set[str], str]], seed: int = 7):
    """Construit une color_func pour WordCloud qui colore en fixe les mots des
    groupes fournis et laisse une couleur aléatoire (mais stable) pour les autres.

    - groups: liste de tuples (set_mots, "#RRGGBB"). Les mots sont comparés en minuscule.
    - seed: graine pour rendre l'aléatoire reproductible.
    """
    # Dictionnaire mot->couleur
    group_color: dict[str, str] = {}
    for gset, color in (groups or []):
        for w in gset:
            group_color[str(w).lower()] = color

    rng = random.Random(seed)
    palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        "#000000"
    ]

    def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
        w = (word or "").lower()
        if w in group_color:
            return group_color[w]
        #return palette[rng.randrange(len(palette))]
        return "#d3d3d3" 

    return color_func


def plot_wordcloud_colored(
    counter: Counter,
    title: str,
    groups: List[tuple[set[str], str]] | None = None,
    *,
    max_words: int = 200,
) -> None:
    """Affiche un nuage de mots avec couleurs imposées pour les groupes
    et couleurs aléatoires pour les autres termes.
    """
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
        random_state=7,
    ).generate_from_frequencies(dict(counter))

    if groups:
        color_func = _make_color_func_groups(groups, seed=7)
        wc = wc.recolor(color_func=color_func, random_state=7)

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
    df_reviews: pd.DataFrame,
    *,
    max_words: int = 200,
    use_bigrams: bool = True,
    drop_commons: bool = True,
    commons_min_count: int = 2,
    pos_groups: list[tuple[set[str], str]] | None = None,
    neg_groups: list[tuple[set[str], str]] | None = None,
) -> None:
    """
    Génére des nuages de mots basés sur le rating :
    - POSITIF : rating >= 4
    - NEG/NEUTRE : rating <= 3
    Pipeline : preprocess -> ngram -> filtre polarité -> (option) retrait des communs -> wordcloud.
    Si pos_groups/neg_groups sont None, la colorisation est aléatoire (WordCloud par défaut).
    """
    logger.info("Nuages par rating avec filtre de polarité… (drop_commons=%s, min_count=%d)", drop_commons, commons_min_count)

    # Séparations
    pos = df_reviews[df_reviews["rating"] >= 4]
    neg = df_reviews[df_reviews["rating"] <= 3]

    logger.info("Reviews positives: %d | négatives/neutres: %d", len(pos), len(neg))

    # ---------- UNIGRAMMES ----------
    if not pos.empty or not neg.empty:
        # Construire et filtrer par polarité
        c_pos_uni = build_counter(pos["review"], ngram_range=(1, 1)) if not pos.empty else Counter()
        c_pos_uni = apply_polarity_filter(c_pos_uni, keep="pos") if c_pos_uni else c_pos_uni

        c_neg_uni = build_counter(neg["review"], ngram_range=(1, 1)) if not neg.empty else Counter()
        c_neg_uni = apply_polarity_filter(c_neg_uni, keep="neg") if c_neg_uni else c_neg_uni

        # Option : retirer les termes communs fréquents
        if drop_commons and c_pos_uni and c_neg_uni:
            c_pos_uni, c_neg_uni = drop_common_terms(c_pos_uni, c_neg_uni, min_count=commons_min_count)

        # Tracé
        if c_pos_uni:
            plot_wordcloud_colored(c_pos_uni, "Nuage — POSITIF (unigrammes, rating ≥ 4)", groups=pos_groups, max_words=max_words)
        else:
            logger.warning("Aucun terme positif après filtrages (unigrammes).")

        if c_neg_uni:
            plot_wordcloud_colored(c_neg_uni, "Nuage — NÉG/NEUTRE (unigrammes, rating ≤ 3)", groups=neg_groups, max_words=max_words)
        else:
            logger.warning("Aucun terme négatif/neutre après filtrages (unigrammes).")

    # ---------- BIGRAMMES ----------
    if use_bigrams and (not pos.empty or not neg.empty):
        c_pos_bi = build_counter(pos["review"], ngram_range=(2, 2)) if not pos.empty else Counter()
        c_pos_bi = apply_polarity_filter(c_pos_bi, keep="pos") if c_pos_bi else c_pos_bi

        c_neg_bi = build_counter(neg["review"], ngram_range=(2, 2)) if not neg.empty else Counter()
        c_neg_bi = apply_polarity_filter(c_neg_bi, keep="neg") if c_neg_bi else c_neg_bi

        if drop_commons and c_pos_bi and c_neg_bi:
            c_pos_bi, c_neg_bi = drop_common_terms(c_pos_bi, c_neg_bi, min_count=commons_min_count)

        if c_pos_bi:
            plot_wordcloud_colored(c_pos_bi, "Nuage — POSITIF (bigrammes, rating ≥ 4)", groups=pos_groups, max_words=max_words)
        else:
            logger.warning("Aucun terme positif après filtrages (bigrammes).")

        if c_neg_bi:
            plot_wordcloud_colored(c_neg_bi, "Nuage — NÉG/NEUTRE (bigrammes, rating ≤ 3)", groups=neg_groups, max_words=max_words)
        else:
            logger.warning("Aucun terme négatif/neutre après filtrages (bigrammes).")


# ---------- Groupes par défaut (après analyse des données, pour coloriser les nuages) ----------
POS_G1 = {  # Propreté / état irréprochable
    "propre", "impeccable", "irréprochable", "propreté", "nickel", "net",
    "propreté", "soigné"
}

POS_G2 = {  # Localisation / proximité atouts (thermes/lac/centre)
    "thermes", "lac", "proximité", "proche", "pied", "emplacement",
    "centre", "centre-ville", "gare", "bus", "quartier", "cadre", "vue"
}

POS_G3 = {  # Relation / service hôte
    "hôte", "accueillant", "chaleureux", "disponible", "écoute", "serviable",
    "réactif", "gentil", "attention", "attentif", "bienveillance", "accueil"
}

NEG_G1 = {  # Literie / confort de sommeil
    "oreiller", "coussin", "lit", "inconfort", "inconfortable", "désagréable",
    "sommeil", "matelas", "froid", "bruyant", "épais"
}

NEG_G2 = {  # Vétusté / état / équipements
    "vieux", "vétuste", "meuble", "dépareillé", "usé", "étroit", "sombre",
    "petit", "humide", "démodé", "réception", "moustique", "isolé"
}

NEG_G3 = {  # Prix / valeur perçue
    "excessif", "montant", "cher", "prix", "argent", "coût", "onéreux"
}

pos_groups = [(POS_G1, "#1f77b4"), (POS_G2, "#2ca02c"), (POS_G3, "#ff7f0e")]
neg_groups = [(NEG_G1, "#d62728"), (NEG_G2, "#9467bd"), (NEG_G3, "#17becf")]

def check_word_polarity(word: str) -> str:
    """
    Vérifie si un mot est présent dans le lexique positif ou négatif.
    """
    if not isinstance(word, str):
        return "absent"

    normalized = word.strip().lower()
    if not normalized:
        return "absent"

    if normalized in POS_LEXICON:
        return "positive"
    elif normalized in NEG_LEXICON:
        return "negative"
    else:
        return "absent"