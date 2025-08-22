"""
Script de configuration initiale :
- Télécharge les stopwords français pour NLTK
- Télécharge le modèle spaCy pour la langue française
"""

import nltk
import spacy.cli

def setup_nltk():
    print("Téléchargement des stopwords NLTK (français)...")
    nltk.download("stopwords")

def setup_spacy():
    print("Téléchargement du modèle spaCy fr_core_news_sm...")
    spacy.cli.download("fr_core_news_sm")

if __name__ == "__main__":
    setup_nltk()
    setup_spacy()
    print("Setup terminé : stopwords et modèle spaCy installés.")