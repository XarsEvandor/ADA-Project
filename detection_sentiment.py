import pandas as pd
import numpy as np
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# --- 1. Configuration et Initialisation (s'exécute à l'import) ---

# Liste des features utilisées par le modèle
FEATURES_COLS = [
    'Compound sentiment calculated by VADER',
    'Positive sentiment calculated by VADER',
    'Negative sentiment calculated by VADER',
    'Number of characters',
    'Number of words',
    'Number of sentences',
    'Average word length',
    'Fraction of uppercase characters',
    'Fraction of special characters, such as comma, exclamation mark, etc.',
    'Number of long words (at least 6 characters)',
    'Automated readability index'
]

# Initialisation des outils NLTK (téléchargement silencieux si nécessaire)
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)

# Initialisation de l'analyseur (variable globale au module)
sid = SentimentIntensityAnalyzer()


# --- 2. Fonctions Utilitaires (Internes) ---

def _extraire_features_phrase(phrase):
    """
    Transforme une phrase brute en un DataFrame d'une ligne contenant les features numériques.
    """
    # Nettoyage et tokenization
    words = word_tokenize(phrase)
    sentences = sent_tokenize(phrase)
    num_chars = len(phrase)
    num_words = len(words)
    num_sentences = len(sentences)
    
    # Gestion cas vide
    if num_words == 0: 
        return pd.DataFrame([[0]*len(FEATURES_COLS)], columns=FEATURES_COLS)

    # Calculs VADER
    scores = sid.polarity_scores(phrase)
    
    # Calculs Structurels
    num_upper = sum(1 for c in phrase if c.isupper())
    frac_upper = num_upper / num_chars if num_chars > 0 else 0
    
    num_special = sum(1 for c in phrase if not c.isalnum() and not c.isspace())
    frac_special = num_special / num_chars if num_chars > 0 else 0
    
    num_long_words = sum(1 for w in words if len(w) >= 6)
    avg_word_len = sum(len(w) for w in words) / num_words
    
    # Automated Readability Index
    ari = 4.71 * (len(phrase.replace(" ", "")) / num_words) + 0.5 * (num_words / num_sentences) - 21.43

    data = {
        'Compound sentiment calculated by VADER': scores['compound'],
        'Positive sentiment calculated by VADER': scores['pos'],
        'Negative sentiment calculated by VADER': scores['neg'],
        'Number of characters': num_chars,
        'Number of words': num_words,
        'Number of sentences': num_sentences,
        'Average word length': avg_word_len,
        'Fraction of uppercase characters': frac_upper,
        'Fraction of special characters, such as comma, exclamation mark, etc.': frac_special,
        'Number of long words (at least 6 characters)': num_long_words,
        'Automated readability index': ari
    }
    
    # On renvoie un DataFrame ordonné selon FEATURES_COLS
    return pd.DataFrame([data], columns=FEATURES_COLS)


# --- 3. Fonctions Principales (Pour le Notebook) ---

def train_model(df, target_col='LINK_SENTIMENT'):
    """
    Entraîne le modèle RandomForest à partir du DataFrame global.
    Retourne le modèle entraîné.
    """
    print("🔄 Préparation des données...")
    
    # Vérification que les colonnes existent dans le DF
    missing_cols = [col for col in FEATURES_COLS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Colonnes manquantes dans le DataFrame : {missing_cols}")

    X = df[FEATURES_COLS]
    # Conversion de la cible : si -1 alors 1 (Agresseur/Sarcasme), sinon 0
    y = (df[target_col] == -1).astype(int)

    print("✂️ Split Train/Test...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("🌲 Entraînement du Random Forest...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    score = model.score(X_test, y_test)
    print(f"✅ Modèle entraîné avec succès ! Précision sur le set de test : {score:.2%}")
    
    return model


def test_sentence(model, sentence, threshold=0.4):
    """
    Utilise le modèle pour prédire si une phrase est sarcastique/négative.
    """
    # 1. Extraction des features
    df_features = _extraire_features_phrase(sentence)
    
    # 2. Prédiction (Probabilité de la classe 1)
    proba_negatif = model.predict_proba(df_features)[0][1]
    
    # 3. Décision
    prediction = 1 if proba_negatif >= threshold else 0
    
    # 4. Affichage
    print(f"📝 Phrase : \"{sentence}\"")
    print(f"⚙️ Seuil : {threshold}")
    
    if prediction == 1:
        print(f"🔴 RÉSULTAT : TOXIQUE / NEGATIF (Probabilité : {proba_negatif:.1%})")
    else:
        print(f"🟢 RÉSULTAT : NORMAL / POSITIF (Score toxique : {proba_negatif:.1%})")
    print("-" * 30)