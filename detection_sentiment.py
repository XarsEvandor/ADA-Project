import pandas as pd
import numpy as np
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# --- 1. Configuration and Initialization (executed on import) ---

# List of features used by the model
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

# Initialization of NLTK tools (silent download if required)
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)

# Analyzer initialization (module-level global variable)
sid = SentimentIntensityAnalyzer()


# --- 2. Utility Functions (Internal) ---

def _extraire_features_phrase(phrase):
    """
    Transforms a raw sentence into a single-row DataFrame containing numerical features.
    """
    # Cleaning and tokenization
    words = word_tokenize(phrase)
    sentences = sent_tokenize(phrase)
    num_chars = len(phrase)
    num_words = len(words)
    num_sentences = len(sentences)
    
    # Handle empty case
    if num_words == 0: 
        return pd.DataFrame([[0]*len(FEATURES_COLS)], columns=FEATURES_COLS)

    # VADER sentiment scores
    scores = sid.polarity_scores(phrase)
    
    # Structural feature calculations
    num_upper = sum(1 for c in phrase if c.isupper())
    frac_upper = num_upper / num_chars if num_chars > 0 else 0
    
    num_special = sum(1 for c in phrase if not c.isalnum() and not c.isspace())
    frac_special = num_special / num_chars if num_chars > 0 else 0
    
    num_long_words = sum(1 for w in words if len(w) >= 6)
    avg_word_len = sum(len(w) for w in words) / num_words
    
    # Automated Readability Index (ARI)
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
    
    # Return a DataFrame ordered according to FEATURES_COLS
    return pd.DataFrame([data], columns=FEATURES_COLS)


# --- 3. Main Functions (For Notebook Usage) ---

def train_model(df, target_col='LINK_SENTIMENT'):
    """
    Trains a Random Forest model using the global DataFrame.
    Returns the trained model.
    """
    print("🔄 Preparing data...")
    
    # Check that all required feature columns exist in the DataFrame
    missing_cols = [col for col in FEATURES_COLS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in the DataFrame: {missing_cols}")

    X = df[FEATURES_COLS]
    # Target conversion: -1 becomes 1 (aggressive/sarcastic), otherwise 0
    y = (df[target_col] == -1).astype(int)

    print("✂️ Train/Test split...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("🌲 Training Random Forest model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    score = model.score(X_test, y_test)
    print(f"✅ Model trained successfully! Test set accuracy: {score:.2%}")
    
    return model


def test_sentence(model, sentence, threshold=0.4):
    """
    Uses the trained model to predict whether a sentence is sarcastic or negative.
    """
    # 1. Feature extraction
    df_features = _extraire_features_phrase(sentence)
    
    # 2. Prediction (probability of class 1)
    proba_negatif = model.predict_proba(df_features)[0][1]
    
    # 3. Decision rule
    prediction = 1 if proba_negatif >= threshold else 0
    
    # 4. Output display
    print(f"📝 Sentence: \"{sentence}\"")
    print(f"⚙️ Threshold: {threshold}")
    
    if prediction == 1:
        print(f"🔴 Result: TOXIC / NEGATIVE (Probability: {proba_negatif:.1%})")
    else:
        print(f"🟢 Result: NORMAL / POSITIVE (Toxicity score: {proba_negatif:.1%})")
    print("-" * 30)