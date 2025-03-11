"""
Zentrale Konfigurationsdatei für das Comment Analyzer Projekt.
Enthält alle konfigurierbaren Parameter und Konstanten.
"""
import os
from pathlib import Path

# Pfad-Konfigurationen
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_PATH = os.path.join(DATA_DIR, "comments.db")

# Spracheinstellungen 
DEFAULT_LANGUAGE = "german"

# Modell-Konfigurationen
SENTIMENT_MODEL = "ssary/XLM-RoBERTa-German-sentiment"
SENTIMENT_THRESHOLD = {
    "neutral_to_sentiment": 0.6  # Schwellenwert für Neutral-Klassifikation
}

# Topic Modeling Konfigurationen
DEFAULT_NUM_TOPICS = 5
DEFAULT_MAX_FEATURES = 1000
DEFAULT_MAX_DF = 0.95
DEFAULT_MIN_DF = 2
DEFAULT_TOP_WORDS = 10

# TextCleaner Konfiguration
TEXT_CLEANING = {
    "remove_urls": True,
    "remove_emails": True,
    "remove_special_chars": True,
    "remove_numbers": True,
    "lowercase": True
}

# Logging Konfiguration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = os.path.join(BASE_DIR, "logs", "comment_analyzer.log")