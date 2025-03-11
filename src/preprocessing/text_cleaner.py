import re
import unicodedata
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from ..config import DEFAULT_LANGUAGE, TEXT_CLEANING
from ..logger import setup_logger

# Logger für dieses Modul
logger = setup_logger(__name__)

# Download necessary NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    logger.info("Lade erforderliche NLTK-Ressourcen...")
    nltk.download('punkt')
    nltk.download('stopwords')


class TextCleaner:
    """
    Class for cleaning and preprocessing text data for analysis.
    """
    
    def __init__(self, language=None):
        """
        Initialize text cleaner with specified language.
        
        Args:
            language (str): Language for stopwords and tokenization (default from config)
        """
        self.language = language or DEFAULT_LANGUAGE
        self.cleaning_config = TEXT_CLEANING
        
        # Load stop words for the selected language
        try:
            self.stop_words = set(stopwords.words(self.language))
            logger.info(f"TextCleaner initialisiert mit Sprache: {self.language}")
            logger.debug(f"{len(self.stop_words)} Stopwörter geladen")
        except LookupError:
            logger.warning(f"Stopwörter für {self.language} nicht gefunden. Lade NLTK-Ressourcen...")
            nltk.download('stopwords')
            self.stop_words = set(stopwords.words(self.language))
        except ValueError as e:
            logger.error(f"Fehler beim Laden der Stopwörter für Sprache {self.language}: {str(e)}")
            logger.info("Fallback auf englische Stopwörter")
            self.stop_words = set(stopwords.words('english'))
        
    def clean_text(self, text):
        """
        Clean text by removing special characters, URLs and normalize whitespace.
        
        Args:
            text (str): Text to clean
            
        Returns:
            str: Cleaned text
        """
        if not isinstance(text, str) or not text:
            logger.warning("Leerer oder ungültiger Text für die Textbereinigung")
            return ""
        
        # Convert to lowercase if configured
        if self.cleaning_config["lowercase"]:
            text = text.lower()
        
        # Remove URLs if configured
        if self.cleaning_config["remove_urls"]:
            text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove email addresses if configured
        if self.cleaning_config["remove_emails"]:
            text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters if configured
        if self.cleaning_config["remove_special_chars"]:
            text = re.sub(r'[^\w\s]', '', text)
        
        # Remove numbers if configured
        if self.cleaning_config["remove_numbers"]:
            text = re.sub(r'\d+', '', text)
        
        # Normalize unicode characters
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        logger.debug(f"Text gereinigt: '{text[:30]}...'")
        return text
    
    def tokenize(self, text):
        """
        Tokenize text into words using a simple split approach.
        
        Args:
            text (str): Text to tokenize
            
        Returns:
            list: List of tokens
        """
        # Simple tokenization by splitting on whitespace
        if not text:
            logger.debug("Versuch, leeren Text zu tokenisieren")
            return []
        return text.split()
    
    def remove_stopwords(self, tokens):
        """
        Remove stopwords from a list of tokens.
        
        Args:
            tokens (list): List of tokens
            
        Returns:
            list: List of tokens with stopwords removed
        """
        filtered_tokens = [word for word in tokens if word not in self.stop_words]
        logger.debug(f"Stopwörter entfernt: {len(tokens) - len(filtered_tokens)} von {len(tokens)} Tokens entfernt")
        return filtered_tokens
    
    def preprocess(self, text):
        """
        Full preprocessing pipeline: clean text, tokenize, and remove stopwords.
        
        Args:
            text (str): Raw text to process
            
        Returns:
            list: Processed tokens
        """
        cleaned_text = self.clean_text(text)
        tokens = self.tokenize(cleaned_text)
        filtered_tokens = self.remove_stopwords(tokens)
        return filtered_tokens
    
    def preprocess_keep_text(self, text):
        """
        Preprocess text but return both cleaned text and processed tokens.
        
        Args:
            text (str): Raw text to process
            
        Returns:
            tuple: (cleaned_text, processed_tokens)
        """
        cleaned_text = self.clean_text(text)
        tokens = self.tokenize(cleaned_text)
        filtered_tokens = self.remove_stopwords(tokens)
        return cleaned_text, filtered_tokens
    
    def preprocess_dataframe(self, df, text_column='text'):
        """
        Preprocess texts in a DataFrame.
        
        Args:
            df (pandas.DataFrame): DataFrame containing texts
            text_column (str): Name of the column containing text data
            
        Returns:
            pandas.DataFrame: DataFrame with added columns for cleaned text and tokens
        """
        if text_column not in df.columns:
            logger.error(f"Spalte '{text_column}' nicht in DataFrame gefunden")
            raise ValueError(f"Column '{text_column}' not found in DataFrame")
            
        logger.info(f"Vorverarbeitung von {len(df)} Texten in DataFrame")
        
        # Make a copy to avoid modifying the original dataframe
        result_df = df.copy()
        
        # Apply preprocessing to each text
        processed_data = result_df[text_column].apply(self.preprocess_keep_text)
        
        # Create new columns for cleaned text and tokens
        result_df['cleaned_text'] = processed_data.apply(lambda x: x[0])
        result_df['tokens'] = processed_data.apply(lambda x: x[1])
        
        logger.info("Textvorverarbeitung abgeschlossen")
        return result_df