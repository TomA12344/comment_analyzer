import re
import unicodedata
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download necessary NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')


class TextCleaner:
    """
    Class for cleaning and preprocessing text data for analysis.
    """
    
    def __init__(self, language='german'):
        """
        Initialize text cleaner with specified language.
        
        Args:
            language (str): Language for stopwords and tokenization (default: 'german')
        """
        self.language = language
        # Load stop words for the selected language
        try:
            self.stop_words = set(stopwords.words(language))
        except LookupError:
            nltk.download('stopwords')
            self.stop_words = set(stopwords.words(language))
        
    def clean_text(self, text):
        """
        Clean text by removing special characters, URLs and normalize whitespace.
        
        Args:
            text (str): Text to clean
            
        Returns:
            str: Cleaned text
        """
        if not isinstance(text, str) or not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters and numbers
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        
        # Normalize unicode characters
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize(self, text):
        """
        Tokenize text into words using a simple split approach instead of nltk's word_tokenize.
        
        Args:
            text (str): Text to tokenize
            
        Returns:
            list: List of tokens
        """
        # Simple tokenization by splitting on whitespace
        if not text:
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
        return [word for word in tokens if word not in self.stop_words]
    
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
            raise ValueError(f"Column '{text_column}' not found in DataFrame")
            
        # Make a copy to avoid modifying the original dataframe
        result_df = df.copy()
        
        # Apply preprocessing to each text
        processed_data = result_df[text_column].apply(self.preprocess_keep_text)
        
        # Create new columns for cleaned text and tokens
        result_df['cleaned_text'] = processed_data.apply(lambda x: x[0])
        result_df['tokens'] = processed_data.apply(lambda x: x[1])
        
        return result_df