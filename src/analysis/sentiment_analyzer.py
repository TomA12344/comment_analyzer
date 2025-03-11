import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import pandas as pd
from ..config import SENTIMENT_MODEL, SENTIMENT_THRESHOLD
from ..logger import setup_logger

logger = setup_logger(__name__)

class SentimentAnalyzer:
    """
    Class for performing sentiment analysis on German text using pre-trained models.
    """
    
    def __init__(self, model_name=None):
        """
        Initialize the sentiment analyzer with a pre-trained model.
        
        Args:
            model_name (str): Name of the pre-trained model to use
                Default is defined in config
        """
        self.model_name = model_name or SENTIMENT_MODEL
        self.tokenizer = None
        self.model = None
        self.labels = ["negative", "neutral", "positive"]
            
        # Konfidenz-Schwellenwerte für bessere Klassifizierung
        self.confidence_thresholds = SENTIMENT_THRESHOLD
        
        logger.info(f"SentimentAnalyzer initialisiert mit Modell: {self.model_name}")
    
    def load_model(self):
        """
        Load the pre-trained model and tokenizer.
        """
        if self.tokenizer is None or self.model is None:
            logger.info(f"Lade Sentiment-Modell: {self.model_name}")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
                logger.info("Modell erfolgreich geladen")
            except Exception as e:
                logger.error(f"Fehler beim Laden des Modells: {str(e)}")
                raise
    
    def analyze_text(self, text):
        """
        Analyze the sentiment of a single text.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            dict: Dictionary containing sentiment label and scores
        """
        self.load_model()
        
        # Handle empty or non-string input
        if not isinstance(text, str) or not text:
            logger.warning("Leerer oder ungültiger Text für Sentiment-Analyse")
            return {"sentiment": "neutral", "scores": {label: (1.0 if label == "neutral" else 0.0) for label in self.labels}}
        
        # Tokenize and get model input
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        
        # Make prediction
        try:
            with torch.no_grad():
                outputs = self.model(**inputs)
                scores = torch.nn.functional.softmax(outputs.logits, dim=1)
                scores_dict = {label: float(score) for label, score in zip(self.labels, scores[0])}
            
            # Verbesserte Sentiment-Bestimmung mit Konfidenz-Schwellenwerten
            max_label = max(scores_dict, key=scores_dict.get)
            max_score = scores_dict[max_label]
            
            # Wenn als neutral klassifiziert, aber Konfidenzscore zu niedrig, 
            # wähle stattdessen positiv oder negativ basierend auf den relativen Scores
            if max_label == "neutral" and max_score < self.confidence_thresholds["neutral_to_sentiment"]:
                if scores_dict["positive"] > scores_dict["negative"]:
                    sentiment = "positive"
                else:
                    sentiment = "negative"
            else:
                sentiment = max_label
            
            logger.debug(f"Sentiment für Text: '{text[:30]}...' ist {sentiment}")
            return {
                "sentiment": sentiment,
                "scores": scores_dict
            }
        except Exception as e:
            logger.error(f"Fehler bei der Sentiment-Analyse: {str(e)}")
            raise
    
    def analyze_texts(self, texts):
        """
        Analyze the sentiment of multiple texts.
        
        Args:
            texts (list): List of texts to analyze
            
        Returns:
            list: List of dictionaries containing sentiment labels and scores
        """
        logger.info(f"Analysiere {len(texts)} Texte")
        return [self.analyze_text(text) for text in texts]
    
    def analyze_dataframe(self, df, text_column='cleaned_text', inplace=False):
        """
        Analyze sentiments for texts in a DataFrame.
        
        Args:
            df (pandas.DataFrame): DataFrame containing texts
            text_column (str): Name of the column containing text data
            inplace (bool): Whether to modify the DataFrame in place
            
        Returns:
            pandas.DataFrame: DataFrame with added sentiment columns
        """
        if text_column not in df.columns:
            logger.error(f"Spalte '{text_column}' nicht in DataFrame gefunden")
            raise ValueError(f"Column '{text_column}' not found in DataFrame")
            
        # Make a copy if not inplace
        result_df = df if inplace else df.copy()
        
        logger.info(f"Führe Sentiment-Analyse für {len(df)} Einträge durch")
        
        # Analyze each text
        sentiments = [self.analyze_text(text) for text in result_df[text_column]]
        
        # Add sentiment columns
        result_df['sentiment'] = [s['sentiment'] for s in sentiments]
        result_df['sentiment_scores'] = [s['scores'] for s in sentiments]
        
        # Add individual score columns for easier filtering and aggregation
        for label in self.labels:
            result_df[f'score_{label}'] = [s['scores'][label] for s in sentiments]
        
        sentiment_counts = result_df['sentiment'].value_counts()
        logger.info(f"Sentiment-Verteilung: {dict(sentiment_counts)}")
        
        return result_df