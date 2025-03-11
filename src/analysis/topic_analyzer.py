import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from ..config import DEFAULT_NUM_TOPICS, DEFAULT_MAX_FEATURES, DEFAULT_MAX_DF, DEFAULT_MIN_DF, DEFAULT_TOP_WORDS
from ..logger import setup_logger

logger = setup_logger(__name__)

class TopicAnalyzer:
    """
    Class for performing topic analysis on text data using LDA.
    """
    
    def __init__(self, n_topics=None, max_features=None, max_df=None, min_df=None):
        """
        Initialize the topic analyzer.
        
        Args:
            n_topics (int): Number of topics to extract
            max_features (int): Maximum number of features for the CountVectorizer
            max_df (float): Maximum document frequency for the CountVectorizer
            min_df (int): Minimum document frequency for the CountVectorizer
        """
        self.n_topics = n_topics or DEFAULT_NUM_TOPICS
        self.max_features = max_features or DEFAULT_MAX_FEATURES
        self.max_df = max_df or DEFAULT_MAX_DF
        self.min_df = min_df or DEFAULT_MIN_DF
        self.vectorizer = None
        self.lda_model = None
        self.feature_names = None
        self.document_topics = None
        
        logger.info(f"TopicAnalyzer initialisiert mit {self.n_topics} Themen, {self.max_features} Features")
        
    def fit(self, texts, random_state=42):
        """
        Fit the LDA model on the provided texts.
        
        Args:
            texts (list): List of texts (tokenized and joined into strings)
            random_state (int): Random state for reproducibility
            
        Returns:
            self: Returns self for chaining
        """
        # Create document-term matrix
        logger.info("Erstelle Dokument-Term-Matrix")
        try:
            self.vectorizer = CountVectorizer(
                max_df=self.max_df, 
                min_df=self.min_df, 
                max_features=self.max_features
            )
            X = self.vectorizer.fit_transform(texts)
            self.feature_names = self.vectorizer.get_feature_names_out()
            
            # Fit LDA model
            logger.info(f"Trainiere LDA-Modell mit {self.n_topics} Themen...")
            self.lda_model = LatentDirichletAllocation(
                n_components=self.n_topics,
                random_state=random_state,
                max_iter=10,
                learning_method='online'
            )
            self.lda_model.fit(X)
            
            # Get document-topic distribution
            logger.info("Berechne Dokument-Themen-Verteilung")
            self.document_topics = self.lda_model.transform(X)
            
            logger.info("LDA-Modell erfolgreich trainiert")
            return self
        except Exception as e:
            logger.error(f"Fehler beim Training des LDA-Modells: {str(e)}")
            raise
    
    def get_top_words(self, n_top_words=None):
        """
        Get the top words for each topic.
        
        Args:
            n_top_words (int): Number of top words to retrieve
            
        Returns:
            list: List of lists containing top words for each topic
        """
        if self.lda_model is None:
            logger.error("Modell wurde noch nicht trainiert")
            raise ValueError("Model not fitted yet. Call fit() first.")
            
        n_top_words = n_top_words or DEFAULT_TOP_WORDS
        
        top_words = []
        for topic_idx, topic in enumerate(self.lda_model.components_):
            words = [self.feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
            top_words.append(words)
            logger.debug(f"Thema {topic_idx}: {', '.join(words[:5])}...")
            
        return top_words
    
    def get_topic_word_dict(self, n_top_words=20):
        """
        Get a dictionary mapping topic indices to their top words with weights.
        
        Args:
            n_top_words (int): Number of top words to retrieve
            
        Returns:
            dict: Dictionary mapping topic indices to lists of (word, weight) tuples
        """
        if self.lda_model is None:
            logger.error("Modell wurde noch nicht trainiert")
            raise ValueError("Model not fitted yet. Call fit() first.")
            
        topic_word_dict = {}
        for topic_idx, topic in enumerate(self.lda_model.components_):
            sorted_idx = topic.argsort()[:-n_top_words - 1:-1]
            topic_word_dict[topic_idx] = [(self.feature_names[i], topic[i]) for i in sorted_idx]
            
        return topic_word_dict
    
    def get_document_topics(self):
        """
        Get the topic distribution for each document.
        
        Returns:
            numpy.ndarray: Array of shape (n_documents, n_topics)
        """
        if self.document_topics is None:
            logger.error("Modell wurde noch nicht trainiert")
            raise ValueError("Model not fitted yet. Call fit() first.")
            
        return self.document_topics
    
    def assign_main_topic(self, texts=None):
        """
        Assign the main topic to each document.
        
        Args:
            texts (list, optional): New texts to predict topics for
            
        Returns:
            list: List of main topic indices
        """
        try:
            if texts is not None:
                # Transform new texts
                logger.info(f"Berechne Hauptthemen für {len(texts)} neue Texte")
                X = self.vectorizer.transform(texts)
                doc_topics = self.lda_model.transform(X)
            else:
                if self.document_topics is None:
                    logger.error("Modell wurde noch nicht trainiert")
                    raise ValueError("Model not fitted yet. Call fit() first.")
                doc_topics = self.document_topics
                
            # Get the topic with highest probability for each document
            main_topics = doc_topics.argmax(axis=1)
            return main_topics
        except Exception as e:
            logger.error(f"Fehler bei der Zuweisung von Hauptthemen: {str(e)}")
            raise
    
    def analyze_dataframe(self, df, text_column='cleaned_text', tokens_column=None, inplace=False):
        """
        Analyze the topics in the texts of a DataFrame.
        
        Args:
            df (pandas.DataFrame): DataFrame containing texts
            text_column (str): Name of the column containing text data
            tokens_column (str, optional): Name of the column containing tokenized texts
            inplace (bool): Whether to modify the DataFrame in place
            
        Returns:
            pandas.DataFrame: DataFrame with added topic columns
        """
        if text_column not in df.columns:
            logger.error(f"Spalte '{text_column}' nicht in DataFrame gefunden")
            raise ValueError(f"Column '{text_column}' not found in DataFrame")
            
        # Make a copy if not inplace
        result_df = df if inplace else df.copy()
        
        # Prepare texts for LDA
        if tokens_column and tokens_column in result_df.columns:
            # Join tokens into space-separated strings
            logger.info(f"Verwende Token-Spalte '{tokens_column}' für die Themenanalyse")
            texts = [' '.join(tokens) for tokens in result_df[tokens_column]]
        else:
            logger.info(f"Verwende Text-Spalte '{text_column}' für die Themenanalyse")
            texts = result_df[text_column].tolist()
        
        # Fit LDA model
        logger.info(f"Starte Themenanalyse für {len(texts)} Texte")
        self.fit(texts)
        
        # Get document-topic distributions
        doc_topics = self.get_document_topics()
        main_topics = self.assign_main_topic()
        
        # Add topic columns
        result_df['main_topic'] = main_topics
        
        # Add topic distribution columns
        for i in range(self.n_topics):
            result_df[f'topic_{i}_score'] = doc_topics[:, i]
        
        # Log the distribution of main topics
        topic_counts = pd.Series(main_topics).value_counts()
        logger.info(f"Verteilung der Hauptthemen: {dict(topic_counts)}")
        
        return result_df
    
    def generate_topic_wordcloud(self, topic_idx, background_color='white', width=800, height=400):
        """
        Generate a word cloud for a specific topic.
        
        Args:
            topic_idx (int): Index of the topic
            background_color (str): Background color for the word cloud
            width (int): Width of the word cloud image
            height (int): Height of the word cloud image
            
        Returns:
            WordCloud: WordCloud object
        """
        if self.lda_model is None:
            logger.error("Modell wurde noch nicht trainiert")
            raise ValueError("Model not fitted yet. Call fit() first.")
            
        if topic_idx < 0 or topic_idx >= self.n_topics:
            logger.error(f"Ungültiger Themen-Index: {topic_idx}")
            raise ValueError(f"Topic index must be between 0 and {self.n_topics-1}")
        
        # Get word weights for the topic
        topic_words = self.get_topic_word_dict()[topic_idx]
        word_dict = {word: weight for word, weight in topic_words}
        
        logger.info(f"Erstelle Wordcloud für Thema {topic_idx}")
        
        # Generate word cloud
        try:
            wordcloud = WordCloud(
                background_color=background_color,
                width=width,
                height=height
            ).generate_from_frequencies(word_dict)
            
            return wordcloud
        except Exception as e:
            logger.error(f"Fehler bei der Wordcloud-Erstellung: {str(e)}")
            raise
    
    def plot_topic_wordclouds(self, n_cols=2, figsize=None):
        """
        Plot word clouds for all topics.
        
        Args:
            n_cols (int): Number of columns in the plot grid
            figsize (tuple): Figure size
            
        Returns:
            matplotlib.figure.Figure: Figure object
        """
        if self.lda_model is None:
            logger.error("Modell wurde noch nicht trainiert")
            raise ValueError("Model not fitted yet. Call fit() first.")
            
        n_rows = (self.n_topics + n_cols - 1) // n_cols
        
        if figsize is None:
            figsize = (n_cols * 5, n_rows * 4)
        
        logger.info(f"Erstelle Wordclouds für {self.n_topics} Themen")
            
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten()
        
        for i, ax in enumerate(axes):
            if i < self.n_topics:
                wordcloud = self.generate_topic_wordcloud(i)
                ax.imshow(wordcloud)
                ax.set_title(f"Topic {i}")
            ax.axis('off')
        
        plt.tight_layout()
        return fig