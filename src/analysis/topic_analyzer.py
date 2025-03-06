import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt
from wordcloud import WordCloud

class TopicAnalyzer:
    """
    Class for performing topic analysis on text data using LDA.
    """
    
    def __init__(self, n_topics=5, max_features=1000, max_df=0.95, min_df=2):
        """
        Initialize the topic analyzer.
        
        Args:
            n_topics (int): Number of topics to extract
            max_features (int): Maximum number of features for the CountVectorizer
            max_df (float): Maximum document frequency for the CountVectorizer
            min_df (int): Minimum document frequency for the CountVectorizer
        """
        self.n_topics = n_topics
        self.max_features = max_features
        self.max_df = max_df
        self.min_df = min_df
        self.vectorizer = None
        self.lda_model = None
        self.feature_names = None
        self.document_topics = None
        
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
        self.vectorizer = CountVectorizer(
            max_df=self.max_df, 
            min_df=self.min_df, 
            max_features=self.max_features
        )
        X = self.vectorizer.fit_transform(texts)
        self.feature_names = self.vectorizer.get_feature_names_out()
        
        # Fit LDA model
        self.lda_model = LatentDirichletAllocation(
            n_components=self.n_topics,
            random_state=random_state,
            max_iter=10,
            learning_method='online'
        )
        self.lda_model.fit(X)
        
        # Get document-topic distribution
        self.document_topics = self.lda_model.transform(X)
        
        return self
    
    def get_top_words(self, n_top_words=10):
        """
        Get the top words for each topic.
        
        Args:
            n_top_words (int): Number of top words to retrieve
            
        Returns:
            list: List of lists containing top words for each topic
        """
        if self.lda_model is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
            
        top_words = []
        for topic_idx, topic in enumerate(self.lda_model.components_):
            words = [self.feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
            top_words.append(words)
            
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
        if texts is not None:
            # Transform new texts
            X = self.vectorizer.transform(texts)
            doc_topics = self.lda_model.transform(X)
        else:
            if self.document_topics is None:
                raise ValueError("Model not fitted yet. Call fit() first.")
            doc_topics = self.document_topics
            
        # Get the topic with highest probability for each document
        main_topics = doc_topics.argmax(axis=1)
        return main_topics
    
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
            raise ValueError(f"Column '{text_column}' not found in DataFrame")
            
        # Make a copy if not inplace
        result_df = df if inplace else df.copy()
        
        # Prepare texts for LDA
        if tokens_column and tokens_column in result_df.columns:
            # Join tokens into space-separated strings
            texts = [' '.join(tokens) for tokens in result_df[tokens_column]]
        else:
            texts = result_df[text_column].tolist()
        
        # Fit LDA model
        self.fit(texts)
        
        # Get document-topic distributions
        doc_topics = self.get_document_topics()
        main_topics = self.assign_main_topic()
        
        # Add topic columns
        result_df['main_topic'] = main_topics
        
        # Add topic distribution columns
        for i in range(self.n_topics):
            result_df[f'topic_{i}_score'] = doc_topics[:, i]
        
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
            raise ValueError("Model not fitted yet. Call fit() first.")
            
        if topic_idx < 0 or topic_idx >= self.n_topics:
            raise ValueError(f"Topic index must be between 0 and {self.n_topics-1}")
        
        # Get word weights for the topic
        topic_words = self.get_topic_word_dict()[topic_idx]
        word_dict = {word: weight for word, weight in topic_words}
        
        # Generate word cloud
        wordcloud = WordCloud(
            background_color=background_color,
            width=width,
            height=height
        ).generate_from_frequencies(word_dict)
        
        return wordcloud
    
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
            raise ValueError("Model not fitted yet. Call fit() first.")
            
        n_rows = (self.n_topics + n_cols - 1) // n_cols
        
        if figsize is None:
            figsize = (n_cols * 5, n_rows * 4)
            
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