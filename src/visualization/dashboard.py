import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
import os

# Add parent directory to path to import from other modules
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data_acquisition.database import CommentDatabase
from src.preprocessing.text_cleaner import TextCleaner
from src.analysis.sentiment_analyzer import SentimentAnalyzer
from src.analysis.topic_analyzer import TopicAnalyzer
from src.config import DEFAULT_LANGUAGE, DEFAULT_NUM_TOPICS
from src.logger import setup_logger

# Logger für dieses Modul
logger = setup_logger(__name__)


class Dashboard:
    """
    A Streamlit dashboard for visualizing comment analysis results.
    """
    
    def __init__(self):
        """Initialize the dashboard components."""
        logger.info("Initialisiere Dashboard-Komponenten")
        self.db = CommentDatabase()
        self.text_cleaner = TextCleaner(language=DEFAULT_LANGUAGE)
        self.sentiment_analyzer = SentimentAnalyzer()
        self.topic_analyzer = TopicAnalyzer(n_topics=DEFAULT_NUM_TOPICS)
        self.df = None
    
    def load_data(self):
        """Load data from the database and preprocess it."""
        # Get comments from database
        logger.info("Lade Daten aus der Datenbank")
        df = self.db.get_all_comments()
        
        # Preprocess text
        logger.info("Führe Textvorverarbeitung durch")
        df = self.text_cleaner.preprocess_dataframe(df)
        
        # Perform sentiment analysis
        logger.info("Führe Sentiment-Analyse durch")
        df = self.sentiment_analyzer.analyze_dataframe(df)
        
        # Perform topic analysis
        logger.info("Führe Themenanalyse durch")
        df = self.topic_analyzer.analyze_dataframe(df, tokens_column='tokens')
        
        self.df = df
        logger.info(f"Daten geladen und verarbeitet: {len(df)} Kommentare")
        return df
    
    def run(self):
        """Run the Streamlit dashboard."""
        st.title("Kommentar-Analyse Dashboard")
        
        # Create sidebar for filters
        st.sidebar.title("Filter")
        
        # Load data if not loaded
        if self.df is None:
            with st.spinner("Daten werden geladen und analysiert..."):
                try:
                    self.load_data()
                    logger.info("Daten erfolgreich geladen")
                except Exception as e:
                    logger.error(f"Fehler beim Laden der Daten: {str(e)}")
                    st.error(f"Fehler beim Laden der Daten: {str(e)}")
                    return
        
        # Display data overview section
        logger.debug("Zeige Datenübersicht an")
        self.display_data_overview()
        
        # Display sentiment analysis section
        logger.debug("Zeige Sentiment-Analyse an")
        self.display_sentiment_analysis()
        
        # Display topic analysis section
        logger.debug("Zeige Themenanalyse an")
        self.display_topic_analysis()
        
        # Display detailed data section
        logger.debug("Zeige Daten-Explorer an")
        self.display_data_explorer()
    
    def display_data_overview(self):
        """Display data overview section."""
        st.header("Datenübersicht")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Total number of comments
            st.metric("Anzahl der Kommentare", len(self.df))
            
            # Comments by category
            if 'category' in self.df.columns:
                category_counts = self.df['category'].value_counts()
                fig = px.pie(
                    values=category_counts.values,
                    names=category_counts.index,
                    title="Kommentare nach Kategorie"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Comments by sentiment
            sentiment_counts = self.df['sentiment'].value_counts()
            fig = px.pie(
                values=sentiment_counts.values,
                names=sentiment_counts.index,
                title="Sentiment-Verteilung",
                color=sentiment_counts.index,
                color_discrete_map={
                    'positive': 'green',
                    'neutral': 'gray',
                    'negative': 'red'
                }
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def display_sentiment_analysis(self):
        """Display sentiment analysis section."""
        st.header("Sentiment-Analyse")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Sentiment by category
            if 'category' in self.df.columns:
                try:
                    sentiment_by_category = pd.crosstab(
                        self.df['category'], 
                        self.df['sentiment'],
                        normalize='index'
                    ) * 100
                    
                    fig = px.bar(
                        sentiment_by_category,
                        title="Sentiment nach Kategorie (%)",
                        color_discrete_map={
                            'positive': 'green',
                            'neutral': 'gray',
                            'negative': 'red'
                        }
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    logger.error(f"Fehler bei der Erstellung des Sentiment-Kategorie-Diagramms: {str(e)}")
                    st.error("Diagramm konnte nicht erstellt werden")
        
        with col2:
            # Average sentiment scores
            try:
                score_cols = [col for col in self.df.columns if col.startswith('score_')]
                avg_scores = self.df[score_cols].mean().reset_index()
                avg_scores.columns = ['Sentiment', 'Average Score']
                avg_scores['Sentiment'] = avg_scores['Sentiment'].str.replace('score_', '')
                
                fig = px.bar(
                    avg_scores,
                    x='Sentiment',
                    y='Average Score',
                    title="Durchschnittliche Sentiment-Scores",
                    color='Sentiment',
                    color_discrete_map={
                        'positive': 'green',
                        'neutral': 'gray',
                        'negative': 'red'
                    }
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                logger.error(f"Fehler bei der Erstellung des durchschnittlichen Sentiment-Score-Diagramms: {str(e)}")
                st.error("Diagramm konnte nicht erstellt werden")
        
        # Display most positive and negative comments
        st.subheader("Top Positive & Negative Kommentare")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("📈 Positivste Kommentare:")
            # Sortierung und Deduplizierung, um verschiedene Kommentare anzuzeigen
            try:
                top_positive = self.df.sort_values('score_positive', ascending=False)
                # Entfernt Duplikate im Text, behält die erste Instanz jedes Textes
                top_positive_unique = top_positive.drop_duplicates(subset=['text']).head(3)
                
                for i, row in enumerate(top_positive_unique.itertuples(), 1):
                    st.info(f"{i}. {row.text} (Score: {row.score_positive:.2f})")
            except Exception as e:
                logger.error(f"Fehler bei der Anzeige der Top-positiven Kommentare: {str(e)}")
                st.error("Kommentare konnten nicht angezeigt werden")
        
        with col2:
            st.write("📉 Negativste Kommentare:")
            # Sortierung und Deduplizierung, um verschiedene Kommentare anzuzeigen
            try:
                top_negative = self.df.sort_values('score_negative', ascending=False)
                # Entfernt Duplikate im Text, behält die erste Instanz jedes Textes
                top_negative_unique = top_negative.drop_duplicates(subset=['text']).head(3)
                
                for i, row in enumerate(top_negative_unique.itertuples(), 1):
                    st.error(f"{i}. {row.text} (Score: {row.score_negative:.2f})")
            except Exception as e:
                logger.error(f"Fehler bei der Anzeige der Top-negativen Kommentare: {str(e)}")
                st.error("Kommentare konnten nicht angezeigt werden")
    
    def display_topic_analysis(self):
        """Display topic analysis section."""
        st.header("Themen-Analyse")
        
        # Display topic distributions
        try:
            topic_dist = pd.DataFrame({
                'Anzahl': self.df['main_topic'].value_counts()
            }).reset_index()
            topic_dist.columns = ['Thema', 'Anzahl']
            
            # Get top words for each topic to create labels
            topic_words = self.topic_analyzer.get_top_words(n_top_words=5)
            topic_labels = [f"Thema {i}: {', '.join(words[:3])}" for i, words in enumerate(topic_words)]
            
            # Map topic numbers to labels
            topic_mapping = {i: label for i, label in enumerate(topic_labels)}
            topic_dist['Thema'] = topic_dist['Thema'].map(lambda x: topic_mapping.get(x, f"Thema {x}"))
            
            st.subheader("Themenverteilung")
            fig = px.bar(
                topic_dist, 
                x='Thema', 
                y='Anzahl',
                title="Anzahl der Kommentare pro Thema"
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            logger.error(f"Fehler bei der Anzeige der Themenverteilung: {str(e)}")
            st.error("Themenverteilung konnte nicht angezeigt werden")
        
        # Word clouds
        st.subheader("Themen-Wordclouds")
        
        # Generate word clouds for each topic
        if st.button("Wordclouds anzeigen"):
            try:
                logger.info("Erstelle Wordclouds für alle Themen")
                fig = self.topic_analyzer.plot_topic_wordclouds()
                st.pyplot(fig)
            except Exception as e:
                logger.error(f"Fehler bei der Erstellung der Wordclouds: {str(e)}")
                st.error("Wordclouds konnten nicht erstellt werden")
            
        # Topic-Sentiment relationship
        st.subheader("Themen-Sentiment Beziehung")
        
        # Create a pivot table of topic vs sentiment
        try:
            topic_sentiment = pd.crosstab(
                self.df['main_topic'].map(lambda x: topic_mapping.get(x, f"Thema {x}")), 
                self.df['sentiment'],
            )
            
            fig = px.bar(
                topic_sentiment,
                title="Sentiment pro Thema",
                color_discrete_map={
                    'positive': 'green',
                    'neutral': 'gray',
                    'negative': 'red'
                }
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            logger.error(f"Fehler bei der Anzeige der Themen-Sentiment-Beziehung: {str(e)}")
            st.error("Themen-Sentiment-Beziehung konnte nicht angezeigt werden")
    
    def display_data_explorer(self):
        """Display data explorer section."""
        st.header("Daten-Explorer")
        
        # Allow filtering by sentiment
        sentiment_filter = st.multiselect(
            "Nach Sentiment filtern:",
            options=sorted(self.df['sentiment'].unique()),
            default=sorted(self.df['sentiment'].unique())
        )
        
        # Allow filtering by category if available
        category_filter = None
        if 'category' in self.df.columns:
            category_filter = st.multiselect(
                "Nach Kategorie filtern:",
                options=sorted(self.df['category'].unique()),
                default=sorted(self.df['category'].unique())
            )
        
        # Apply filters
        filtered_df = self.df
        if sentiment_filter:
            filtered_df = filtered_df[filtered_df['sentiment'].isin(sentiment_filter)]
        if category_filter:
            filtered_df = filtered_df[filtered_df['category'].isin(category_filter)]
        
        # Show filtered data
        try:
            display_columns = ['text', 'sentiment', 'category', 'author', 'source', 'main_topic']
            valid_columns = [col for col in display_columns if col in filtered_df.columns]
            
            st.dataframe(
                filtered_df[valid_columns],
                height=300
            )
            logger.debug(f"Zeige gefilterte Daten an: {len(filtered_df)} Einträge")
        except Exception as e:
            logger.error(f"Fehler bei der Anzeige der gefilterten Daten: {str(e)}")
            st.error("Daten konnten nicht angezeigt werden")
        
        # Search functionality
        search_term = st.text_input("Kommentare durchsuchen:")
        if search_term:
            try:
                search_results = self.df[self.df['text'].str.contains(search_term, case=False)]
                st.write(f"{len(search_results)} Ergebnisse gefunden:")
                st.dataframe(
                    search_results[valid_columns],
                    height=300
                )
                logger.info(f"Suchergebnisse für '{search_term}': {len(search_results)} Treffer")
            except Exception as e:
                logger.error(f"Fehler bei der Suche nach '{search_term}': {str(e)}")
                st.error("Suchergebnisse konnten nicht angezeigt werden")


def main():
    """Main function to run the dashboard."""
    logger.info("Dashboard wird gestartet")
    dashboard = Dashboard()
    dashboard.run()
    logger.info("Dashboard-Ausführung beendet")


if __name__ == "__main__":
    main()