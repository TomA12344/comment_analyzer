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


class Dashboard:
    """
    A Streamlit dashboard for visualizing comment analysis results.
    """
    
    def __init__(self):
        """Initialize the dashboard components."""
        self.db = CommentDatabase()
        self.text_cleaner = TextCleaner(language='german')
        self.sentiment_analyzer = SentimentAnalyzer()
        self.topic_analyzer = TopicAnalyzer(n_topics=4)
        self.df = None
    
    def load_data(self):
        """Load data from the database and preprocess it."""
        # Get comments from database
        df = self.db.get_all_comments()
        
        # Preprocess text
        df = self.text_cleaner.preprocess_dataframe(df)
        
        # Perform sentiment analysis
        df = self.sentiment_analyzer.analyze_dataframe(df)
        
        # Perform topic analysis
        df = self.topic_analyzer.analyze_dataframe(df, tokens_column='tokens')
        
        self.df = df
        return df
    
    def run(self):
        """Run the Streamlit dashboard."""
        st.title("Kommentar-Analyse Dashboard")
        
        # Create sidebar for filters
        st.sidebar.title("Filter")
        
        # Load data if not loaded
        if self.df is None:
            with st.spinner("Daten werden geladen und analysiert..."):
                self.load_data()
        
        # Display data overview section
        self.display_data_overview()
        
        # Display sentiment analysis section
        self.display_sentiment_analysis()
        
        # Display topic analysis section
        self.display_topic_analysis()
        
        # Display detailed data section
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
        
        with col2:
            # Average sentiment scores
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
        
        # Display most positive and negative comments
        st.subheader("Top Positive & Negative Kommentare")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("📈 Positivste Kommentare:")
            top_positive = self.df.sort_values('score_positive', ascending=False).head(3)
            for i, row in enumerate(top_positive.itertuples(), 1):
                st.info(f"{i}. {row.text} (Score: {row.score_positive:.2f})")
        
        with col2:
            st.write("📉 Negativste Kommentare:")
            top_negative = self.df.sort_values('score_negative', ascending=False).head(3)
            for i, row in enumerate(top_negative.itertuples(), 1):
                st.error(f"{i}. {row.text} (Score: {row.score_negative:.2f})")
    
    def display_topic_analysis(self):
        """Display topic analysis section."""
        st.header("Themen-Analyse")
        
        # Display topic distributions
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
        
        # Word clouds
        st.subheader("Themen-Wordclouds")
        
        # Generate word clouds for each topic
        if st.button("Wordclouds anzeigen"):
            fig = self.topic_analyzer.plot_topic_wordclouds()
            st.pyplot(fig)
            
        # Topic-Sentiment relationship
        st.subheader("Themen-Sentiment Beziehung")
        
        # Create a pivot table of topic vs sentiment
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
        st.dataframe(
            filtered_df[['text', 'sentiment', 'category', 'author', 'source', 'main_topic']],
            height=300
        )
        
        # Search functionality
        search_term = st.text_input("Kommentare durchsuchen:")
        if search_term:
            search_results = self.df[self.df['text'].str.contains(search_term, case=False)]
            st.write(f"{len(search_results)} Ergebnisse gefunden:")
            st.dataframe(
                search_results[['text', 'sentiment', 'category', 'author', 'source', 'main_topic']],
                height=300
            )

def main():
    """Main function to run the dashboard."""
    dashboard = Dashboard()
    dashboard.run()

if __name__ == "__main__":
    main()