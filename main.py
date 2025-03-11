import argparse
import os
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent))

from src.data_acquisition.database import CommentDatabase, initialize_database_with_sample_data
from src.preprocessing.text_cleaner import TextCleaner
from src.analysis.sentiment_analyzer import SentimentAnalyzer
from src.analysis.topic_analyzer import TopicAnalyzer
from src.config import DEFAULT_LANGUAGE, DEFAULT_NUM_TOPICS
from src.logger import setup_logger

# Setup logger
logger = setup_logger(__name__)


def initialize_database():
    """Initialize the database with sample data."""
    logger.info("Initialisiere Datenbank mit Beispieldaten...")
    initialize_database_with_sample_data()
    logger.info("Datenbankinitialisierung abgeschlossen.")


def run_dashboard():
    """Run the Streamlit dashboard."""
    try:
        import streamlit
        # Check if streamlit command is accessible
        logger.info("Starte Dashboard...")
        dashboard_path = os.path.join('src', 'visualization', 'dashboard.py')
        
        # Führe den Streamlit-Befehl aus
        exit_code = os.system(f"streamlit run {dashboard_path}")
        
        if exit_code != 0:
            logger.error(f"Dashboard konnte nicht gestartet werden. Exit-Code: {exit_code}")
            print("Fehler beim Starten des Dashboards. Überprüfen Sie das Log für Details.")
    except ImportError:
        logger.error("Streamlit ist nicht installiert")
        print("Streamlit ist nicht installiert. Bitte installieren Sie es mit 'pip install streamlit'.")


def analyze_comments():
    """Run a basic analysis and print the results."""
    logger.info("Lade und analysiere Kommentare...")
    
    # Load data from database
    db = CommentDatabase()
    comments_df = db.get_all_comments()
    logger.info(f"{len(comments_df)} Kommentare aus Datenbank geladen.")
    
    # Preprocess text
    cleaner = TextCleaner(language=DEFAULT_LANGUAGE)
    processed_df = cleaner.preprocess_dataframe(comments_df)
    logger.info("Textvorverarbeitung abgeschlossen.")
    
    # Perform sentiment analysis
    sentiment_analyzer = SentimentAnalyzer()
    sentiment_df = sentiment_analyzer.analyze_dataframe(processed_df)
    logger.info("Sentiment-Analyse abgeschlossen.")
    
    # Print sentiment distribution
    print("\nSentiment-Verteilung:")
    sentiment_counts = sentiment_df['sentiment'].value_counts()
    for sentiment, count in sentiment_counts.items():
        print(f"  {sentiment}: {count} ({count/len(sentiment_df)*100:.1f}%)")
    
    # Perform topic analysis
    logger.info("Führe Themenanalyse durch...")
    topic_analyzer = TopicAnalyzer(n_topics=DEFAULT_NUM_TOPICS)
    topic_df = topic_analyzer.analyze_dataframe(sentiment_df, tokens_column='tokens')
    logger.info("Themenanalyse abgeschlossen.")
    
    # Print topics
    print("\nEntdeckte Themen:")
    topic_words = topic_analyzer.get_top_words(n_top_words=7)
    for i, words in enumerate(topic_words):
        print(f"  Thema {i}: {', '.join(words)}")
    
    # Print sample comments with sentiment and topic
    print("\nBeispiel-Kommentare mit Analyse:")
    for i, row in topic_df.sample(min(5, len(topic_df))).iterrows():
        print(f"\n  Kommentar: {row['text']}")
        print(f"  Sentiment: {row['sentiment']} (Positiv: {row['score_positive']:.2f}, "
              f"Neutral: {row['score_neutral']:.2f}, Negativ: {row['score_negative']:.2f})")
        print(f"  Hauptthema: {row['main_topic']} ({', '.join(topic_words[row['main_topic']][:3])})")
    
    logger.info("Analyse abgeschlossen und Ergebnisse angezeigt.")


def main():
    """Main entry point for the comment analyzer tool."""
    parser = argparse.ArgumentParser(description="Comment Analysis Tool")
    parser.add_argument('action', choices=['init', 'analyze', 'dashboard'], 
                      help='Action to perform: initialize database, analyze comments, or launch dashboard')
    
    args = parser.parse_args()
    
    logger.info(f"Kommentar-Analyzer gestartet mit Aktion: {args.action}")
    
    if args.action == 'init':
        initialize_database()
    elif args.action == 'analyze':
        analyze_comments()
    elif args.action == 'dashboard':
        run_dashboard()
    
    logger.info("Programm beendet.")


if __name__ == "__main__":
    main()