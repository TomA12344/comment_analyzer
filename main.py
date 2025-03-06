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


def initialize_database():
    """Initialize the database with sample data."""
    print("Initializing database with sample data...")
    initialize_database_with_sample_data()
    print("Database initialization complete.")


def run_dashboard():
    """Run the Streamlit dashboard."""
    try:
        import streamlit
        # Check if streamlit command is accessible
        print("Starting dashboard...")
        dashboard_path = os.path.join('src', 'visualization', 'dashboard.py')
        os.system(f"streamlit run {dashboard_path}")
    except ImportError:
        print("Streamlit is not installed. Please install it using 'pip install streamlit'.")


def analyze_comments():
    """Run a basic analysis and print the results."""
    print("Loading and analyzing comments...")
    
    # Load data from database
    db = CommentDatabase()
    comments_df = db.get_all_comments()
    print(f"Loaded {len(comments_df)} comments from database.")
    
    # Preprocess text
    cleaner = TextCleaner(language='german')
    processed_df = cleaner.preprocess_dataframe(comments_df)
    print("Text preprocessing complete.")
    
    # Perform sentiment analysis
    sentiment_analyzer = SentimentAnalyzer()
    sentiment_df = sentiment_analyzer.analyze_dataframe(processed_df)
    print("Sentiment analysis complete.")
    
    # Print sentiment distribution
    print("\nSentiment Distribution:")
    sentiment_counts = sentiment_df['sentiment'].value_counts()
    for sentiment, count in sentiment_counts.items():
        print(f"  {sentiment}: {count} ({count/len(sentiment_df)*100:.1f}%)")
    
    # Perform topic analysis
    print("\nPerforming topic analysis...")
    topic_analyzer = TopicAnalyzer(n_topics=3)
    topic_df = topic_analyzer.analyze_dataframe(sentiment_df, tokens_column='tokens')
    
    # Print topics
    print("\nDiscovered Topics:")
    topic_words = topic_analyzer.get_top_words(n_top_words=7)
    for i, words in enumerate(topic_words):
        print(f"  Topic {i}: {', '.join(words)}")
    
    # Print sample comments with sentiment and topic
    print("\nSample Comments with Analysis:")
    for i, row in topic_df.sample(min(5, len(topic_df))).iterrows():
        print(f"\n  Comment: {row['text']}")
        print(f"  Sentiment: {row['sentiment']} (Positive: {row['score_positive']:.2f}, "
              f"Neutral: {row['score_neutral']:.2f}, Negative: {row['score_negative']:.2f})")
        print(f"  Main Topic: {row['main_topic']} ({', '.join(topic_words[row['main_topic']][:3])})")


def main():
    """Main entry point for the comment analyzer tool."""
    parser = argparse.ArgumentParser(description="Comment Analysis Tool")
    parser.add_argument('action', choices=['init', 'analyze', 'dashboard'], 
                      help='Action to perform: initialize database, analyze comments, or launch dashboard')
    
    args = parser.parse_args()
    
    if args.action == 'init':
        initialize_database()
    elif args.action == 'analyze':
        analyze_comments()
    elif args.action == 'dashboard':
        run_dashboard()


if __name__ == "__main__":
    main()