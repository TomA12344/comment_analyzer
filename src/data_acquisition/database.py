import os
import sqlite3
import pandas as pd
from pathlib import Path

class CommentDatabase:
    """
    Class for handling the SQLite database operations for comment storage and retrieval.
    """
    
    def __init__(self, db_path=None):
        """
        Initialize the database connection.
        
        Args:
            db_path (str): Path to the SQLite database file. If None, uses default path.
        """
        if db_path is None:
            # Default path is in the data directory
            base_dir = Path(__file__).parent.parent.parent
            db_path = os.path.join(base_dir, 'data', 'comments.db')
            
        # Ensure directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        self.db_path = db_path
        self.connection = None
        
    def connect(self):
        """Establish connection to the database."""
        self.connection = sqlite3.connect(self.db_path)
        return self.connection
    
    def close(self):
        """Close the database connection."""
        if self.connection:
            self.connection.close()
            self.connection = None
    
    def create_tables(self):
        """Create necessary tables if they don't exist."""
        conn = self.connect()
        cursor = conn.cursor()
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS comments (
            id INTEGER PRIMARY KEY,
            text TEXT NOT NULL,
            author TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            source TEXT,
            category TEXT
        )
        ''')
        
        conn.commit()
        self.close()
        
    def insert_comment(self, text, author=None, source=None, category=None):
        """
        Insert a single comment into the database.
        
        Args:
            text (str): The comment text
            author (str): Comment author name
            source (str): Source of the comment
            category (str): Category/topic of the comment
        
        Returns:
            int: The ID of the inserted comment
        """
        conn = self.connect()
        cursor = conn.cursor()
        
        cursor.execute(
            "INSERT INTO comments (text, author, source, category) VALUES (?, ?, ?, ?)",
            (text, author, source, category)
        )
        
        comment_id = cursor.lastrowid
        conn.commit()
        self.close()
        
        return comment_id
    
    def insert_many_comments(self, comments_data):
        """
        Insert multiple comments into the database.
        
        Args:
            comments_data (list): List of tuples containing (text, author, source, category)
        
        Returns:
            int: Number of inserted comments
        """
        conn = self.connect()
        cursor = conn.cursor()
        
        cursor.executemany(
            "INSERT INTO comments (text, author, source, category) VALUES (?, ?, ?, ?)",
            comments_data
        )
        
        count = cursor.rowcount
        conn.commit()
        self.close()
        
        return count
    
    def get_all_comments(self):
        """
        Retrieve all comments from the database.
        
        Returns:
            pandas.DataFrame: DataFrame containing all comments
        """
        conn = self.connect()
        
        query = "SELECT * FROM comments"
        df = pd.read_sql_query(query, conn)
        
        self.close()
        return df
    
    def get_comments_by_category(self, category):
        """
        Retrieve comments filtered by category.
        
        Args:
            category (str): Category to filter by
            
        Returns:
            pandas.DataFrame: DataFrame with filtered comments
        """
        conn = self.connect()
        
        query = "SELECT * FROM comments WHERE category = ?"
        df = pd.read_sql_query(query, conn, params=(category,))
        
        self.close()
        return df

    def search_comments(self, search_term):
        """
        Search for comments containing the specified term.
        
        Args:
            search_term (str): Term to search for in comments
            
        Returns:
            pandas.DataFrame: DataFrame with matching comments
        """
        conn = self.connect()
        
        query = "SELECT * FROM comments WHERE text LIKE ?"
        df = pd.read_sql_query(query, conn, params=(f'%{search_term}%',))
        
        self.close()
        return df


def initialize_database_with_sample_data():
    """
    Initialize the database with sample German comments for testing.
    """
    db = CommentDatabase()
    db.create_tables()
    
    # Sample German comments for different categories
    sample_comments = [
        # Positive comments about products
        ("Das Produkt hat alle meine Erwartungen übertroffen. Sehr zufrieden!", "Max Mustermann", "Amazon", "Produkt"),
        ("Die Qualität ist hervorragend und der Preis ist angemessen.", "Lisa Schmidt", "Online-Shop", "Produkt"),
        ("Schnelle Lieferung und einwandfreie Ware. Gerne wieder!", "Thomas Müller", "eBay", "Produkt"),
        
        # Negative comments about products
        ("Leider entspricht die Qualität nicht dem Preis. Bin enttäuscht.", "Anna Weber", "Amazon", "Produkt"),
        ("Nach nur einer Woche ist das Gerät kaputt gegangen. Nicht zu empfehlen.", "Michael Bauer", "Elektronikmarkt", "Produkt"),
        
        # Service-related comments
        ("Der Kundenservice war sehr hilfsbereit und freundlich.", "Sabine Fischer", "Hotline", "Service"),
        ("Lange Wartezeiten am Telefon und unfreundliches Personal.", "Klaus Wagner", "Support", "Service"),
        ("Die Beratung war kompetent und hat mir sehr geholfen.", "Julia König", "Fachgeschäft", "Service"),
        
        # Restaurant reviews
        ("Das Essen war köstlich und das Ambiente sehr angenehm.", "Stefan Schneider", "Restaurant", "Gastronomie"),
        ("Zu lange Wartezeit und das Essen war nur lauwarm.", "Petra Hoffmann", "Restaurant", "Gastronomie"),
        ("Freundliche Bedienung, aber das Preis-Leistungs-Verhältnis stimmt nicht.", "Martin Wolf", "Café", "Gastronomie"),
        
        # Website feedback
        ("Die Webseite ist übersichtlich und benutzerfreundlich gestaltet.", "Daniel Meyer", "Website", "Online"),
        ("Zu viele Pop-ups und die Ladezeit ist viel zu lang.", "Laura Schäfer", "Website", "Online"),
        ("Die neue App funktioniert einwandfrei und hat ein tolles Design.", "Sophia Richter", "Mobile App", "Online")
    ]
    
    db.insert_many_comments(sample_comments)
    print(f"Inserted {len(sample_comments)} sample comments into the database.")

if __name__ == "__main__":
    # Run this script directly to initialize the database with sample data
    initialize_database_with_sample_data()