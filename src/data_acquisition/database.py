import os
import sqlite3
import pandas as pd
from pathlib import Path
from ..config import DB_PATH
from ..logger import setup_logger

# Logger für dieses Modul
logger = setup_logger(__name__)

class CommentDatabase:
    """
    Class for handling the SQLite database operations for comment storage and retrieval.
    """
    
    def __init__(self, db_path=None):
        """
        Initialize the database connection.
        
        Args:
            db_path (str): Path to the SQLite database file. If None, uses default path from config.
        """
        self.db_path = db_path or DB_PATH
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        self.connection = None
        logger.info(f"Datenbankverbindung initialisiert: {self.db_path}")
        
    def connect(self):
        """Establish connection to the database."""
        try:
            self.connection = sqlite3.connect(self.db_path)
            logger.debug("Datenbankverbindung hergestellt")
            return self.connection
        except sqlite3.Error as e:
            logger.error(f"Fehler beim Verbinden zur Datenbank: {str(e)}")
            raise
    
    def close(self):
        """Close the database connection."""
        if self.connection:
            self.connection.close()
            self.connection = None
            logger.debug("Datenbankverbindung geschlossen")
    
    def create_tables(self):
        """Create necessary tables if they don't exist."""
        conn = self.connect()
        cursor = conn.cursor()
        
        try:
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
            logger.info("Tabellen erfolgreich erstellt oder waren bereits vorhanden")
        except sqlite3.Error as e:
            logger.error(f"Fehler beim Erstellen der Tabellen: {str(e)}")
            raise
        finally:
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
        
        try:
            cursor.execute(
                "INSERT INTO comments (text, author, source, category) VALUES (?, ?, ?, ?)",
                (text, author, source, category)
            )
            
            comment_id = cursor.lastrowid
            conn.commit()
            logger.debug(f"Kommentar eingefügt mit ID {comment_id}")
            return comment_id
        except sqlite3.Error as e:
            logger.error(f"Fehler beim Einfügen eines Kommentars: {str(e)}")
            conn.rollback()
            raise
        finally:
            self.close()
    
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
        
        try:
            cursor.executemany(
                "INSERT INTO comments (text, author, source, category) VALUES (?, ?, ?, ?)",
                comments_data
            )
            
            count = cursor.rowcount
            conn.commit()
            logger.info(f"{count} Kommentare erfolgreich eingefügt")
            return count
        except sqlite3.Error as e:
            logger.error(f"Fehler beim Einfügen mehrerer Kommentare: {str(e)}")
            conn.rollback()
            raise
        finally:
            self.close()
    
    def get_all_comments(self):
        """
        Retrieve all comments from the database.
        
        Returns:
            pandas.DataFrame: DataFrame containing all comments
        """
        conn = self.connect()
        
        try:
            query = "SELECT * FROM comments"
            df = pd.read_sql_query(query, conn)
            logger.info(f"{len(df)} Kommentare aus der Datenbank abgerufen")
            return df
        except (sqlite3.Error, pd.io.sql.DatabaseError) as e:
            logger.error(f"Fehler beim Abrufen aller Kommentare: {str(e)}")
            raise
        finally:
            self.close()
    
    def get_comments_by_category(self, category):
        """
        Retrieve comments filtered by category.
        
        Args:
            category (str): Category to filter by
            
        Returns:
            pandas.DataFrame: DataFrame with filtered comments
        """
        conn = self.connect()
        
        try:
            query = "SELECT * FROM comments WHERE category = ?"
            df = pd.read_sql_query(query, conn, params=(category,))
            logger.info(f"{len(df)} Kommentare für Kategorie '{category}' gefunden")
            return df
        except (sqlite3.Error, pd.io.sql.DatabaseError) as e:
            logger.error(f"Fehler beim Abrufen von Kommentaren nach Kategorie: {str(e)}")
            raise
        finally:
            self.close()

    def search_comments(self, search_term):
        """
        Search for comments containing the specified term.
        
        Args:
            search_term (str): Term to search for in comments
            
        Returns:
            pandas.DataFrame: DataFrame with matching comments
        """
        conn = self.connect()
        
        try:
            query = "SELECT * FROM comments WHERE text LIKE ?"
            df = pd.read_sql_query(query, conn, params=(f'%{search_term}%',))
            logger.info(f"{len(df)} Kommentare gefunden, die '{search_term}' enthalten")
            return df
        except (sqlite3.Error, pd.io.sql.DatabaseError) as e:
            logger.error(f"Fehler bei der Suche nach Kommentaren: {str(e)}")
            raise
        finally:
            self.close()


def initialize_database_with_sample_data():
    """
    Initialize the database with sample German comments for testing.
    """
    logger.info("Initialisiere Datenbank mit Beispieldaten")
    
    db = CommentDatabase()
    db.create_tables()
    
    # Sample German comments for different categories
    # Hier folgt eine gekürzte Version der Beispieldaten
    sample_comments = [
        # Einige positive Kommentare
        ("Das Produkt hat meine Erwartungen übertroffen.", "Max M.", "Amazon", "Produkt"),
        ("Die Qualität ist hervorragend.", "Lisa S.", "Online-Shop", "Produkt"),
        
        # Einige negative Kommentare
        ("Leider entspricht die Qualität nicht dem Preis.", "Anna W.", "Amazon", "Produkt"),
        ("Das Gerät funktioniert nicht richtig.", "Michael B.", "Elektronikmarkt", "Produkt"),
        
        # Service-bezogene Kommentare
        ("Der Kundenservice war sehr hilfsbereit.", "Sabine F.", "Hotline", "Service"),
        ("Lange Wartezeiten am Telefon.", "Klaus W.", "Support", "Service"),
        
        # Neutrale Kommentare
        ("Das Produkt entspricht den Angaben.", "Christian W.", "Bewertung", "Elektronik"),
        ("Die Lieferung erfolgte wie angekündigt.", "Monika S.", "Versand", "Logistik")
    ]
    
    try:
        count = db.insert_many_comments(sample_comments)
        logger.info(f"Datenbank erfolgreich mit {count} Beispielkommentaren initialisiert")
    except Exception as e:
        logger.error(f"Fehler bei der Initialisierung der Datenbank: {str(e)}")
        raise

if __name__ == "__main__":
    # Run this script directly to initialize the database with sample data
    initialize_database_with_sample_data()