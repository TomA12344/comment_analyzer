"""
Unit-Tests für die CommentDatabase-Komponente.
"""
import unittest
import pandas as pd
import sqlite3
from unittest.mock import patch, MagicMock, mock_open
import sys
import os
from pathlib import Path
import tempfile

# Füge das Root-Verzeichnis zum Python-Pfad hinzu
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.data_acquisition.database import CommentDatabase


class TestCommentDatabase(unittest.TestCase):
    """Test-Klasse für die CommentDatabase-Komponente."""

    def setUp(self):
        """Setup für die Tests mit einer temporären In-Memory-Datenbank."""
        # Wir nutzen eine In-Memory-Datenbank für die Tests
        self.db = CommentDatabase(db_path=":memory:")
        self.db.create_tables()

    def tearDown(self):
        """Cleanup nach den Tests."""
        self.db.close()

    def test_init(self):
        """Test, ob CommentDatabase korrekt initialisiert wird."""
        self.assertEqual(self.db.db_path, ":memory:")
        self.assertIsNone(self.db.connection)

    def test_connect(self):
        """Test der Datenbankverbindung."""
        conn = self.db.connect()
        self.assertIsNotNone(conn)
        self.assertIsInstance(conn, sqlite3.Connection)
        self.db.close()

    def test_create_tables(self):
        """Test, ob Tabellen korrekt erstellt werden."""
        # Verbinden und prüfen, ob die comments-Tabelle existiert
        conn = self.db.connect()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='comments'"
        )
        result = cursor.fetchone()
        self.assertIsNotNone(result)
        self.assertEqual(result[0], 'comments')
        self.db.close()

    def test_insert_comment(self):
        """Test des Einfügens eines einzelnen Kommentars."""
        comment_id = self.db.insert_comment(
            text="Das ist ein Testkommentar",
            author="Test Autor",
            source="Test Quelle",
            category="Test Kategorie"
        )
        
        # Prüfen, ob der Kommentar eingefügt wurde und die ID zurückgegeben wird
        self.assertIsNotNone(comment_id)
        self.assertIsInstance(comment_id, int)
        
        # Direkter SQL-Zugriff zum Überprüfen des eingefügten Kommentars
        conn = self.db.connect()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM comments WHERE id=?", (comment_id,))
        result = cursor.fetchone()
        self.assertIsNotNone(result)
        self.assertEqual(result[1], "Das ist ein Testkommentar")
        self.assertEqual(result[2], "Test Autor")
        self.assertEqual(result[4], "Test Quelle")
        self.assertEqual(result[5], "Test Kategorie")
        self.db.close()

    def test_insert_many_comments(self):
        """Test des Einfügens mehrerer Kommentare."""
        comments_data = [
            ("Kommentar 1", "Autor 1", "Quelle 1", "Kategorie 1"),
            ("Kommentar 2", "Autor 2", "Quelle 2", "Kategorie 2"),
            ("Kommentar 3", "Autor 3", "Quelle 3", "Kategorie 3")
        ]
        
        count = self.db.insert_many_comments(comments_data)
        
        # Prüfen, ob die richtige Anzahl eingefügt wurde
        self.assertEqual(count, 3)
        
        # Prüfen, ob alle Kommentare in der Datenbank sind
        conn = self.db.connect()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM comments")
        result = cursor.fetchone()
        self.assertEqual(result[0], 3)
        self.db.close()

    def test_get_all_comments(self):
        """Test des Abrufs aller Kommentare."""
        # Füge Testdaten ein
        comments_data = [
            ("Kommentar 1", "Autor 1", "Quelle 1", "Kategorie 1"),
            ("Kommentar 2", "Autor 2", "Quelle 2", "Kategorie 2")
        ]
        self.db.insert_many_comments(comments_data)
        
        # Rufe alle Kommentare ab
        df = self.db.get_all_comments()
        
        # Überprüfe das Ergebnis
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 2)
        self.assertIn('text', df.columns)
        self.assertIn('author', df.columns)
        self.assertIn('source', df.columns)
        self.assertIn('category', df.columns)
        self.assertEqual(df['text'].iloc[0], "Kommentar 1")
        self.assertEqual(df['text'].iloc[1], "Kommentar 2")

    def test_get_comments_by_category(self):
        """Test des Abrufs von Kommentaren nach Kategorie."""
        # Füge Testdaten ein
        comments_data = [
            ("Kommentar 1", "Autor 1", "Quelle 1", "Kategorie A"),
            ("Kommentar 2", "Autor 2", "Quelle 2", "Kategorie B"),
            ("Kommentar 3", "Autor 3", "Quelle 3", "Kategorie A")
        ]
        self.db.insert_many_comments(comments_data)
        
        # Rufe Kommentare nach Kategorie ab
        df = self.db.get_comments_by_category("Kategorie A")
        
        # Überprüfe das Ergebnis
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 2)
        self.assertTrue((df['category'] == "Kategorie A").all())
        self.assertEqual(df['text'].iloc[0], "Kommentar 1")
        self.assertEqual(df['text'].iloc[1], "Kommentar 3")

    def test_search_comments(self):
        """Test der Kommentarsuche."""
        # Füge Testdaten ein
        comments_data = [
            ("Dies ist ein Testkommentar", "Autor 1", "Quelle 1", "Kategorie 1"),
            ("Ein anderer Kommentar ohne Suchbegriff", "Autor 2", "Quelle 2", "Kategorie 2"),
            ("Noch ein Test", "Autor 3", "Quelle 3", "Kategorie 3")
        ]
        self.db.insert_many_comments(comments_data)
        
        # Suche nach Kommentaren
        df = self.db.search_comments("Test")
        
        # Überprüfe das Ergebnis
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 2)
        self.assertIn("Test", df['text'].iloc[0])
        self.assertIn("Test", df['text'].iloc[1])


if __name__ == '__main__':
    unittest.main()