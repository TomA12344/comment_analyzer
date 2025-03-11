"""
Unit-Tests für die TextCleaner-Komponente.
"""
import unittest
import pandas as pd
from unittest.mock import patch, MagicMock
import sys
import os
from pathlib import Path

# Füge das Root-Verzeichnis zum Python-Pfad hinzu
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.preprocessing.text_cleaner import TextCleaner


class TestTextCleaner(unittest.TestCase):
    """Test-Klasse für die TextCleaner-Komponente."""

    def setUp(self):
        """Setup für die Tests."""
        # Mock für das Laden der Stopwörter
        with patch('nltk.corpus.stopwords.words') as mock_stopwords:
            mock_stopwords.return_value = ['der', 'die', 'das', 'und', 'in', 'ist', 'mit']
            self.cleaner = TextCleaner(language='german')

    def test_init(self):
        """Test, ob TextCleaner korrekt initialisiert wird."""
        self.assertEqual(self.cleaner.language, 'german')
        self.assertIsNotNone(self.cleaner.stop_words)
        self.assertIn('der', self.cleaner.stop_words)
        self.assertIn('und', self.cleaner.stop_words)

    def test_clean_text(self):
        """Test der Textbereinigungsfunktion."""
        # Test mit normalem Text
        text = "Dieser Text enthält URLs wie https://example.com und E-Mails wie test@example.com!"
        cleaned = self.cleaner.clean_text(text)
        self.assertNotIn("https://example.com", cleaned)
        self.assertNotIn("test@example.com", cleaned)
        self.assertNotIn("!", cleaned)
        self.assertEqual(cleaned, "dieser text enthält urls wie  und emails wie ")
        
        # Test mit leerem Text
        self.assertEqual(self.cleaner.clean_text(""), "")
        
        # Test mit None
        self.assertEqual(self.cleaner.clean_text(None), "")

    def test_tokenize(self):
        """Test der Tokenisierungsfunktion."""
        text = "dies ist ein test"
        tokens = self.cleaner.tokenize(text)
        self.assertEqual(tokens, ["dies", "ist", "ein", "test"])
        
        # Test mit leerem Text
        self.assertEqual(self.cleaner.tokenize(""), [])

    def test_remove_stopwords(self):
        """Test der Stopwort-Entfernung."""
        tokens = ["dies", "ist", "ein", "test", "und", "mehr"]
        filtered = self.cleaner.remove_stopwords(tokens)
        self.assertNotIn("und", filtered)
        self.assertNotIn("ist", filtered)
        self.assertIn("dies", filtered)
        self.assertIn("test", filtered)
        self.assertIn("ein", filtered)
        self.assertIn("mehr", filtered)

    def test_preprocess(self):
        """Test des gesamten Vorverarbeitungsprozesses."""
        text = "Dies ist ein Test-Text mit URLs https://example.com!"
        processed = self.cleaner.preprocess(text)
        self.assertIn("dies", processed)
        self.assertIn("test", processed)
        self.assertIn("text", processed)
        self.assertNotIn("ist", processed)
        self.assertNotIn("mit", processed)
        self.assertNotIn("https", processed)
        self.assertNotIn("example", processed)
        self.assertNotIn("com", processed)

    def test_preprocess_keep_text(self):
        """Test der Vorverarbeitung mit Beibehaltung des bereinigten Textes."""
        text = "Dies ist ein Test-Text!"
        cleaned, tokens = self.cleaner.preprocess_keep_text(text)
        self.assertEqual(cleaned, "dies ist ein testtext")
        self.assertIn("dies", tokens)
        self.assertIn("test", tokens)
        self.assertIn("text", tokens)
        self.assertNotIn("ist", tokens)

    def test_preprocess_dataframe(self):
        """Test der DataFrame-Vorverarbeitung."""
        # Test-DataFrame erstellen
        df = pd.DataFrame({
            'text': ['Dies ist ein Test!', 'Ein anderer Test mit Stopwörtern.']
        })
        
        # Vorverarbeitung durchführen
        result_df = self.cleaner.preprocess_dataframe(df)
        
        # Überprüfen, ob die erwarteten Spalten hinzugefügt wurden
        self.assertIn('cleaned_text', result_df.columns)
        self.assertIn('tokens', result_df.columns)
        
        # Prüfen, ob die Tokens wie erwartet sind
        self.assertIn('test', result_df['tokens'].iloc[0])
        self.assertIn('test', result_df['tokens'].iloc[1])
        self.assertIn('stopwörtern', result_df['tokens'].iloc[1])
        
        # Prüfen, ob bei einem falschen Spaltennamen eine ValueError ausgelöst wird
        with self.assertRaises(ValueError):
            self.cleaner.preprocess_dataframe(df, text_column='nonexistent')


if __name__ == '__main__':
    unittest.main()