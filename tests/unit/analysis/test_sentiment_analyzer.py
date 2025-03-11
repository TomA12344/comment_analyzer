"""
Unit-Tests für die SentimentAnalyzer-Komponente.
"""
import unittest
import pandas as pd
from unittest.mock import patch, MagicMock
import sys
import os
from pathlib import Path

# Füge das Root-Verzeichnis zum Python-Pfad hinzu
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.analysis.sentiment_analyzer import SentimentAnalyzer


class TestSentimentAnalyzer(unittest.TestCase):
    """Test-Klasse für die SentimentAnalyzer-Komponente."""

    def setUp(self):
        """Setup für die Tests."""
        # Mock the model loading since we don't need it for the tests
        self.patcher = patch('src.analysis.sentiment_analyzer.SentimentAnalyzer.load_model')
        self.mock_load_model = self.patcher.start()
        self.analyzer = SentimentAnalyzer()

    def tearDown(self):
        """Cleanup nach den Tests."""
        self.patcher.stop()

    def test_init(self):
        """Test, ob SentimentAnalyzer korrekt initialisiert wird."""
        self.assertEqual(self.analyzer.labels, ["negative", "neutral", "positive"])
        self.assertIsNotNone(self.analyzer.confidence_thresholds)
        self.assertEqual(self.analyzer.confidence_thresholds["neutral_to_sentiment"], 0.6)

    def test_analyze_text_empty(self):
        """Test, ob leerer Text korrekt behandelt wird."""
        result = self.analyzer.analyze_text("")
        self.assertEqual(result["sentiment"], "neutral")
        self.assertEqual(result["scores"]["neutral"], 1.0)
        self.assertEqual(result["scores"]["positive"], 0.0)
        self.assertEqual(result["scores"]["negative"], 0.0)

    @patch('torch.nn.functional.softmax')
    @patch('torch.no_grad')
    def test_analyze_text_positive(self, mock_no_grad, mock_softmax):
        """Test, ob positiver Text korrekt klassifiziert wird."""
        # Mocks für die Torch-Funktionalität
        mock_context = MagicMock()
        mock_no_grad.return_value = mock_context
        mock_context.__enter__ = MagicMock(return_value=None)
        mock_context.__exit__ = MagicMock(return_value=None)
        
        # Mock für softmax, der ein positives Sentiment simuliert
        mock_tensor = MagicMock()
        mock_tensor.__getitem__ = lambda self, idx: [0.1, 0.2, 0.7]
        mock_softmax.return_value = mock_tensor
        
        # Mock für das Modell und Tokenizer
        self.analyzer.model = MagicMock()
        self.analyzer.model.return_value = MagicMock()
        self.analyzer.tokenizer = MagicMock()
        self.analyzer.tokenizer.return_value = {}
        
        # Analyse durchführen
        result = self.analyzer.analyze_text("Das ist ein positiver Text.")
        
        # Überprüfen, ob das Ergebnis positiv ist
        self.assertEqual(result["sentiment"], "positive")
        self.assertEqual(result["scores"]["positive"], 0.7)
        self.assertEqual(result["scores"]["neutral"], 0.2)
        self.assertEqual(result["scores"]["negative"], 0.1)

    def test_analyze_dataframe(self):
        """Test, ob DataFrame-Analyse korrekt funktioniert."""
        # Mock für analyze_text
        self.analyzer.analyze_text = MagicMock(return_value={
            "sentiment": "positive",
            "scores": {"positive": 0.7, "neutral": 0.2, "negative": 0.1}
        })
        
        # Test-DataFrame erstellen
        df = pd.DataFrame({
            'text': ['Text 1', 'Text 2'],
            'cleaned_text': ['Text 1', 'Text 2']
        })
        
        # Analyse durchführen
        result_df = self.analyzer.analyze_dataframe(df)
        
        # Überprüfen, ob die erwarteten Spalten hinzugefügt wurden
        self.assertIn('sentiment', result_df.columns)
        self.assertIn('sentiment_scores', result_df.columns)
        self.assertIn('score_positive', result_df.columns)
        self.assertIn('score_neutral', result_df.columns)
        self.assertIn('score_negative', result_df.columns)
        
        # Überprüfen, ob alle Sentiments korrekt sind
        self.assertTrue((result_df['sentiment'] == 'positive').all())
        self.assertTrue((result_df['score_positive'] == 0.7).all())


if __name__ == '__main__':
    unittest.main()