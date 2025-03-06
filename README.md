# Comment Analyzer

Eine Python-Anwendung zur Analyse deutschsprachiger Kommentare mit Schwerpunkt auf Sentiment-Analyse und Themenmodellierung.

## Funktionalitäten

- **Datenerfassung**: Speichert und verwaltet Kommentare in einer SQLite-Datenbank
- **Textvorverarbeitung**: Bereinigt Texte und entfernt Stoppwörter
- **Sentiment-Analyse**: Klassifiziert Kommentare als positiv, neutral oder negativ mit einem vortrainierten deutschen BERT-Modell
- **Themenanalyse**: Extrahiert Hauptthemen aus Kommentaren mittels Latent Dirichlet Allocation (LDA)
- **Visualisierung**: Interaktives Dashboard zur Darstellung von Analyse-Ergebnissen

## Installation

1. Klonen Sie das Repository
2. Installieren Sie die erforderlichen Abhängigkeiten:

```
pip install -r requirements.txt
```

## Verwendung

Die Anwendung bietet drei Hauptbefehle:

### 1. Datenbank initialisieren

```
python main.py init
```
Initialisiert die Datenbank mit Beispielkommentaren für Testzwecke.

### 2. Kommentare analysieren

```
python main.py analyze
```
Führt eine Basisanalyse durch und gibt die Ergebnisse in der Konsole aus.

### 3. Dashboard starten

```
python main.py dashboard
```
Startet ein interaktives Streamlit-Dashboard zur Visualisierung der Analyse-Ergebnisse.

## Projektstruktur

```
comment_analyzer/
├── data/                   # Verzeichnis für Datenbanken und Dateien
├── src/                    # Quellcode
│   ├── data_acquisition/   # Datenbankfunktionen
│   ├── preprocessing/      # Textvorverarbeitung
│   ├── analysis/           # Sentiment- und Themenanalyse
│   └── visualization/      # Dashboard und Visualisierung
├── main.py                 # Hauptskript
└── requirements.txt        # Abhängigkeiten
```

## Technologien

- **Datenverarbeitung**: pandas, numpy, sqlite3
- **Textverarbeitung**: NLTK
- **Sentiment-Analyse**: transformers, torch (BERT-Modell)
- **Themenmodellierung**: scikit-learn, matplotlib, wordcloud
- **Visualisierung**: streamlit, plotly

## Beispiel-Workflow

1. Initialisieren Sie die Datenbank mit `python main.py init`
2. Führen Sie eine Analyse durch mit `python main.py analyze`
3. Starten Sie das Dashboard mit `python main.py dashboard`
4. Erkunden Sie die Sentiment-Verteilung und Themenanalyse im Browser

## Erweiterungsmöglichkeiten

- Integration weiterer Datenquellen (z.B. API-Anbindung für Social Media)
- Implementierung zeitbasierter Analysen
- Unterstützung weiterer Sprachen
- Benutzerdefinierte Kategorisierung von Kommentaren