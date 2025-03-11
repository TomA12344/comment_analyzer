"""
Logging-Konfiguration für das gesamte Comment Analyzer Projekt.
Stellt ein einheitliches Logging-Format und -Verhalten sicher.
"""
import logging
import os
from pathlib import Path
from .config import LOG_LEVEL, LOG_FORMAT, LOG_FILE, BASE_DIR

# Stelle sicher, dass das Logs-Verzeichnis existiert
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

def setup_logger(name, level=None):
    """
    Erstellt und konfiguriert einen Logger mit einheitlichem Format.
    
    Args:
        name (str): Name des Loggers, typischerweise der Modulname (__name__)
        level (str, optional): Log-Level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
                              Falls None, wird der Wert aus der Konfiguration verwendet
    
    Returns:
        logging.Logger: Konfigurierter Logger
    """
    logger = logging.getLogger(name)
    
    # Log-Level setzen (Konfiguration oder Parameter)
    log_level = level or LOG_LEVEL
    logger.setLevel(getattr(logging, log_level))
    
    # Handler entfernen, falls bereits vorhanden (verhindert Doppeleinträge)
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Console Handler hinzufügen
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(console_handler)
    
    # File Handler hinzufügen
    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(file_handler)
    
    return logger

# Root-Logger konfigurieren
root_logger = setup_logger('comment_analyzer')