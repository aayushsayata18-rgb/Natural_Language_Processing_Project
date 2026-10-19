"""
Phase 3 - NLTK Resource Setup
============================================
Sarcasm & Irony Detection System using NLP

Downloads all required NLTK resources for the
preprocessing pipeline.

Usage:
    python src/preprocessing/setup_nltk.py
"""

import nltk
import sys
import logging

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

REQUIRED_RESOURCES = [
    ("tokenizers/punkt_tab", "punkt_tab"),
    ("corpora/stopwords", "stopwords"),
    ("corpora/wordnet", "wordnet"),
    ("corpora/omw-1.4", "omw-1.4"),
    ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
]


def setup_nltk():
    """Download all required NLTK resources."""
    logger.info("Checking and downloading required NLTK resources...")
    
    all_ok = True
    for resource_path, resource_name in REQUIRED_RESOURCES:
        try:
            nltk.data.find(resource_path)
            logger.info(f"  [OK] {resource_name} already available")
        except LookupError:
            logger.info(f"  [DOWNLOADING] {resource_name}...")
            try:
                nltk.download(resource_name, quiet=True)
                logger.info(f"  [OK] {resource_name} downloaded successfully")
            except Exception as e:
                logger.error(f"  [FAIL] {resource_name}: {e}")
                all_ok = False
    
    if all_ok:
        logger.info("All NLTK resources are ready.")
    else:
        logger.error("Some NLTK resources failed to download.")
        sys.exit(1)
    
    return all_ok


if __name__ == "__main__":
    setup_nltk()
