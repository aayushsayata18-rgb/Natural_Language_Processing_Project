"""
Phase 3 - Preprocessing Configuration
============================================
Sarcasm & Irony Detection System using NLP

Configurable preprocessing system with predefined variants
for controlled experimentation.

Key principle: Do NOT hard-code one irreversible preprocessing pipeline.
Each setting can be toggled independently.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional
import json
import copy


@dataclass
class PreprocessingConfig:
    """
    Configuration for text preprocessing pipeline.
    
    Each boolean flag controls whether a specific preprocessing
    step is applied. Default settings preserve maximum information.
    """
    
    # --- Identification ---
    name: str = "custom"
    description: str = ""
    
    # --- Text Normalization ---
    normalize_whitespace: bool = True
    lowercase: bool = False
    
    # --- Social Media ---
    normalize_urls: bool = True
    url_placeholder: str = "URL"
    normalize_mentions: bool = True
    mention_placeholder: str = "USER_MENTION"
    
    # Hashtag handling: "preserve" keeps #tag, "strip" removes # prefix,
    # "segment" attempts CamelCase segmentation
    hashtag_mode: str = "preserve"
    
    # Emoji handling: "preserve" keeps emojis, "text" converts to text description,
    # "remove" removes emojis
    emoji_mode: str = "preserve"
    
    # --- Punctuation ---
    # "preserve" keeps all punctuation, "normalize" collapses repeated punctuation,
    # "remove" removes all punctuation
    punctuation_mode: str = "preserve"
    
    # --- Repeated Characters ---
    normalize_repeated_chars: bool = False
    max_char_repeat: int = 2  # e.g., "soooo" -> "soo"
    
    # --- Negation ---
    protect_negations: bool = True
    
    # --- Stopwords ---
    remove_stopwords: bool = False
    
    # --- Morphological Normalization ---
    lemmatize: bool = False
    stem: bool = False
    
    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return asdict(self)
    
    def to_json(self, filepath: Optional[str] = None) -> str:
        """Serialize config to JSON."""
        json_str = json.dumps(self.to_dict(), indent=2)
        if filepath:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(json_str)
        return json_str
    
    @classmethod
    def from_dict(cls, d: dict) -> "PreprocessingConfig":
        """Create config from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
    
    @classmethod
    def from_json(cls, filepath: str) -> "PreprocessingConfig":
        """Load config from JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))
    
    def validate(self) -> None:
        """Validate configuration for conflicting settings."""
        if self.lemmatize and self.stem:
            raise ValueError(
                "Cannot enable both lemmatization and stemming simultaneously. "
                "Choose one morphological normalization strategy."
            )
        if self.hashtag_mode not in ("preserve", "strip", "segment"):
            raise ValueError(f"Invalid hashtag_mode: {self.hashtag_mode}")
        if self.emoji_mode not in ("preserve", "text", "remove"):
            raise ValueError(f"Invalid emoji_mode: {self.emoji_mode}")
        if self.punctuation_mode not in ("preserve", "normalize", "remove"):
            raise ValueError(f"Invalid punctuation_mode: {self.punctuation_mode}")


# ============================================================
# PREDEFINED PREPROCESSING VARIANTS
# ============================================================

def variant_a_minimal() -> PreprocessingConfig:
    """
    Variant A — Minimal Social Media
    
    Preserve maximum linguistic information.
    Only normalize URLs, mentions, and whitespace.
    """
    return PreprocessingConfig(
        name="minimal",
        description="Minimal preprocessing - preserve maximum linguistic information",
        normalize_whitespace=True,
        lowercase=False,
        normalize_urls=True,
        normalize_mentions=True,
        hashtag_mode="preserve",
        emoji_mode="preserve",
        punctuation_mode="preserve",
        normalize_repeated_chars=False,
        protect_negations=True,
        remove_stopwords=False,
        lemmatize=False,
        stem=False,
    )


def variant_b_lowercase() -> PreprocessingConfig:
    """
    Variant B — Lowercase
    
    Same as Variant A, but with lowercasing enabled.
    """
    config = variant_a_minimal()
    config.name = "lowercase"
    config.description = "Variant A + lowercase"
    config.lowercase = True
    return config


def variant_c_lemmatized() -> PreprocessingConfig:
    """
    Variant C — Lemmatized
    
    Variant A plus lemmatization.
    """
    config = variant_a_minimal()
    config.name = "lemmatized"
    config.description = "Variant A + lemmatization"
    config.lemmatize = True
    return config


def variant_d_stopword() -> PreprocessingConfig:
    """
    Variant D — Stopword Experiment
    
    Variant A plus stopword removal with negation protection.
    """
    config = variant_a_minimal()
    config.name = "stopword"
    config.description = "Variant A + stopword removal (negation protected)"
    config.remove_stopwords = True
    config.protect_negations = True
    return config


def variant_e_stemmed() -> PreprocessingConfig:
    """
    Variant E — Stemmed
    
    Variant A plus stemming.
    """
    config = variant_a_minimal()
    config.name = "stemmed"
    config.description = "Variant A + stemming"
    config.stem = True
    return config


def variant_f_classical() -> PreprocessingConfig:
    """
    Variant F — Classical Clean
    
    Conventional NLP preprocessing for comparison:
    lowercase, stopword removal (with negation protection),
    lemmatization, normalized punctuation.
    """
    return PreprocessingConfig(
        name="classical",
        description="Classical NLP preprocessing for baseline comparison",
        normalize_whitespace=True,
        lowercase=True,
        normalize_urls=True,
        normalize_mentions=True,
        hashtag_mode="strip",
        emoji_mode="text",
        punctuation_mode="normalize",
        normalize_repeated_chars=True,
        max_char_repeat=2,
        protect_negations=True,
        remove_stopwords=True,
        lemmatize=True,
        stem=False,
    )


def get_all_variants() -> dict:
    """Return all predefined preprocessing variants."""
    return {
        "minimal": variant_a_minimal(),
        "lowercase": variant_b_lowercase(),
        "lemmatized": variant_c_lemmatized(),
        "stopword": variant_d_stopword(),
        "stemmed": variant_e_stemmed(),
        "classical": variant_f_classical(),
    }


def get_variant(name: str) -> PreprocessingConfig:
    """Get a specific preprocessing variant by name."""
    variants = get_all_variants()
    if name not in variants:
        raise ValueError(
            f"Unknown variant '{name}'. Available: {list(variants.keys())}"
        )
    return variants[name]
