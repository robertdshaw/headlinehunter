import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
from textstat import flesch_reading_ease
import warnings

warnings.filterwarnings("ignore")

PREP_DIR = Path("data/preprocessed")


def load_preprocessing_components():
    """Load preprocessing components for exact feature replication"""
    try:
        with open(
            PREP_DIR / "processed_data" / "preprocessing_metadata.json", "r"
        ) as f:
            preprocessing_metadata = json.load(f)

        with open(PREP_DIR / "processed_data" / "category_encoder.pkl", "rb") as f:
            category_encoder = pickle.load(f)

        pca_transformer = None
        pca_file = PREP_DIR / "processed_data" / "pca_transformer.pkl"
        if pca_file.exists():
            with open(pca_file, "rb") as f:
                pca_transformer = pickle.load(f)

        return {
            "preprocessing_metadata": preprocessing_metadata,
            "category_encoder": category_encoder,
            "pca_transformer": pca_transformer,
            "training_median_ctr": preprocessing_metadata.get(
                "training_median_ctr", 0.030
            ),
            "feature_order": preprocessing_metadata.get("available_features", []),
        }
    except Exception as e:
        print(f"Error loading preprocessing components: {e}")
        return None


_embedder = None


def get_embedder():
    """Get cached sentence transformer"""
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


def create_article_features_exact(title, abstract="", category="news", components=None):
    """Create features for a single article using EXACT replication of preprocessing pipeline"""

    # Editorial criteria (must match EDA_preprocess_features.py)
    EDITORIAL_CRITERIA = {
        "target_reading_ease": 60,
        "readability_weight": 0.3,
        "engagement_weight": 0.4,
        "headline_quality_weight": 0.2,
        "timeliness_weight": 0.1,
        "target_ctr_gain": 0.05,
        "optimal_word_count": (8, 12),
        "max_title_length": 75,
    }

    features = {}

    # Basic text features
    features["title_length"] = len(title)
    features["abstract_length"] = len(abstract)
    features["title_word_count"] = len(title.split())
    features["abstract_word_count"] = len(abstract.split())

    # Flesch Reading Ease
    features["title_reading_ease"] = flesch_reading_ease(title) if title else 0
    features["abstract_reading_ease"] = flesch_reading_ease(abstract) if abstract else 0

    # Headline Quality Indicators
    features["has_question"] = 1 if "?" in title else 0
    features["has_exclamation"] = 1 if "!" in title else 0
    features["has_number"] = 1 if any(c.isdigit() for c in title) else 0
    features["has_colon"] = 1 if ":" in title else 0
    features["has_quotes"] = 1 if any(q in title for q in ['"', "'", '"', '"']) else 0
    features["has_dash"] = 1 if any(d in title for d in ["-", "–", "—"]) else 0

    # Advanced headline metrics
    features["title_upper_ratio"] = (
        sum(c.isupper() for c in title) / len(title) if title else 0
    )
    features["title_caps_words"] = len(
        [w for w in title.split() if w.isupper() and len(w) > 1]
    )
    features["avg_word_length"] = (
        np.mean([len(word) for word in title.split()]) if title.split() else 0
    )

    # Content depth indicators
    features["has_abstract"] = 1 if len(abstract) > 0 else 0
    features["title_abstract_ratio"] = features["title_length"] / (
        features["abstract_length"] + 1
    )

    # Editorial scores
    features["editorial_readability_score"] = (
        np.clip(features["title_reading_ease"] / 100, 0, 1)
        * EDITORIAL_CRITERIA["readability_weight"]
    )
    features["editorial_headline_score"] = (
        (features["has_question"] + features["has_number"] + features["has_colon"])
        / 3
        * EDITORIAL_CRITERIA["headline_quality_weight"]
    )

    # Editorial quality flags
    features["needs_readability_improvement"] = (
        1
        if features["title_reading_ease"] < EDITORIAL_CRITERIA["target_reading_ease"]
        else 0
    )
    features["suboptimal_word_count"] = (
        1
        if (
            features["title_word_count"] < EDITORIAL_CRITERIA["optimal_word_count"][0]
            or features["title_word_count"]
            > EDITORIAL_CRITERIA["optimal_word_count"][1]
        )
        else 0
    )
    features["too_long_title"] = (
        1 if features["title_length"] > EDITORIAL_CRITERIA["max_title_length"] else 0
    )

    # Category encoding
    if components and components["category_encoder"] is not None:
        try:
            category_encoder = components["category_encoder"]
            category_clean = (
                str(category).replace("nan", "unknown")
                if pd.notna(category)
                else "unknown"
            )

            if category_clean in category_encoder.classes_:
                features["category_enc"] = category_encoder.transform([category_clean])[
                    0
                ]
            else:
                features["category_enc"] = (
                    category_encoder.transform(["unknown"])[0]
                    if "unknown" in category_encoder.classes_
                    else 0
                )
        except Exception as e:
            print(f"Category encoding failed for '{category}': {e}")
            features["category_enc"] = 0
    else:
        features["category_enc"] = 0

    # Create title embeddings
    try:
        embedder = get_embedder()
        title_embedding = embedder.encode([title])[0]

        for i, emb_val in enumerate(title_embedding[:384]):
            features[f"title_emb_{i}"] = float(emb_val)

        if components and components["pca_transformer"] is not None:
            embedding_matrix = np.array([title_embedding[:384]]).astype(np.float32)
            pca_embeddings = components["pca_transformer"].transform(embedding_matrix)[
                0
            ]
            for i, pca_val in enumerate(pca_embeddings):
                features[f"title_pca_{i}"] = float(pca_val)

    except Exception as e:
        print(f"Could not create embeddings for title: {e}")
        for i in range(384):
            features[f"title_emb_{i}"] = 0.0
        if components and components["pca_transformer"] is not None:
            for i in range(components["pca_transformer"].n_components_):
                features[f"title_pca_{i}"] = 0.0

    return features
