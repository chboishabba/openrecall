from typing import Any, List, Optional
import hashlib
import os
import logging
import re

import numpy as np

logger = logging.getLogger(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Constants
MODEL_NAME: str = "all-MiniLM-L6-v2"
EMBEDDING_DIM: int = 384  # Dimension for all-MiniLM-L6-v2
_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9']+")
_SENTENCE_TRANSFORMER_MODEL: Optional[Any] = None
_SENTENCE_TRANSFORMER_LOAD_ATTEMPTED: bool = False


def _get_sentence_transformer_model() -> Optional[Any]:
    """Lazily load and cache the optional sentence-transformers model."""
    global _SENTENCE_TRANSFORMER_MODEL
    global _SENTENCE_TRANSFORMER_LOAD_ATTEMPTED

    if _SENTENCE_TRANSFORMER_LOAD_ATTEMPTED:
        return _SENTENCE_TRANSFORMER_MODEL

    _SENTENCE_TRANSFORMER_LOAD_ATTEMPTED = True
    mode = os.environ.get("OPENRECALL_EMBEDDING_BACKEND", "auto").lower()
    if mode == "local":
        logger.info(
            "OPENRECALL_EMBEDDING_BACKEND=local; skipping SentenceTransformer load."
        )
        return None

    try:
        from sentence_transformers import SentenceTransformer
    except Exception as exc:
        logger.warning(
            "SentenceTransformer dependency is unavailable; using local embedding "
            f"fallback instead. ({exc})"
        )
        return None

    try:
        _SENTENCE_TRANSFORMER_MODEL = SentenceTransformer(MODEL_NAME)
        logger.info(f"SentenceTransformer model '{MODEL_NAME}' loaded successfully.")
    except Exception as exc:  # pragma: no cover - load failure varies by host
        logger.error(
            f"Failed to load SentenceTransformer model '{MODEL_NAME}': {exc}"
        )
        _SENTENCE_TRANSFORMER_MODEL = None
    return _SENTENCE_TRANSFORMER_MODEL


def get_embedding(text: str) -> np.ndarray:
    """
    Generates a sentence embedding for the given text.

    Uses the optional `sentence-transformers` model when available and
    requested, then falls back to a deterministic local hash embedding when the
    model is unavailable or unavailable by configuration.

    Args:
        text: The input string to embed.

    Returns:
        A numpy array representing the mean embedding of the text lines,
        or a zero vector if the input is empty, whitespace only, or the
        model failed to load. The array type is float32.
    """
    if not text or text.isspace():
        logger.warning("Input text is empty or whitespace. Returning zero vector.")
        return np.zeros(EMBEDDING_DIM, dtype=np.float32)

    # Split text into non-empty lines
    sentences = [line for line in text.split("\n") if line.strip()]

    if not sentences:
        logger.warning("No non-empty lines found after splitting. Returning zero vector.")
        return np.zeros(EMBEDDING_DIM, dtype=np.float32)

    model = _get_sentence_transformer_model()
    if model is not None:
        try:
            sentence_embeddings = model.encode(sentences)
            # Calculate the mean embedding
            return np.mean(sentence_embeddings, axis=0, dtype=np.float32)
        except Exception as exc:
            logger.error(f"Error generating embedding with SentenceTransformer: {exc}")

    return _hash_embedding(sentences)


def _hash_embedding(lines: List[str]) -> np.ndarray:
    """Build a deterministic local embedding when transformer models are unavailable."""
    vector = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    for line in lines:
        for token in _TOKEN_PATTERN.findall(line.lower()):
            token_bytes = token.encode("utf-8", errors="ignore")
            token_digest = hashlib.blake2s(token_bytes, digest_size=4).digest()
            idx = int.from_bytes(token_digest[:2], byteorder="little", signed=False)
            idx %= EMBEDDING_DIM
            vector[idx] += 1.0 if token_digest[2] % 2 == 0 else -1.0

    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return (vector / norm).astype(np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Calculates the cosine similarity between two numpy vectors.

    Args:
        a: The first numpy array.
        b: The second numpy array.

    Returns:
        The cosine similarity score (float between -1 and 1),
        or 0.0 if either vector has zero magnitude.
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        logger.warning("One or both vectors have zero magnitude. Returning 0 similarity.")
        return 0.0

    similarity = np.dot(a, b) / (norm_a * norm_b)
    # Clip values to handle potential floating-point inaccuracies slightly outside [-1, 1]
    return float(np.clip(similarity, -1.0, 1.0))
