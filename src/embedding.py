"""
src/embedding.py — Quilter HNW Advisor Assistant 

EmbeddingEngine with L1 explainability via Leave-One-Out token attribution.
Extracted from notebook cell 9 into a standalone module.

Model hierarchy:
  1. BAAI/bge-large-en-v1.5 (1024-dim, SOTA MTEB retrieval)
  2. all-MiniLM-L6-v2 (384-dim, lightweight fallback)
  3. Deterministic seeded random vectors (demo/test fallback)
"""

from __future__ import annotations

import hashlib
import logging
from typing import TYPE_CHECKING, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
    ST_OK = True
except ImportError:
    SentenceTransformer = None  # type: ignore[assignment,misc]
    ST_OK = False
    logger.warning("sentence-transformers not installed — using random embedding fallback")

if TYPE_CHECKING:
    from src.config import Config


class EmbeddingEngine:
    """
    Dense text embedding with BGE-large-en-v1.5 (primary) or fallback.

    Public API:
      embed(texts)            → np.ndarray shape (N, dim), L2-normalised
      embed_single(text)      → np.ndarray shape (dim,)
      token_importance(query) → List[(token, importance_score)]

    L1 Explainability:
      token_importance() implements Leave-One-Out (LOO) attribution.
      Each query token is ablated; cosine delta vs full embedding measures importance.
      Scores are normalised to sum=1.
    """

    def __init__(self, cfg: "Config") -> None:
        self.cfg = cfg
        self.model = None
        self.dim = 384
        self.model_name_used = "random_fallback"

        if not ST_OK:
            logger.warning("EmbeddingEngine: using deterministic random vectors (demo mode)")
            return

        for model_name in [cfg.embed_model, cfg.embed_fallback]:
            try:
                self.model = SentenceTransformer(model_name)
                self.dim = self.model.get_sentence_embedding_dimension()
                self.model_name_used = model_name
                logger.info("EmbeddingEngine loaded: %s (dim=%d)", model_name, self.dim)
                break
            except Exception as exc:
                logger.warning("Could not load %s: %s", model_name, exc)
                continue

        if self.model is None:
            logger.warning("All embedding models failed — using deterministic random vectors")

    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Encode list of texts. Returns L2-normalised vectors shape (N, dim).
        Falls back to deterministic seeded random vectors if model unavailable.
        """
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)

        if self.model is not None:
            try:
                vecs = self.model.encode(
                    texts,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    batch_size=32,
                )
                return np.array(vecs, dtype=np.float32)
            except Exception as exc:
                logger.error("Embedding encode failed: %s — falling back to random", exc)

        # Deterministic random fallback: same text → same vector across calls
        vecs = []
        for text in texts:
            seed = int(hashlib.md5(text.encode()).hexdigest()[:8], 16) % (2**32)
            rng = np.random.default_rng(seed)
            v = rng.standard_normal(self.dim).astype(np.float32)
            norm = np.linalg.norm(v)
            vecs.append(v / norm if norm > 0 else v)
        return np.array(vecs, dtype=np.float32)

    def embed_single(self, text: str) -> np.ndarray:
        """Convenience: embed one text, return shape (dim,)."""
        return self.embed([text])[0]

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two L2-normalised vectors."""
        # Vectors are already L2-normalised so dot product = cosine
        return float(np.dot(a, b))

    def token_importance(
        self,
        query: str,
        top_n: int = 8,
    ) -> List[Tuple[str, float]]:
        """
        L1 Explainability: Leave-One-Out (LOO) token attribution.

        Algorithm:
          1. Embed full query → q_vec
          2. For each token t in query:
               ablated = query with t removed
               ablated_vec = embed(ablated)
               delta[t] = cosine(q_vec, q_vec) - cosine(ablated_vec, q_vec)
          3. Normalise deltas to sum=1 (importance distribution)
          4. Return top_n tokens sorted by importance (descending)

        Tokens with high importance are the primary retrieval drivers.
        If wrong chunks are retrieved, check if these tokens appear in the index.
        """
        tokens = query.split()
        if not tokens:
            return []

        q_vec = self.embed_single(query)
        baseline = self._cosine(q_vec, q_vec)  # = 1.0 (normalised)

        deltas: List[Tuple[str, float]] = []
        for i, token in enumerate(tokens):
            ablated_tokens = tokens[:i] + tokens[i + 1:]
            if not ablated_tokens:
                deltas.append((token, 1.0))
                continue
            ablated_text = " ".join(ablated_tokens)
            ablated_vec = self.embed_single(ablated_text)
            delta = baseline - self._cosine(ablated_vec, q_vec)
            deltas.append((token, max(0.0, delta)))

        # Normalise to sum=1
        total = sum(d for _, d in deltas)
        if total > 0:
            deltas = [(t, d / total) for t, d in deltas]

        # Sort descending, return top_n
        deltas.sort(key=lambda x: x[1], reverse=True)
        return deltas[:top_n]
