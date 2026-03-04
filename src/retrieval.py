"""
src/retrieval.py — Quilter HNW Advisor Assistant 

HybridIndex: FAISS dense + BM25 sparse + RRF fusion + CrossEncoder reranking.


"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np

from src.models import Chunk, RetrievalResult

logger = logging.getLogger(__name__)

try:
    import faiss
    FAISS_OK = True
except ImportError:
    faiss = None  # type: ignore[assignment]
    FAISS_OK = False
    logger.warning("faiss-cpu not installed — using numpy dot-product fallback")

try:
    from rank_bm25 import BM25Okapi
    BM25_OK = True
except ImportError:
    BM25Okapi = None  # type: ignore[assignment]
    BM25_OK = False
    logger.warning("rank-bm25 not installed — sparse search disabled")

try:
    from sentence_transformers import CrossEncoder
    ST_OK = True
except ImportError:
    CrossEncoder = None  # type: ignore[assignment]
    ST_OK = False

if TYPE_CHECKING:
    from src.config import Config
    from src.embedding import EmbeddingEngine


class HybridIndex:
    """
    Hybrid retrieval index: dense (FAISS) + sparse (BM25) + RRF + CrossEncoder rerank.

    Added in v3 (gap fixes):
      _hyde_expand() — GAP-04: Hypothetical Document Embedding for query expansion
      _mmr()         — GAP-05: Maximal Marginal Relevance post-rerank diversification

    Build with:  build(chunks)
    Search with: search(query, top_k, cfg)
    Persist with: save(path) / load(path)
    """

    def __init__(self, cfg: "Config", emb: "EmbeddingEngine") -> None:
        self.cfg   = cfg
        self.emb   = emb
        self.chunks: List[Chunk] = []

        # Dense index
        self._faiss_index = None
        self._dense_mat: Optional[np.ndarray] = None  # (N, dim) — used by MMR

        # Sparse index
        self._bm25: Optional["BM25Okapi"] = None

        # Cross-encoder reranker
        self._reranker = None

    # Build


    def _load_reranker(self) -> None:
        if self._reranker is not None or not ST_OK:
            return
        try:
            self._reranker = CrossEncoder(self.cfg.rerank_model)
            logger.info("CrossEncoder loaded: %s", self.cfg.rerank_model)
        except Exception as exc:
            logger.warning("Could not load reranker %s: %s", self.cfg.rerank_model, exc)

    def build(self, chunks: List[Chunk]) -> None:
        """
        Build dense (FAISS) and sparse (BM25) indexes from chunks.
        Embeds parent_context (section heading + text) for richer retrieval.
        Stores _dense_mat for MMR (GAP-05) and BM25Okapi for sparse search.
        """
        if not chunks:
            logger.warning("build() called with empty chunk list")
            return

        self.chunks = chunks
        texts = [c.parent_context for c in chunks]

        logger.info("Building index for %d chunks...", len(chunks))

        vecs = self.emb.embed(texts)  # (N, dim), L2-normalised
        self._dense_mat = vecs

        if FAISS_OK:
            dim = vecs.shape[1]
            self._faiss_index = faiss.IndexFlatIP(dim)  # Inner product = cosine (normalised)
            self._faiss_index.add(vecs.astype(np.float32))
            logger.info("FAISS IndexFlatIP built: %d vectors, dim=%d", len(chunks), dim)
        else:
            logger.info("Using numpy dot-product fallback (FAISS not available)")

        if BM25_OK:
            tokenised = [c.text.lower().split() for c in chunks]
            self._bm25 = BM25Okapi(tokenised)
            logger.info("BM25Okapi built: %d documents", len(chunks))
        else:
            logger.warning("BM25 not available — dense-only retrieval")

        self._load_reranker()
        logger.info("Index built successfully")

    # Persistence


    def save(self, path: str) -> None:
        """Pickle index state; save FAISS index file."""
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)

        state = {
            "chunks":      self.chunks,
            "dense_mat":   self._dense_mat,
        }
        (p / "index_state.pkl").write_bytes(pickle.dumps(state))

        if FAISS_OK and self._faiss_index is not None:
            faiss.write_index(self._faiss_index, str(p / "faiss.index"))

        if BM25_OK and self._bm25 is not None:
            (p / "bm25.pkl").write_bytes(pickle.dumps(self._bm25))

        logger.info("Index saved to %s", path)

    def load(self, path: str) -> bool:
        """Restore index from disk. Returns True on success."""
        p = Path(path)
        state_file = p / "index_state.pkl"
        if not state_file.exists():
            logger.info("No saved index at %s", path)
            return False

        try:
            state = pickle.loads(state_file.read_bytes())
            self.chunks     = state["chunks"]
            self._dense_mat = state.get("dense_mat")

            if FAISS_OK:
                faiss_file = p / "faiss.index"
                if faiss_file.exists():
                    self._faiss_index = faiss.read_index(str(faiss_file))
                else:
                    logger.warning("FAISS index file missing — will use numpy fallback")

            bm25_file = p / "bm25.pkl"
            if BM25_OK and bm25_file.exists():
                self._bm25 = pickle.loads(bm25_file.read_bytes())

            self._load_reranker()
            logger.info("Index loaded: %d chunks", len(self.chunks))
            return True
        except Exception as exc:
            logger.error("Failed to load index: %s", exc)
            return False

    # Search components


    def _dense_search(self, qv: np.ndarray, k: int) -> List[Tuple[int, float]]:
        """
        Dense search using FAISS (inner product) or numpy fallback.
        Returns list of (chunk_index, score) sorted by score descending.
        """
        if not self.chunks or self._dense_mat is None:
            return []

        qv32 = qv.reshape(1, -1).astype(np.float32)

        if FAISS_OK and self._faiss_index is not None:
            scores, indices = self._faiss_index.search(qv32, min(k, len(self.chunks)))
            return [(int(i), float(s)) for i, s in zip(indices[0], scores[0]) if i >= 0]
        else:
            # Numpy fallback
            scores = self._dense_mat @ qv32.T  # (N, 1)
            scores = scores.flatten()
            top_k_idx = np.argsort(-scores)[:k]
            return [(int(i), float(scores[i])) for i in top_k_idx]

    def _sparse_search(self, query: str, k: int) -> List[Tuple[int, float]]:
        """
        BM25 sparse search. Returns list of (chunk_index, bm25_score) for positive scores.
        """
        if not BM25_OK or self._bm25 is None or not self.chunks:
            return []

        tokenised = query.lower().split()
        scores = self._bm25.get_scores(tokenised)
        top_k_idx = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in top_k_idx if scores[i] > 0]

    @staticmethod
    def _rrf(
        lists: List[List[Tuple[int, float]]],
        k: int = 60,
    ) -> Dict[int, float]:
        """
        Reciprocal Rank Fusion of multiple ranked lists.
        RRF score for item at rank r: 1 / (k + r)
        Returns {chunk_index: rrf_score} dict.
        """
        scores: Dict[int, float] = {}
        for ranked_list in lists:
            for rank, (idx, _) in enumerate(ranked_list, start=1):
                scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)
        return scores

    # GAP-04: HyDE — Hypothetical Document Embedding


    def _hyde_expand(self, query: str, cfg: "Config") -> np.ndarray:
        """
        GAP-04 fix: Hypothetical Document Embedding (HyDE) for query expansion.

        Algorithm:
          1. Call LLM to generate a hypothetical document excerpt answering the query
          2. Embed the hypothetical text → hyp_vec
          3. Embed the original query → q_vec
          4. Return (q_vec + hyp_vec) / 2, renormalised

        This bridges the vocabulary gap between query language ("what is the MPAA")
        and document language ("money purchase annual allowance of £10,000").

        Falls back to plain query embedding if LLM call fails.
        """
        from src.llm_client import call_fast, SYS

        try:
            hyp_prompt = SYS["hyde"].format(query=query)
            hyp_text = call_fast(
                system="Generate a factual document excerpt.",
                user=hyp_prompt,
                cfg=cfg,
                max_tokens=256,
            )
            if hyp_text.startswith("[LLM"):
                raise ValueError(f"LLM error: {hyp_text}")

            hyp_vec = self.emb.embed_single(hyp_text)
            q_vec   = self.emb.embed_single(query)

            # Average and renormalise
            combined = (q_vec + hyp_vec) / 2.0
            norm = np.linalg.norm(combined)
            if norm > 0:
                combined = combined / norm
            return combined

        except Exception as exc:
            logger.warning("HyDE expansion failed (%s) — using plain query embedding", exc)
            return self.emb.embed_single(query)

    # GAP-05: MMR — Maximal Marginal Relevance


    @staticmethod
    def _mmr(
        results: List[RetrievalResult],
        query_vec: np.ndarray,
        dense_mat: np.ndarray,
        chunk_to_idx: Dict[str, int],
        top_k: int,
        lambda_: float = 0.7,
    ) -> List[RetrievalResult]:
        """
        GAP-05 fix: Maximal Marginal Relevance diversification.

        Uses pre-computed _dense_mat (no re-encoding — O(n) not O(n²)).

        Algorithm:
          selected = []
          candidates = copy of results
          while len(selected) < top_k and candidates:
            for each c in candidates:
              rel  = q_vec · embed(c)           [from dense_mat]
              div  = max(embed(s) · embed(c) for s in selected) if selected else 0
              mmr  = lambda_ * rel - (1 - lambda_) * div
            select c with highest mmr score
            selected.append(c), remove from candidates

        lambda_=0.7 → 70% relevance, 30% diversity (good for multi-domain HNW queries).
        """
        if not results or dense_mat is None:
            return results[:top_k]

        candidates = list(results)
        selected:   List[RetrievalResult] = []

        def get_vec(r: RetrievalResult) -> Optional[np.ndarray]:
            idx = chunk_to_idx.get(r.chunk.chunk_id)
            if idx is not None and idx < len(dense_mat):
                return dense_mat[idx]
            return None

        while len(selected) < top_k and candidates:
            best_score = float("-inf")
            best_result = candidates[0]

            for cand in candidates:
                cand_vec = get_vec(cand)
                if cand_vec is None:
                    # No embedding cached — use rrf_score as proxy
                    rel = cand.rrf_score
                    div = 0.0
                else:
                    rel = float(np.dot(query_vec, cand_vec))
                    if selected:
                        divs = []
                        for sel in selected:
                            sel_vec = get_vec(sel)
                            if sel_vec is not None:
                                divs.append(float(np.dot(sel_vec, cand_vec)))
                        div = max(divs) if divs else 0.0
                    else:
                        div = 0.0

                mmr_score = lambda_ * rel - (1 - lambda_) * div

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_result = cand

            selected.append(best_result)
            candidates.remove(best_result)

        return selected

    # Main search pipeline


    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        cfg: Optional["Config"] = None,
    ) -> List[RetrievalResult]:
        """
        Full retrieval pipeline:
          1. HyDE query expansion (if cfg.use_hyde and cfg provided)
          2. Dense FAISS search with (possibly expanded) query vector
          3. Sparse BM25 search with original query string
          4. RRF fusion
          5. CrossEncoder reranking on top 2*top_k candidates
          6. MMR diversification on reranked results
          7. Return top_k results

        cfg is optional to preserve backwards compatibility with demo mode.
        When cfg=None, HyDE and MMR are skipped.
        """
        if not self.chunks:
            return []

        _cfg = cfg or self.cfg
        _top_k = top_k or _cfg.top_k_rerank

        if cfg is not None and cfg.use_hyde:
            q_vec = self._hyde_expand(query, cfg)
        else:
            q_vec = self.emb.embed_single(query)

        dense_hits = self._dense_search(q_vec, _cfg.top_k_dense)

        sparse_hits = self._sparse_search(query, _cfg.top_k_sparse)

        rrf_scores = self._rrf([dense_hits, sparse_hits], k=_cfg.rrf_k)

        # Build dense/sparse rank dicts
        dense_rank_map  = {idx: rank for rank, (idx, _) in enumerate(dense_hits,  1)}
        dense_score_map = {idx: sc   for idx, sc          in dense_hits}
        sparse_rank_map = {idx: rank for rank, (idx, _) in enumerate(sparse_hits, 1)}
        bm25_score_map  = {idx: sc   for idx, sc          in sparse_hits}

        # Sort by RRF score descending
        candidate_idxs = sorted(rrf_scores, key=lambda i: rrf_scores[i], reverse=True)
        candidate_idxs = candidate_idxs[:_top_k * 2]  # Take 2× for reranking

        candidates: List[RetrievalResult] = []
        for idx in candidate_idxs:
            r = RetrievalResult(
                chunk        = self.chunks[idx],
                dense_rank   = dense_rank_map.get(idx),
                sparse_rank  = sparse_rank_map.get(idx),
                dense_score  = dense_score_map.get(idx),
                bm25_score   = bm25_score_map.get(idx),
                rrf_score    = rrf_scores[idx],
                rerank_score = None,
            )
            candidates.append(r)

        if self._reranker is not None and candidates:
            try:
                pairs = [(query, r.chunk.text) for r in candidates]
                scores = self._reranker.predict(pairs, show_progress_bar=False)
                # Sigmoid normalisation to [0, 1]
                import math as _math
                for r, sc in zip(candidates, scores):
                    r.rerank_score = 1 / (1 + _math.exp(-float(sc)))
                candidates.sort(key=lambda r: r.rerank_score or 0.0, reverse=True)
            except Exception as exc:
                logger.warning("Reranking failed: %s", exc)
                candidates.sort(key=lambda r: r.rrf_score, reverse=True)
        else:
            candidates.sort(key=lambda r: r.rrf_score, reverse=True)

        # Assign final ranks
        for i, r in enumerate(candidates, 1):
            r.final_rank = i

        if cfg is not None and self._dense_mat is not None:
            chunk_to_idx = {c.chunk_id: i for i, c in enumerate(self.chunks)}
            final_results = self._mmr(
                candidates,
                q_vec,
                self._dense_mat,
                chunk_to_idx,
                top_k=_top_k,
                lambda_=_cfg.mmr_lambda,
            )
        else:
            final_results = candidates[:_top_k]

        # Assign final ranks after MMR
        for i, r in enumerate(final_results, 1):
            r.final_rank = i

        logger.debug(
            "search('%s...') → %d results, max_rrf=%.4f",
            query[:40], len(final_results),
            max((r.rrf_score for r in final_results), default=0.0),
        )
        return final_results
