"""
src/faithfulness.py — Quilter HNW Advisor Assistant 

Sentence-level NLI faithfulness evaluation (L3 Explainability).
Writes SentenceAttribution records to separate sentence_attribution.jsonl
linked to audit_log.jsonl via query_id.

 NLI now compares each answer sentence against the TEXT of the
best-matching retrieved chunk. source_chunk_id,
source_file, and source_page are now populated from retrieval results.
The best-matching chunk per sentence is found by computing
token-overlap between the sentence and each candidate chunk's text, then
using that chunk as the NLI premise. This gives genuine ENTAILMENT/NEUTRAL/
CONTRADICTION labels rather than NEUTRAL for everything.

NLI model: cross-encoder/nli-deberta-v3-small
Labels: CONTRADICTION, ENTAILMENT ,NEUTRAL
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Tuple

from src.models import FaithfulnessReport, RagTriadReport, SentenceAttribution, SentenceDetail

if TYPE_CHECKING:
    from src.models import RetrievalResult

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import CrossEncoder
    NLI_OK = True
except ImportError:
    CrossEncoder = None  # type: ignore[assignment]
    NLI_OK = False
    logger.warning("sentence-transformers not available — NLI heuristic fallback will be used")

NLI_MODEL_NAME = "cross-encoder/nli-deberta-v3-small"
# DeBERTa NLI label order: 0=contradiction, 1=entailment, 2=neutral
_LABEL_MAP = {0: "CONTRADICTION", 1: "ENTAILMENT", 2: "NEUTRAL"}


class FaithfulnessEvaluator:
    """
    Per-sentence NLI faithfulness evaluator.

    Evaluates each sentence in the model's answer against the retrieved context.
    Produces a FaithfulnessReport with per-sentence ENTAILMENT/NEUTRAL/CONTRADICTION labels.

    L3 Explainability: NEUTRAL sentences are flagged as the primary hallucination risk surface.
    NeMo output rail and the Fact-Check Agent both target NEUTRAL sentences for elimination.

    GAP-15: sentence-level attributions written to separate sentence_attribution.jsonl.
    """

    def __init__(self, cfg=None) -> None:
        """
        Args:
            cfg: Optional Config instance. When provided, log filenames are read
                 from cfg (e.g. cfg.log_attribution) rather than hardcoded strings.
        """
        self.cfg = cfg
        self._model = None
        self._model_ok = False

        if NLI_OK:
            try:
                self._model = CrossEncoder(NLI_MODEL_NAME, num_labels=3)
                self._model_ok = True
                logger.info("NLI model loaded: %s", NLI_MODEL_NAME)
            except Exception as exc:
                logger.warning("Could not load NLI model %s: %s — using heuristic", NLI_MODEL_NAME, exc)

    # Sentence splitting


    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        """
        Split answer text into sentences.
        Filters: must be >= 4 words, not pure citation lines, not empty.
        """
        # Split on sentence-ending punctuation followed by space or newline
        raw = re.split(r'(?<=[.!?])\s+|\n', text)
        sentences = []
        for s in raw:
            s = s.strip()
            if len(s.split()) < 4:
                continue
            # Skip pure citation lines like "[Source: quilter_charges.pdf, p.1]"
            if re.match(r'^\[Source:', s):
                continue
            # Skip pure header/working lines (e.g., "Working:", "Sources:")
            if re.match(r'^(?:Working|Sources?|Result|Note):\s*$', s, re.I):
                continue
            sentences.append(s)
        return sentences

    # Heuristic NLI fallback


    @staticmethod
    def _heuristic(sentence: str, context: str) -> Tuple[str, float]:
        """
        Word-overlap heuristic NLI when DeBERTa is unavailable.

        Improved over notebook version: includes negation detection.
        Uses stopword-filtered unigrams.
        """
        STOPWORDS = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "shall", "to",
            "of", "in", "for", "on", "with", "at", "by", "from", "as",
            "into", "through", "during", "before", "after", "above",
            "below", "between", "each", "or", "and", "but", "if", "then",
            "that", "this", "these", "those", "it", "its",
        }
        NEGATIONS = {"not", "no", "never", "cannot", "isn't", "aren't",
                     "wasn't", "weren't", "don't", "doesn't", "didn't",
                     "won't", "wouldn't", "couldn't", "shouldn't"}

        def tokens(text: str) -> set:
            words = re.findall(r'\b\w+\b', text.lower())
            return {w for w in words if w not in STOPWORDS}

        s_tokens  = tokens(sentence)
        ctx_tokens = tokens(context)

        # Check negation: if sentence has negation words not in context, flag NEUTRAL
        s_negs   = s_tokens & NEGATIONS
        ctx_negs = ctx_tokens & NEGATIONS
        has_novel_negation = bool(s_negs - ctx_negs)

        if not s_tokens:
            return "NEUTRAL", 0.50

        overlap = len(s_tokens & ctx_tokens) / len(s_tokens)

        if has_novel_negation and overlap < 0.80:
            # Sentence introduces negation not in context — suspect
            return "NEUTRAL", 0.45

        if overlap >= 0.55:
            return "ENTAILMENT", min(0.55 + overlap * 0.3, 0.90)
        elif overlap >= 0.25:
            return "NEUTRAL", 0.45
        else:
            return "CONTRADICTION", 0.35

    # Best-chunk finder (BUG-FAITH-01 / BUG-FAITH-02 fix)


    @staticmethod
    def _best_chunk_for_sentence(
        sentence: str,
        results: "List[RetrievalResult]",
    ) -> "Optional[RetrievalResult]":
        """
        BUG-FAITH-01 fix: Find the retrieved chunk whose text has the highest
        token-overlap with the answer sentence.

        Previously source_chunk_id was always "". Now we bind each sentence to
        the most semantically relevant chunk so the NLI premise is actual source
        text, not an empty string.

        Returns the RetrievalResult with the highest unigram overlap, or the
        top-ranked result if no overlap is found.
        """
        if not results:
            return None

        STOPWORDS = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "shall", "to",
            "of", "in", "for", "on", "with", "at", "by", "from", "as",
        }

        def _tokens(text: str) -> set:
            words = re.findall(r'\b\w+\b', text.lower())
            return {w for w in words if w not in STOPWORDS and len(w) > 2}

        sent_tokens = _tokens(sentence)
        if not sent_tokens:
            return results[0]

        best_result = results[0]
        best_overlap = -1.0
        for r in results:
            chunk_tokens = _tokens(r.chunk.text)
            if not chunk_tokens:
                continue
            overlap = len(sent_tokens & chunk_tokens) / len(sent_tokens)
            if overlap > best_overlap:
                best_overlap = overlap
                best_result = r

        return best_result

    # Main evaluation


    def evaluate(
        self,
        answer: str,
        context: str,
        is_fallback: bool = False,
        query_id: str = "",
        log_dir: Optional[str] = None,
        results: "Optional[List[RetrievalResult]]" = None,
    ) -> FaithfulnessReport:
        """
        Evaluate faithfulness of answer against context.

        BUG-FAITH-01/02 fix: `results` (List[RetrievalResult]) is now accepted
        so that each answer sentence is compared against the TEXT of its best-
        matching retrieved chunk, not against an empty string. This gives genuine
        ENTAILMENT/NEUTRAL/CONTRADICTION labels and populates source_chunk_id,
        source_file, and source_page in sentence_attribution.jsonl.

        For each sentence:
          - Find the best-matching chunk from `results` by token overlap
          - Use chunk.text as the NLI premise (not the concatenated context string)
          - If DeBERTa available: NLI classification via CrossEncoder
          - Else: heuristic word-overlap classification

        GAP-15 fix: if query_id and log_dir are provided, writes per-sentence
        SentenceAttribution records to {log_dir}/sentence_attribution.jsonl.
        These are linked to audit_log.jsonl via query_id.

        Returns FaithfulnessReport with per-sentence labels.
        """
        sentences = self._split_sentences(answer)
        if not sentences:
            report = FaithfulnessReport(
                overall_score=1.0,
                sentence_scores=[],
                unsupported=[],
                is_fallback=is_fallback,
            )
            return report

        details: List[SentenceDetail] = []
        attributions: List[SentenceAttribution] = []

        # BUG-FAITH-01: use per-sentence best-chunk text as NLI premise
        use_per_chunk = results is not None and len(results) > 0

        if self._model_ok and self._model is not None:
            try:
                import numpy as np

                if use_per_chunk:
                    # BUG-FAITH-02: pair each sentence with its best-matching chunk text
                    best_chunks = [
                        self._best_chunk_for_sentence(s, results) for s in sentences
                    ]
                    pairs = [
                        (s, bc.chunk.text if bc else context)
                        for s, bc in zip(sentences, best_chunks)
                    ]
                else:
                    best_chunks = [None] * len(sentences)
                    pairs = [(s, context) for s in sentences]

                raw_scores = self._model.predict(pairs, show_progress_bar=False)
                # raw_scores shape: (N, 3) — [contradiction, entailment, neutral]
                for i, (sentence, scores) in enumerate(zip(sentences, raw_scores)):
                    e = np.exp(scores - np.max(scores))
                    probs = e / e.sum()
                    label_idx = int(np.argmax(probs))
                    label = _LABEL_MAP[label_idx]
                    confidence = float(probs[label_idx])
                    supported = label == "ENTAILMENT"
                    details.append(SentenceDetail(
                        sentence=sentence, label=label,
                        confidence=confidence, supported=supported,
                    ))
                    if query_id:
                        bc = best_chunks[i]
                        attributions.append(SentenceAttribution(
                            query_id=query_id,
                            sentence_index=i,
                            sentence_text=sentence,
                            nli_label=label,
                            nli_confidence=confidence,
                            # BUG-FAITH-01 fix: now populated from retrieval result
                            source_chunk_id=bc.chunk.chunk_id if bc else "",
                            source_file=bc.chunk.source_file if bc else "",
                            source_page=bc.chunk.page_num if bc else 0,
                            supported=supported,
                        ))
            except Exception as exc:
                logger.warning("DeBERTa NLI failed: %s — falling back to heuristic", exc)
                details = []

        if not details:
            # Heuristic fallback — also uses per-chunk text when available
            if use_per_chunk:
                best_chunks = [
                    self._best_chunk_for_sentence(s, results) for s in sentences
                ]
            else:
                best_chunks = [None] * len(sentences)

            for i, sentence in enumerate(sentences):
                bc = best_chunks[i]
                premise = bc.chunk.text if bc else context
                label, confidence = self._heuristic(sentence, premise)
                supported = label == "ENTAILMENT"
                details.append(SentenceDetail(
                    sentence=sentence, label=label,
                    confidence=confidence, supported=supported,
                ))
                if query_id:
                    attributions.append(SentenceAttribution(
                        query_id=query_id,
                        sentence_index=i,
                        sentence_text=sentence,
                        nli_label=label,
                        nli_confidence=confidence,
                        # BUG-FAITH-01 fix: populated from retrieval result
                        source_chunk_id=bc.chunk.chunk_id if bc else "",
                        source_file=bc.chunk.source_file if bc else "",
                        source_page=bc.chunk.page_num if bc else 0,
                        supported=supported,
                    ))

        # GAP-15: write attributions to separate file
        if attributions and log_dir:
            self._write_attributions(attributions, log_dir)

        n_supported = sum(1 for d in details if d.supported)
        overall = n_supported / len(details) if details else 1.0
        unsupported = [d.sentence for d in details if not d.supported]

        return FaithfulnessReport(
            overall_score=overall,
            sentence_scores=details,
            unsupported=unsupported,
            is_fallback=not self._model_ok,
        )

    # RAG Triad — Context Relevance · Groundedness · Answer Relevance


    _RAG_TRIAD_THRESHOLD: float = 0.60   # leg score below this → weak_legs

    @staticmethod
    def _rag_stopwords() -> set:
        return {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "shall", "to",
            "of", "in", "for", "on", "with", "at", "by", "from", "as",
            "into", "through", "during", "before", "after", "above",
            "below", "between", "each", "or", "and", "but", "if", "then",
            "that", "this", "these", "those", "it", "its", "what", "how",
            "when", "where", "which", "who", "me", "my", "your", "their",
        }

    @classmethod
    def _rag_tokens(cls, text: str) -> set:
        """Stopword-filtered lowercase unigrams for RAG Triad scoring."""
        sw = cls._rag_stopwords()
        return {
            w for w in re.findall(r'\b[a-z][a-z0-9]{1,}\b', text.lower())
            if w not in sw
        }


    @classmethod
    def _score_context_relevance(
        cls,
        query: str,
        results: "List[RetrievalResult]",
    ) -> Tuple[float, "List[Tuple[str, float]]"]:
        """
        Context Relevance: are the retrieved chunks relevant to the query?

        Algorithm (NeMo/TruLens-aligned):
          For each chunk c_i with RRF rank r_i:
            relevance_i = |tokens(query) ∩ tokens(c_i.text)| / max(|tokens(query)|, 1)
          Weighted mean = Σ (relevance_i * weight_i) / Σ weight_i
          Weight = 1 / (1 + r_i)   (higher-ranked chunks contribute more)

        Returns (mean_weighted_relevance, [(chunk_id, score), ...])

        Score interpretation:
          ≥ 0.60  PASS  — retrieved context strongly covers the query terms
          0.40–0.59 BORDERLINE — partial coverage, may miss key concepts
          < 0.40  FAIL  — retrieval pulled largely irrelevant chunks
        """
        if not results:
            return 0.0, []

        q_tokens = cls._rag_tokens(query)
        if not q_tokens:
            return 1.0, []   # Empty query → vacuously relevant

        per_chunk: List[Tuple[str, float]] = []
        weighted_sum = 0.0
        weight_total = 0.0

        for r in results:
            c_tokens = cls._rag_tokens(r.chunk.text)
            if not c_tokens:
                score = 0.0
            else:
                # Recall-oriented: what fraction of query terms appear in chunk?
                score = len(q_tokens & c_tokens) / len(q_tokens)
            weight = 1.0 / (1.0 + r.final_rank)
            weighted_sum += score * weight
            weight_total += weight
            per_chunk.append((r.chunk.chunk_id, round(score, 4)))

        mean_score = weighted_sum / weight_total if weight_total > 0 else 0.0
        return round(min(mean_score, 1.0), 4), per_chunk


    @staticmethod
    def _score_groundedness(faith_report: FaithfulnessReport) -> float:
        """
        Groundedness: is every answer sentence grounded in retrieved chunks?

        Reuses the existing faithfulness evaluation result (DeBERTa NLI or
        heuristic).  The faithfulness overall_score is already the fraction of
        ENTAILMENT sentences — exactly the TruLens/NeMo groundedness definition.

        Returns faith_report.overall_score (0–1).
        """
        return round(faith_report.overall_score, 4)


    @classmethod
    def _score_answer_relevance(
        cls,
        query: str,
        answer: str,
    ) -> float:
        """
        Answer Relevance: does the answer address the question asked?

        Algorithm (NeMo/TruLens-aligned):
          base   = |tokens(query) ∩ tokens(answer)| / max(|tokens(query)|, 1)
          boost  = 1 + 0.15 * min(exact_phrase_hits, 3)
                   where exact_phrase_hits = number of query 3-grams found in answer
          score  = min(base * boost, 1.0)

        The boost rewards answers that re-use key multi-word query phrases
        (e.g., "Money Purchase Annual Allowance", "flexi-access drawdown")
        rather than just co-incidentally sharing single tokens.

        Score interpretation:
          ≥ 0.70  PASS  — answer squarely addresses the question
          0.50–0.69 BORDERLINE — answer partially addresses the question
          < 0.50  FAIL  — answer drifted from the question topic
        """
        if not answer.strip():
            return 0.0

        q_tokens = cls._rag_tokens(query)
        a_tokens = cls._rag_tokens(answer)

        if not q_tokens:
            return 1.0   # Empty query → vacuously relevant

        # Base: query-term recall in answer
        base = len(q_tokens & a_tokens) / len(q_tokens)

        # Boost: 3-gram phrase overlap
        def _ngrams(text: str, n: int = 3) -> set:
            words = re.findall(r'\b[a-z][a-z0-9]{1,}\b', text.lower())
            return {" ".join(words[i:i+n]) for i in range(len(words) - n + 1)}

        q_phrases = _ngrams(query, 3)
        a_text_lower = answer.lower()
        hits = sum(1 for ph in q_phrases if ph in a_text_lower)
        boost = 1.0 + 0.15 * min(hits, 3)

        return round(min(base * boost, 1.0), 4)


    def evaluate_rag_triad(
        self,
        query: str,
        answer: str,
        results: "List[RetrievalResult]",
        faith_report: FaithfulnessReport,
        query_id: str = "",
        log_dir: Optional[str] = None,
    ) -> RagTriadReport:
        """
        Compute all three RAG Triad legs for a single query-answer pair.

        Called immediately after answer generation and faithfulness evaluation.
        Writes a compact JSON record to {log_dir}/rag_triad.jsonl for monitoring.

        The harmonic mean is used for triad_score (penalises any weak leg):
          triad_score = 3 / (1/ctx_rel + 1/ground + 1/ans_rel)
          (with 0-division protection: zero leg → triad_score = 0)

        Parameters
        ----------
        query        : Original adviser query
        answer       : Final answer text (post doubt-fallback)
        results      : Retrieved chunks (List[RetrievalResult])
        faith_report : Already-computed faithfulness report (reuse groundedness)
        query_id     : For log correlation
        log_dir      : If provided, appends to rag_triad.jsonl

        Returns
        -------
        RagTriadReport with all scores, weak_legs, and per-chunk relevance.
        """

        ctx_rel, per_chunk = self._score_context_relevance(query, results)

        groundedness = self._score_groundedness(faith_report)

        ans_rel = self._score_answer_relevance(query, answer)

        legs = [ctx_rel, groundedness, ans_rel]
        if all(l > 0 for l in legs):
            triad_score = round(3.0 / sum(1.0 / l for l in legs), 4)
        else:
            triad_score = 0.0

        leg_names = ["context_relevance", "groundedness", "answer_relevance"]
        weak_legs = [
            name for name, score in zip(leg_names, legs)
            if score < self._RAG_TRIAD_THRESHOLD
        ]
        passed = triad_score >= self._RAG_TRIAD_THRESHOLD

        report = RagTriadReport(
            context_relevance=ctx_rel,
            groundedness=groundedness,
            answer_relevance=ans_rel,
            triad_score=triad_score,
            passed=passed,
            weak_legs=weak_legs,
            per_chunk_relevance=per_chunk,
            query_id=query_id,
        )

        logger.debug(
            "[%s] RAG Triad: ctx=%.2f ground=%.2f ans=%.2f → triad=%.2f [%s]",
            query_id, ctx_rel, groundedness, ans_rel, triad_score,
            "PASS" if passed else "REVIEW",
        )

        if log_dir:
            self._write_rag_triad(report, log_dir)

        return report

    @staticmethod
    def _write_rag_triad(report: RagTriadReport, log_dir: str) -> None:
        """Append RagTriadReport to {log_dir}/rag_triad.jsonl."""
        import json as _json
        path = Path(log_dir) / "rag_triad.jsonl"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(_json.dumps(report.to_dict()) + "\n")
        except Exception as exc:
            logger.error("Failed to write rag_triad.jsonl: %s", exc)

    # GAP-15: Separate sentence attribution log


    def _write_attributions(self, records: List[SentenceAttribution], log_dir: str) -> None:
        """
        GAP-15 fix: Write SentenceAttribution records to the sentence attribution log file.

        Filename is read from self.cfg.log_attribution when cfg is available,
        otherwise falls back to the default "sentence_attribution.jsonl".
        This file is SEPARATE from audit_log.jsonl.
        Linked via query_id for cross-reference.
        Compliance officers can query: SELECT * WHERE query_id = 'q_143102'
        """
        filename = (
            self.cfg.log_attribution if self.cfg is not None else "sentence_attribution.jsonl"
        )
        path = self.cfg.log_path(filename) if self.cfg is not None else Path(log_dir) / filename
        try:
            with open(path, "a", encoding="utf-8") as f:
                for rec in records:
                    f.write(json.dumps(asdict(rec)) + "\n")
        except Exception as exc:
            logger.error("Failed to write attribution log: %s", exc)
