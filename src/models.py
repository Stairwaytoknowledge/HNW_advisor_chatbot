"""
src/models.py — Quilter HNW Advisor Assistant v3
All shared dataclasses. Imported by every other module.


"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# Document chunk


@dataclass
class Chunk:
    chunk_id:      str   # SHA256[:12] of text content
    doc_id:        str   # SHA256[:8] of source filename
    source_file:   str
    page_num:      int
    section:       str
    parent_context: str  # section heading + "\n\n" + text (fed to embedder)
    text:          str
    token_count:   int
    doc_version:   str = ""   # SHA256 of PDF file (from manifest) — RISK-01
    ingestion_ts:  str = field(default_factory=_now_iso)
    numerical_entities:             Dict[str, List[str]] = field(default_factory=dict)
    contains_regulatory_threshold:  bool                 = False
    regulatory_keywords:            List[str]            = field(default_factory=list)


# Retrieval result


@dataclass
class RetrievalResult:
    chunk:        Chunk
    dense_rank:   Optional[int]   = None
    sparse_rank:  Optional[int]   = None
    dense_score:  Optional[float] = None
    bm25_score:   Optional[float] = None
    rrf_score:    float           = 0.0
    rerank_score: Optional[float] = None
    final_rank:   int             = 0

    def score_breakdown(self) -> str:
        parts = []
        if self.dense_score is not None:
            parts.append(f"dense={self.dense_score:.4f}(rank {self.dense_rank})")
        if self.bm25_score is not None:
            parts.append(f"bm25={self.bm25_score:.4f}(rank {self.sparse_rank})")
        parts.append(f"rrf={self.rrf_score:.4f}")
        if self.rerank_score is not None:
            parts.append(f"rerank={self.rerank_score:.4f}")
        return " | ".join(parts)

    def dominant_signal(self) -> str:
        dr = self.dense_rank if self.dense_rank is not None else 9999
        sr = self.sparse_rank if self.sparse_rank is not None else 9999
        return "dense" if dr <= sr else "sparse"


# Precision engine output


@dataclass
class PrecisionResult:
    query_type:       str
    computed_value:   Optional[str]
    working_shown:    List[str]
    source_citations: List[str]
    confidence:       float
    warnings:         List[str] = field(default_factory=list)
    raw_values:       Dict[str, Any] = field(default_factory=dict)

    def format_for_answer(self) -> str:
        lines = []
        if self.computed_value:
            lines.append(f"**Result:** {self.computed_value}")
        if self.working_shown:
            lines.append("\n**Working:**")
            lines.extend(f"  {w}" for w in self.working_shown)
        if self.source_citations:
            lines.append("\n**Sources:**")
            lines.extend(f"  {c}" for c in self.source_citations)
        if self.warnings:
            lines.append("\n**Warnings:**")
            lines.extend(f"  WARNING: {w}" for w in self.warnings)
        return "\n".join(lines)


# NeMo guardrail record


@dataclass
class RailActivation:
    timestamp:    str
    query_id:     str
    rail_type:    str   # "input" | "retrieval" | "output"
    rail_name:    str
    trigger:      str
    action_taken: str
    was_blocked:  bool
    metadata:     Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)


# Faithfulness / NLI


@dataclass
class SentenceDetail:
    sentence:   str
    label:      str    # ENTAILMENT | NEUTRAL | CONTRADICTION
    confidence: float
    supported:  bool


@dataclass
class FaithfulnessReport:
    overall_score:   float          # fraction of ENTAILMENT sentences
    sentence_scores: List[SentenceDetail]
    unsupported:     List[str]      # sentences labelled NEUTRAL or CONTRADICTION
    is_fallback:     bool           # True if heuristic used instead of model

    def summary(self) -> str:
        total = len(self.sentence_scores)
        supported = sum(1 for s in self.sentence_scores if s.supported)
        label = "PASS" if self.overall_score >= 0.50 else "REVIEW"
        return (f"Faithfulness: {self.overall_score:.2f} "
                f"({supported}/{total} sentences supported) [{label}]")

    def render(self, full: bool = True) -> None:
        print(self.summary())
        if full:
            for i, s in enumerate(self.sentence_scores, 1):
                icon = "OK" if s.supported else "FAIL"
                print(f"  [{icon}] {s.label:14s} {s.confidence:.2f}  {s.sentence[:80]}")
        if self.unsupported:
            print(f"\n  Unsupported sentences ({len(self.unsupported)}):")
            for u in self.unsupported:
                print(f"    - {u[:100]}")


# RAG Triad — Context Relevance · Groundedness · Answer Relevance


@dataclass
class RagTriadReport:
    """
    NeMo / TruLens-style RAG Triad evaluation for every answered query.

    Three scores, each in [0, 1]:

    1. context_relevance   — Are the retrieved chunks actually relevant to the query?
                             Computed as mean token-overlap (query ∩ chunk) / |query|
                             per chunk, weighted by RRF rank.  Catches retrieval failures
                             where the system pulled unrelated documents.

    2. groundedness        — Is every sentence in the answer supported by at least one
                             retrieved chunk?  This is the existing faithfulness score
                             (DeBERTa NLI or heuristic), re-exposed as a triad leg.
                             Directly addresses the 18% post-fix hallucination risk.

    3. answer_relevance    — Does the answer actually address the query?
                             Computed as token-overlap (query ∩ answer) / |query|,
                             boosted by presence of query key-terms in answer.
                             Catches model drift where the answer is grounded in the
                             docs but does not answer the question asked.

    triad_score            — Harmonic mean of the three legs (penalises any weak leg).
    passed                 — True if triad_score >= 0.60 (production threshold).
    weak_legs              — List of leg names scoring < 0.60 (for targeted remediation).
    per_chunk_relevance    — List of (chunk_id, score) for traceability.
    """
    context_relevance:    float
    groundedness:         float
    answer_relevance:     float
    triad_score:          float
    passed:               bool
    weak_legs:            List[str]
    per_chunk_relevance:  List[Tuple[str, float]]  # [(chunk_id, score), ...]
    query_id:             str = ""
    timestamp:            str = field(default_factory=_now_iso)

    def summary(self) -> str:
        status = "PASS" if self.passed else "REVIEW"
        return (
            f"RAG Triad [{status}]  "
            f"CtxRel={self.context_relevance:.2f}  "
            f"Ground={self.groundedness:.2f}  "
            f"AnsRel={self.answer_relevance:.2f}  "
            f"Triad={self.triad_score:.2f}"
            + (f"  WeakLegs={self.weak_legs}" if self.weak_legs else "")
        )

    def to_dict(self) -> Dict:
        return asdict(self)


# Sentence attribution — GAP-15 (written to separate sentence_attribution.jsonl)


@dataclass
class SentenceAttribution:
    query_id:        str
    sentence_index:  int
    sentence_text:   str
    nli_label:       str    # ENTAILMENT | NEUTRAL | CONTRADICTION
    nli_confidence:  float
    source_chunk_id: str    # Chunk.chunk_id of best-matching chunk (may be empty)
    source_file:     str
    source_page:     int
    supported:       bool
    timestamp:       str = field(default_factory=_now_iso)

    def to_dict(self) -> Dict:
        return asdict(self)


# Agent trace — L2 explainability


@dataclass
class AgentOutput:
    agent_name:  str
    task_name:   str
    output:      str
    tool_calls:  List[Dict] = field(default_factory=list)
    latency_ms:  float      = 0.0
    reasoning:   str        = ""

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class CrewTrace:
    query_id:     str
    query:        str
    agent_steps:  List[AgentOutput] = field(default_factory=list)
    final_answer: str   = ""
    total_ms:     float = 0.0

    def to_dict(self) -> Dict:
        return {
            "query_id":    self.query_id,
            "query":       self.query,
            "total_ms":    self.total_ms,
            "agent_steps": [s.to_dict() for s in self.agent_steps],
            "final_answer_preview": self.final_answer[:200],
        }


# Final answer — complete answer with full audit chain


@dataclass
class FinalAnswer:
    query_id:         str
    query:            str
    answer:           str
    route_used:       str
    citations:        List[str]
    precision:        Optional[PrecisionResult]
    faithfulness:     FaithfulnessReport
    crew_trace:       Optional[CrewTrace]
    nemo_activations: List[RailActivation]
    token_importance: List[Tuple[str, float]]
    max_rrf_score:    float
    latency_ms:       float
    review_needed:    bool
    warnings:         List[str]
    doc_versions:     Dict[str, str] = field(default_factory=dict)   # GAP-16
    rag_triad:        Optional["RagTriadReport"] = None               # RAG-TRIAD
    timestamp:        str = field(default_factory=_now_iso)

    def audit_dict(self) -> Dict:
        """
        Scalar audit record for audit_log.jsonl.
        GAP-16: includes doc_versions {source_file: sha256}.
        RAG-TRIAD: includes triad scores inline for dashboard/monitoring.
        Does NOT include full crew_trace or sentence_attribution
        (those have separate files — GAP-15).
        """
        triad_dict: Dict = {}
        if self.rag_triad is not None:
            triad_dict = {
                "rag_context_relevance": round(self.rag_triad.context_relevance, 4),
                "rag_groundedness":      round(self.rag_triad.groundedness, 4),
                "rag_answer_relevance":  round(self.rag_triad.answer_relevance, 4),
                "rag_triad_score":       round(self.rag_triad.triad_score, 4),
                "rag_passed":            self.rag_triad.passed,
                "rag_weak_legs":         self.rag_triad.weak_legs,
            }
        return {
            "ts":               self.timestamp,
            "qid":              self.query_id,
            "query":            self.query[:200],
            "route":            self.route_used,
            "max_rrf":          round(self.max_rrf_score, 4),
            "faithfulness":     round(self.faithfulness.overall_score, 4),
            "review_needed":    self.review_needed,
            "nemo_rails_fired": len(self.nemo_activations),
            "nemo_blocked":     any(a.was_blocked for a in self.nemo_activations),
            "agent_steps":      len(self.crew_trace.agent_steps) if self.crew_trace else 1,
            "sources":          list({
                c.split(",")[0].replace("[Source: ", "").strip()
                for c in self.citations
            }),
            "latency_ms":       round(self.latency_ms, 1),
            "warnings":         self.warnings,
            "doc_versions":     self.doc_versions,   # GAP-16
            **triad_dict,                             # RAG-TRIAD (inline)
        }
