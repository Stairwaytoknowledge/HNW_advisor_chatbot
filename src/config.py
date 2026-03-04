"""
src/config.py — Quilter HNW Advisor Assistant 
Single authoritative Config dataclass. All modules import from here.
Ollama local LLM stack: qwen2.5:14b / qwen2.5:7b / llama3.2:3b
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:

    project_root:  str = "C:/Code/quilter"
    pdf_dir:       str = "C:/Code/quilter/quilter_docs"
    index_dir:     str = "C:/Code/quilter/index_v3"
    log_dir:       str = "C:/Code/quilter/logs_v3"
    rails_dir:     str = "C:/Code/quilter/rails"
    eval_data_dir: str = "C:/Code/quilter/eval_data"
    version:       str = "3.0.0"

    chunk_size:    int = 400   # tokens (whitespace-split words)
    chunk_overlap: int = 80    # tokens of overlap between consecutive windows

    embed_model:    str = "BAAI/bge-large-en-v1.5"   # primary (1024-dim)
    embed_fallback: str = "all-MiniLM-L6-v2"          # fallback (384-dim)

    top_k_dense:  int   = 15
    top_k_sparse: int   = 15
    top_k_rerank: int   = 5
    rrf_k:        int   = 60     # RRF fusion constant
    mmr_lambda:   float = 0.7   # 1.0=pure relevance, 0.0=pure diversity
    use_hyde:     bool  = True  # Hypothetical Document Embedding query expansion
    rerank_model: str   = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # GAP-06 fix: all three tiers mapped to open-source Ollama models
    llm_provider:      str   = "ollama"
    ollama_base_url:   str   = "http://localhost:11434"
    llm_model_manager: str   = "qwen2.5:14b"   # Manager/Orchestrator
    llm_model_worker:  str   = "qwen2.5:7b"    # Precision / Compliance agents
    llm_model_fast:    str   = "llama3.2:3b"   # Retrieval / Fact-check / Single-agent
    temperature:       float = 0.0              # Deterministic outputs (compliance)
    max_context_tokens: int  = 4000
    llm_max_tokens:    int   = 1024             # Max output tokens per LLM call
    llm_retries:       int   = 2               # Exponential backoff retries

    nemo_enabled:                 bool  = True
    doc_freshness_threshold_days: int   = 90     # Flag docs older than this
    rrf_contact_centre_threshold: float = 0.015  # RRF score below → fallback
    nli_faithfulness_threshold:   float = 0.50   # NLI score below → review queue

    bertscore_model: str = "distilbert-base-uncased"
    recall_k:        int = 5

    log_audit:       str = "audit_log.jsonl"
    log_crew:        str = "crew_trace.jsonl"
    log_attribution: str = "sentence_attribution.jsonl"
    log_nemo:        str = "nemo_rail_log.jsonl"
    log_update:      str = "update_log.jsonl"
    log_compare:     str = "compare_log.jsonl"

    index_thresholds: str = "thresholds.json"

    eval_gold: str = "gold_eval_set.json"
    eval_oos:  str = "oos_eval_set.json"

    def ensure_dirs(self) -> None:
        """Create all required directories. Call once at startup."""
        for d in [
            self.pdf_dir,
            self.index_dir,
            self.log_dir,
            self.rails_dir,
            self.eval_data_dir,
        ]:
            Path(d).mkdir(parents=True, exist_ok=True)

    def log_path(self, filename: str) -> Path:
        """Convenience: resolve a filename inside log_dir."""
        return Path(self.log_dir) / filename

    def index_path(self, filename: str) -> Path:
        """Convenience: resolve a filename inside index_dir."""
        return Path(self.index_dir) / filename

    def eval_path(self, filename: str) -> Path:
        """Convenience: resolve a filename inside eval_data_dir."""
        return Path(self.eval_data_dir) / filename
