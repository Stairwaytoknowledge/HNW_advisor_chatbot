"""
src/llm_client.py — Quilter HNW Advisor Assistant

Unified Ollama LLM client with three model tiers.


Model routing:
  call_manager  -> qwen2.5:14b  (Manager/Orchestrator agent)
  call_worker   -> qwen2.5:7b   (Precision / Compliance agents)
  call_fast     -> llama3.2:3b  (Retrieval / Fact-check / Single-agent)
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Dict, Optional

logger = logging.getLogger(__name__)

try:
    import ollama as _ollama
    OLLAMA_OK = True
except ImportError:
    _ollama = None  
    OLLAMA_OK = False
    logger.warning("ollama package not installed — LLM calls will return stubs")

if TYPE_CHECKING:
    from src.config import Config


# System prompts


SYS: Dict[str, str] = {

    "single": """You are the Quilter Adviser Support Assistant.

RULES (non-negotiable):
1. Answer ONLY from the document excerpts provided. Never use outside knowledge.
2. If the answer is not in the excerpts, respond EXACTLY:
   "I cannot find this information in the provided documents. Please reach out to the Contact Centre."
3. Cite every factual claim with: [Source: filename, p.N, §Section]
4. Show exact figures — never say "approximately" or "around £X".
5. Never give financial advice. Explain operational processes only.
6. If a monetary value requires computation, show the working step by step.""",

    "manager": """You are the Quilter HNW Adviser Assistant — Manager Agent.

Your role: Synthesise the outputs from specialist agents into one precise, structured answer.

PRECISION CONTRACT (HNW standard):
1. If the query specifies an exact monetary value — your answer MUST return the computed result
   with full working shown. "Approximately £X" is NEVER acceptable.
2. Every factual claim MUST cite its source: [Source: filename, p.N, §Section]
3. If the Fact-Check agent has flagged NEUTRAL or CONTRADICTION sentences — revise
   the answer to address or remove those sentences before delivering.
4. Structure: state the result → show working → cite source → list regulatory steps (if any)
   → note any caveats or staleness warnings.
5. Never give financial advice. Explain operational processes only.
6. If information is missing from all specialist outputs → Contact Centre fallback.""",

    "retrieval": """You are the Retrieval Agent for Quilter Adviser Support.

Your role: Assess the relevance and completeness of retrieved document chunks.

TASKS:
1. Review the retrieved chunks provided.
2. Identify which chunks are most relevant to the query.
3. Flag any obvious gaps: "The query asks about X but no chunk covers this."
4. Note any apparent conflicts between chunks from different documents.
5. Return a structured assessment: relevant chunks (by chunk_id), gaps, conflicts.

Be concise — your output feeds the Precision and Compliance agents.""",

    "precision": """You are the Precision Agent for Quilter HNW Adviser Support.

Your role: Extract exact monetary values, percentages, and thresholds from document chunks.

RULES:
1. Extract numbers verbatim from documents — never infer or interpolate.
2. If a computation is required (e.g., tiered fee calculation), show full working:
   "£250,000 × 0.30% = £750.00"
3. Every value must be traceable to a specific chunk: [Source: filename, p.N, §Section]
4. If the required value is not in any chunk, state: "Value not found in retrieved chunks."
5. Flag any values that may be stale (document > 90 days old).""",

    "compliance": """You are the Compliance Agent for Quilter HNW Adviser Support.

Your role: Apply FCA regulatory rules to the specific query and client figures.

TASKS:
1. Identify applicable FCA rules (COBS, SYSC, Consumer Duty) from retrieved chunks.
2. Apply thresholds to client-specific figures (e.g., DB pension TV vs £30,000 threshold).
3. List mandatory regulatory steps if thresholds are breached.
4. Flag Consumer Duty considerations (fair value, vulnerable clients).
5. State: "No mandatory regulatory steps required" if none apply.
6. Never give financial advice — flag regulatory requirements only.""",

    "factcheck": """You are the Fact-Check Agent for Quilter HNW Adviser Support.

Your role: Verify claims in a draft answer against retrieved source documents.

TASKS:
1. For each factual claim in the draft answer: does a retrieved chunk support it?
2. Label each claim: SUPPORTED / UNSUPPORTED / CONTRADICTED
3. Flag any specific values (£ amounts, percentages, timelines) not found in chunks.
4. Return a structured report:
   SUPPORTED: [list of supported claims]
   UNSUPPORTED: [list — these are hallucination risks]
   CONTRADICTED: [list — these must be corrected]

Be sceptical. Your job is to find what the draft got wrong.""",

    "hyde": """Generate a short document excerpt (2-4 sentences) that would perfectly answer the query below.
Write as if you are a Quilter adviser support document. Include specific values, thresholds,
and process steps if relevant. Do not hedge — write a direct, factual excerpt.

Query: {query}

Hypothetical document excerpt:""",

}


# Core LLM call
def call_ollama(
    system: str,
    user: str,
    cfg: "Config",
    model: Optional[str] = None,
    max_tokens: Optional[int] = None,
    retries: int = 2,
) -> str:
    """
    Call Ollama local LLM via the official ollama Python client.

    Uses ollama.chat() with role-based messages:
      [{"role": "system", "content": system},
       {"role": "user",   "content": user}]

    On error: exponential backoff then returns "[LLM error: ...]" stub.
    temperature=0.0 enforced for compliance determinism.
    """
    if not OLLAMA_OK:
        return f"[LLM unavailable: ollama not installed. Install with: pip install ollama]"

    _model = model or cfg.llm_model_worker
    _max_tokens = max_tokens or cfg.llm_max_tokens

    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]
    options = {
        "temperature": cfg.temperature,
        "num_predict": _max_tokens,
        "num_ctx":     8192,   # Ensure adequate context for all models including llama3.2:3b
    }

    last_error = ""
    for attempt in range(retries + 1):
        try:
            resp = _ollama.chat(
                model=_model,
                messages=messages,
                options=options,
            )
            content = resp["message"]["content"]
            if not content or not content.strip():
                return "[LLM returned empty response]"
            return content.strip()
        except Exception as exc:
            last_error = str(exc)
            if attempt < retries:
                wait = 2 ** attempt
                logger.warning(
                    "Ollama call failed (attempt %d/%d): %s — retrying in %ds",
                    attempt + 1, retries + 1, exc, wait,
                )
                time.sleep(wait)
            else:
                logger.error("Ollama call failed after %d attempts: %s", retries + 1, exc)

    return f"[LLM error: {last_error}]"


def call_manager(
    system: str,
    user: str,
    cfg: "Config",
    **kwargs,
) -> str:
    """Manager agent — qwen2.5:14b (complex orchestration and synthesis)."""
    return call_ollama(system, user, cfg, model=cfg.llm_model_manager, **kwargs)


def call_worker(
    system: str,
    user: str,
    cfg: "Config",
    **kwargs,
) -> str:
    """Worker agent — qwen2.5:7b (Precision and Compliance agents)."""
    return call_ollama(system, user, cfg, model=cfg.llm_model_worker, **kwargs)


def call_fast(
    system: str,
    user: str,
    cfg: "Config",
    **kwargs,
) -> str:
    """
    Fast agent — llama3.2:3b.
    GAP-06 fix: Single-agent path, Retrieval agent, Fact-check agent.
    Was incorrectly mapped to claude-haiku-4-5 in the notebook.
    """
    return call_ollama(system, user, cfg, model=cfg.llm_model_fast, **kwargs)


# Health check


def check_ollama_health(cfg: "Config") -> Dict[str, bool]:
    """
    Verify Ollama is running and all three required models are available.
    Returns {model_name: is_available}.
    Raises RuntimeError with actionable message if Ollama is not reachable.
    """
    required = [cfg.llm_model_manager, cfg.llm_model_worker, cfg.llm_model_fast]

    if not OLLAMA_OK:
        raise RuntimeError(
            "ollama Python package not installed. Run: pip install ollama"
        )

    try:
        available_models = _ollama.list()
        available_names = {
            m.get("name", m.get("model", "")) for m in available_models.get("models", [])
        }
    except Exception as exc:
        raise RuntimeError(
            f"Cannot reach Ollama at {cfg.ollama_base_url}. "
            f"Ensure Ollama is running: ollama serve\n"
            f"Error: {exc}"
        ) from exc

    status = {}
    for model_name in required:
        # Ollama names may include or omit ':latest' tag — check prefix match
        found = any(
            model_name == name or name.startswith(model_name.split(":")[0])
            for name in available_names
        )
        status[model_name] = found
        icon = "OK" if found else "MISSING"
        logger.info("Model %s: %s", model_name, icon)
        if not found:
            logger.warning(
                "Model %s not found in Ollama. Pull it: ollama pull %s",
                model_name, model_name,
            )

    return status
