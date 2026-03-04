"""
src/agents.py — Quilter HNW Advisor Assistant 

Five specialist agent runner functions + real CrewAI tool classes.

"""

from __future__ import annotations

import time
import logging
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from src.models import AgentOutput, FaithfulnessReport
from src.llm_client import SYS, call_fast, call_manager, call_worker

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from src.config import Config
    from src.faithfulness import FaithfulnessEvaluator
    from src.models import PrecisionResult, RetrievalResult
    from src.precision_engine import HNWPrecisionEngine
    from src.retrieval import HybridIndex

# Real CrewAI availability
try:
    from crewai import Agent, Task, Crew, Process
    from crewai.tools import BaseTool
    CREWAI_OK = True
except ImportError:
    CREWAI_OK = False
    logger.info("crewai not installed — agent runner functions active (no orchestration framework)")


# Context formatting helper


def _format_context(results: "List[RetrievalResult]", max_tokens: int = 3000) -> str:
    """Format retrieved chunks into a context string with [SOURCE:] headers."""
    parts = []
    total_tokens = 0
    for r in results:
        c = r.chunk
        header = f"[SOURCE: {c.source_file}, p.{c.page_num}, §{c.section}]"
        entry = f"{header}\n{c.text}\n(rrf={r.rrf_score:.4f})"
        entry_tokens = len(entry.split())
        if total_tokens + entry_tokens > max_tokens:
            break
        parts.append(entry)
        total_tokens += entry_tokens
    return "\n\n---\n\n".join(parts)


# Agent 1: Retrieval Agent


def run_retrieval_agent(
    query: str,
    index: "HybridIndex",
    cfg: "Config",
) -> AgentOutput:
    """
    Agent 1 — Retrieval Agent.
    GAP-06 fix: uses call_fast (llama3.2:3b) — was incorrectly using worker model.

    Performs hybrid retrieval, then asks the LLM to assess relevance and gaps.
    Returns structured assessment of retrieved chunks.
    """
    t0 = time.perf_counter()

    # Hybrid search (includes HyDE + MMR)
    results = index.search(query, cfg=cfg)
    context = _format_context(results, max_tokens=2000)

    tool_calls = [{
        "tool": "hybrid_search",
        "input": query,
        "output": f"{len(results)} chunks retrieved, max_rrf={max((r.rrf_score for r in results), default=0):.4f}",
    }]

    # LLM assessment of relevance (fast model — llama3.2:3b)
    user_prompt = (
        f"Query: {query}\n\n"
        f"Retrieved chunks:\n{context}\n\n"
        f"Assess: Which chunks are most relevant? Are there any gaps? "
        f"Any conflicts between sources? Be concise."
    )
    assessment = call_fast(SYS["retrieval"], user_prompt, cfg, max_tokens=400)

    latency_ms = (time.perf_counter() - t0) * 1000
    return AgentOutput(
        agent_name="Retrieval Agent",
        task_name="Hybrid Search + Relevance Assessment",
        output=f"Retrieved {len(results)} chunks.\n{assessment}",
        tool_calls=tool_calls,
        latency_ms=latency_ms,
        reasoning=f"Hybrid FAISS+BM25+RRF search with HyDE expansion. max_rrf={max((r.rrf_score for r in results), default=0):.4f}",
    ), results


# Agent 2: Precision Agent


def run_precision_agent(
    query: str,
    retrieved: str,
    ptype: str,
    params: Dict,
    results: "List[RetrievalResult]",
    precision_engine: "HNWPrecisionEngine",
    cfg: "Config",
) -> Tuple[AgentOutput, Optional["PrecisionResult"]]:
    """
    Agent 2 — Precision Agent.
    Uses call_worker (qwen2.5:7b) for LLM-based extraction.

    GAP-07 fix: 'chaps_fee' ptype → precision_engine.compute_chaps_fee()
    GAP-08 fix: 'carry_forward_mpaa' ptype → precision_engine.check_mpaa(..., prior_year_unused)
    GAP-09 fix: 'ufpls_tax' ptype → precision_engine.compute_ufpls_tax()
    """
    t0 = time.perf_counter()
    precision_result = None
    tool_calls = []

    if ptype == "fee_calculation" and "aum" in params:
        aum = params["aum"]
        precision_result = precision_engine.compute_platform_fee(aum, results)
        tool_calls.append({
            "tool": "compute_platform_fee",
            "input": f"aum={aum}",
            "output": precision_result.computed_value,
        })

    elif ptype == "threshold_check_db" and "tv" in params:
        tv = params["tv"]
        precision_result = precision_engine.check_db_threshold(tv, results)
        tool_calls.append({
            "tool": "check_regulatory_threshold",
            "input": f"type=db_transfer, value={tv}",
            "output": precision_result.computed_value,
        })

    elif ptype in ("threshold_check_mpaa", "carry_forward_mpaa") and "proposed" in params:
        has_flex  = params.get("has_flex", False)
        proposed  = params["proposed"]
        unused    = params.get("unused", [])
        precision_result = precision_engine.check_mpaa(has_flex, proposed, unused or None)
        tool_calls.append({
            "tool": "check_regulatory_threshold",
            "input": f"type=mpaa, has_flex={has_flex}, proposed={proposed}, unused={unused}",
            "output": precision_result.computed_value,
        })

    elif ptype == "chaps_fee" and "amount" in params:
        # GAP-07: CHAPS fee computation
        amount = params["amount"]
        precision_result = precision_engine.compute_chaps_fee(amount, results)
        tool_calls.append({
            "tool": "compute_chaps_fee",
            "input": f"withdrawal_amount={amount}",
            "output": precision_result.computed_value,
        })

    elif ptype == "ufpls_tax" and "amount" in params:
        # GAP-09: UFPLS tax computation
        amount  = params["amount"]
        rate    = params.get("marginal_rate", 0.20)
        precision_result = precision_engine.compute_ufpls_tax(amount, rate, results)
        tool_calls.append({
            "tool": "compute_ufpls_tax",
            "input": f"amount={amount}, marginal_rate={rate}",
            "output": precision_result.computed_value,
        })

    prec_context = precision_result.format_for_answer() if precision_result else ""

    user_prompt = (
        f"Query: {query}\n\n"
        f"Retrieved context:\n{retrieved}\n\n"
        f"{'Precision computation result:\n' + prec_context + chr(10) + chr(10) if prec_context else ''}"
        f"Extract ALL exact monetary values, percentages, timelines, and thresholds relevant to this query. "
        f"Trace every value to its source chunk. Show any computation working."
    )
    llm_output = call_worker(SYS["precision"], user_prompt, cfg, max_tokens=600)

    latency_ms = (time.perf_counter() - t0) * 1000
    output = prec_context + "\n\n" + llm_output if prec_context else llm_output

    return AgentOutput(
        agent_name="Precision Agent",
        task_name="Exact Value Extraction + Computation",
        output=output,
        tool_calls=tool_calls,
        latency_ms=latency_ms,
        reasoning=f"Precision type: {ptype}. Engine used: {bool(precision_result)}",
    ), precision_result


# Agent 3: Compliance Agent


def run_compliance_agent(
    query: str,
    retrieved: str,
    precision_out: str,
    cfg: "Config",
) -> AgentOutput:
    """
    Agent 3 — Compliance Agent.
    Uses call_worker (qwen2.5:7b).

    Identifies applicable FCA rules, applies thresholds to client figures,
    lists mandatory regulatory steps, flags Consumer Duty considerations.
    """
    t0 = time.perf_counter()

    user_prompt = (
        f"Query: {query}\n\n"
        f"Retrieved regulatory context:\n{retrieved}\n\n"
        f"Precision findings:\n{precision_out}\n\n"
        f"Identify: applicable FCA rules (COBS/SYSC/Consumer Duty), mandatory advice requirements, "
        f"regulatory process steps, and Consumer Duty considerations. "
        f"State 'No mandatory regulatory steps required' if none apply."
    )
    output = call_worker(SYS["compliance"], user_prompt, cfg, max_tokens=500)

    latency_ms = (time.perf_counter() - t0) * 1000
    return AgentOutput(
        agent_name="Compliance Agent",
        task_name="FCA Regulatory Validation",
        output=output,
        tool_calls=[{"tool": "fca_lookup", "input": query, "output": "regulatory_assessment"}],
        latency_ms=latency_ms,
        reasoning="Applied FCA COBS/SYSC/Consumer Duty rules to retrieved context",
    )


# Agent 4: Fact-Check Agent (MOVED BEFORE MANAGER — GAP-03 fix)


def run_factcheck_agent(
    query: str,
    draft_answer: str,
    context: str,
    faith_eval: "FaithfulnessEvaluator",
    cfg: "Config",
    query_id: str = "",
    log_dir: str = "",
    results: "Optional[List[RetrievalResult]]" = None,
) -> Tuple[AgentOutput, FaithfulnessReport]:
    """
    Agent 4 — Fact-Check Agent.

    GAP-03 fix: Now runs on a DRAFT answer (from precision + compliance)
                BEFORE the Manager synthesises the final answer.
                Manager receives fact-check output and can revise accordingly.

    GAP-06 fix: Uses call_fast (llama3.2:3b).
    GAP-15 fix: Passes query_id and log_dir to faith_eval.evaluate() so that
                sentence attributions are written to sentence_attribution.jsonl.
    BUG-FAITH-01 fix: Passes `results` to faith_eval.evaluate() so each sentence
                is grounded to the actual chunk text — source_chunk_id, source_file,
                and source_page are now correctly populated in the attribution log.
    """
    t0 = time.perf_counter()

    # BUG-FAITH-01 fix: pass retrieval results for per-sentence chunk binding
    faith_report = faith_eval.evaluate(
        answer=draft_answer,
        context=context,
        query_id=query_id,
        log_dir=log_dir,
        results=results,
    )

    # LLM sceptical review
    user_prompt = (
        f"Query: {query}\n\n"
        f"Draft answer to fact-check:\n{draft_answer}\n\n"
        f"Source documents:\n{context}\n\n"
        f"NLI faithfulness report:\n{faith_report.summary()}\n"
        f"Unsupported sentences: {faith_report.unsupported}\n\n"
        f"For each factual claim: is it SUPPORTED, UNSUPPORTED, or CONTRADICTED by the sources? "
        f"Flag any specific £ amounts or percentages not found in sources."
    )
    llm_output = call_fast(SYS["factcheck"], user_prompt, cfg, max_tokens=500)

    latency_ms = (time.perf_counter() - t0) * 1000
    output = (
        f"NLI Faithfulness: {faith_report.summary()}\n\n"
        f"Fact-Check Review:\n{llm_output}"
    )

    return AgentOutput(
        agent_name="Fact-Check Agent",
        task_name="NLI Faithfulness + Sceptical Review",
        output=output,
        tool_calls=[{
            "tool": "nli_verify",
            "input": f"{len(faith_report.sentence_scores)} sentences",
            "output": faith_report.summary(),
        }],
        latency_ms=latency_ms,
        reasoning=f"DeBERTa NLI: {faith_report.overall_score:.2f} faithfulness. "
                  f"Unsupported: {len(faith_report.unsupported)}",
    ), faith_report


# Agent 5: Manager Agent (synthesises AFTER fact-check — GAP-03 fix)


def run_manager_agent(
    query: str,
    retrieval_out: str,
    precision_out: str,
    compliance_out: str,
    factcheck_out: str,
    context: str,
    cfg: "Config",
) -> AgentOutput:
    """
    Agent 5 — Manager Agent.

    GAP-03 fix: Manager now receives factcheck_out BEFORE synthesis.
    The prompt explicitly instructs the Manager to address NEUTRAL/CONTRADICTION
    sentences flagged by the Fact-Check Agent before delivering the final answer.

    Uses call_manager (qwen2.5:14b — most capable model).
    """
    t0 = time.perf_counter()

    user_prompt = (
        f"Query: {query}\n\n"
        f"=== SPECIALIST AGENT OUTPUTS ===\n\n"
        f"[RETRIEVAL AGENT]\n{retrieval_out}\n\n"
        f"[PRECISION AGENT]\n{precision_out}\n\n"
        f"[COMPLIANCE AGENT]\n{compliance_out}\n\n"
        f"[FACT-CHECK AGENT — Address issues BEFORE final answer]\n{factcheck_out}\n\n"
        f"=== SOURCE DOCUMENTS ===\n{context}\n\n"
        f"=== SYNTHESIS INSTRUCTIONS ===\n"
        f"1. Synthesise a single precise answer from the specialist outputs above.\n"
        f"2. For any UNSUPPORTED or CONTRADICTED claims flagged by the Fact-Check Agent: "
        f"REMOVE or CORRECT them in the final answer.\n"
        f"3. Show all monetary computations with full working.\n"
        f"4. Cite every factual claim: [Source: filename, p.N, §Section]\n"
        f"5. List any mandatory regulatory steps required.\n"
        f"6. Include any staleness warnings from the retrieval context.\n"
        f"7. If information is missing: 'Please reach out to the Contact Centre.'\n"
    )
    final = call_manager(SYS["manager"], user_prompt, cfg, max_tokens=1200)

    latency_ms = (time.perf_counter() - t0) * 1000
    return AgentOutput(
        agent_name="Manager Agent",
        task_name="Multi-Domain Synthesis (post fact-check)",
        output=final,
        tool_calls=[],
        latency_ms=latency_ms,
        reasoning=(
            "Synthesised from retrieval + precision + compliance outputs. "
            "Fact-check issues addressed before delivery (GAP-03 fix)."
        ),
    )


# Real CrewAI tool classes (when crewai is installed)


if CREWAI_OK:
    class QuilterSearchTool(BaseTool):
        name: str = "quilter_document_search"
        description: str = (
            "Search Quilter adviser support documents using hybrid retrieval. "
            "Input: natural language query. "
            "Output: top 5 relevant document chunks with source citations."
        )

        def __init__(self, index: "HybridIndex", cfg: "Config"):
            super().__init__()
            self._index = index
            self._cfg   = cfg

        def _run(self, query: str) -> str:
            results = self._index.search(query, cfg=self._cfg)
            return _format_context(results, max_tokens=2000)


    class ComputeFeeTool(BaseTool):
        name: str = "compute_platform_fee"
        description: str = (
            "Compute EXACT Quilter tiered platform fee for a given AUM (£). "
            "Input: AUM as a number (e.g., 1850000 for £1,850,000). "
            "Output: annual fee with full tier-by-tier working shown. NEVER approximates."
        )

        def __init__(self, precision_engine: "HNWPrecisionEngine"):
            super().__init__()
            self._pe = precision_engine

        def _run(self, aum_pounds: float) -> str:
            result = self._pe.compute_platform_fee(float(aum_pounds), [])
            return result.format_for_answer()


    class CheckThresholdTool(BaseTool):
        name: str = "check_regulatory_threshold"
        description: str = (
            "Check if a value crosses a regulatory threshold. "
            "threshold_type options: 'db_transfer', 'mpaa', 'chaps', 'ufpls'. "
            "Input format: '<threshold_type>|<value>|[optional: marginal_rate or unused_list]'. "
            "Output: threshold check result with regulatory steps if applicable."
        )

        def __init__(self, precision_engine: "HNWPrecisionEngine"):
            super().__init__()
            self._pe = precision_engine

        def _run(self, input_str: str) -> str:
            parts = input_str.split("|")
            if len(parts) < 2:
                return "Error: expected format 'threshold_type|value'"
            ttype = parts[0].strip()
            value = float(parts[1].strip().replace(",", "").replace("£", ""))

            if ttype == "db_transfer":
                r = self._pe.check_db_threshold(value, [])
            elif ttype == "mpaa":
                has_flex = len(parts) > 2 and parts[2].strip().lower() == "true"
                r = self._pe.check_mpaa(has_flex, value)
            elif ttype == "chaps":
                r = self._pe.compute_chaps_fee(value, [])
            elif ttype == "ufpls":
                rate = float(parts[2]) if len(parts) > 2 else 0.20
                r = self._pe.compute_ufpls_tax(value, rate)
            else:
                return f"Unknown threshold_type: {ttype}"
            return r.format_for_answer()


    class NLIVerifyTool(BaseTool):
        name: str = "nli_verify"
        description: str = (
            "Verify if a sentence is supported by the provided context using NLI. "
            "Input format: 'sentence|||context'. "
            "Output: ENTAILMENT, NEUTRAL, or CONTRADICTION with confidence score."
        )

        def __init__(self, faith_eval: "FaithfulnessEvaluator"):
            super().__init__()
            self._fe = faith_eval

        def _run(self, input_str: str) -> str:
            if "|||" not in input_str:
                return "Error: expected format 'sentence|||context'"
            sent, ctx = input_str.split("|||", 1)
            label, conf = self._fe._heuristic(sent.strip(), ctx.strip())
            return f"{label} (confidence: {conf:.2f})"
