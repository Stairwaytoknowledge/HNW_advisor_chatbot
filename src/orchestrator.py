"""
src/orchestrator.py — Quilter HNW Advisor Assistant 

QuilterAdvisorSystem — main pipeline with  agent order.

pipeline order:
  retrieval -> precision ->compliance ->fact-check(draft) -> manager(+fc) -> output
  Manager now synthesises AFTER fact-check, not before.

crew_trace.jsonl and sentence_attribution.jsonl written as separate files.
doc_versions (chunk SHA256) recorded per-query in audit_log.jsonl.
 _audit_log_override parameter isolates compare_configs() writes.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from src.models import CrewTrace, FinalAnswer

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from src.config import Config
    from src.embedding import EmbeddingEngine
    from src.faithfulness import FaithfulnessEvaluator
    from src.guardrails import NeMoEngine
    from src.models import Chunk, RetrievalResult
    from src.precision_engine import HNWPrecisionEngine
    from src.retrieval import HybridIndex


# Contact Centre fallback answer
_CONTACT_CENTRE = (
    "I cannot find sufficient information in the approved documents to answer this query. "
    "Please reach out to the Contact Centre for assistance."
)

# Single-agent system prompt key
_SINGLE_SYS_KEY = "single"


class QuilterAdvisorSystem:
    """
    Main orchestrator for the Quilter HNW Advisor Assistant.

    Implements the 7-layer defence stack:
      L1  NeMo input rail
      L2  Retrieval confidence gate (RRF)
      L3  HNW Precision Engine
      L4  Constitutional prompt + temperature=0
      L5  NeMo output rail
      L6  Fact-Check Agent (pre-manager, GAP-03)
      L7  Human review queue flag

    Routes:
      "blocked"         → Contact Centre (injection / OOS)
      "fallback"        → Contact Centre (low RRF / OOS)
      "single_agent"    → Fast single LLM call (llama3.2:3b)
      "crewai_standard" → Retrieval + Precision + Fact-Check + Manager
      "crewai_hnw"      → All 5 agents (full multi-domain)
    """

    def __init__(
        self,
        cfg: "Config",
        index: "HybridIndex",
        nemo: "NeMoEngine",
        faith_eval: "FaithfulnessEvaluator",
        precision_engine: "HNWPrecisionEngine",
        _audit_log_override: Optional[str] = None,
    ) -> None:
        """
        GAP-17 fix: _audit_log_override allows compare_configs() to write to a
        separate log file (e.g., compare_log.jsonl), preventing pollution of
        the production audit_log.jsonl.
        """
        self.cfg             = cfg
        self.index           = index
        self.nemo            = nemo
        self.faith_eval      = faith_eval
        self.precision_engine = precision_engine

        # GAP-17: separate audit log for non-production runs
        self._audit_path = Path(
            _audit_log_override
            if _audit_log_override
            else str(cfg.log_path(cfg.log_audit))
        )
        self._crew_path = cfg.log_path(cfg.log_crew)   # GAP-15
        self._counter     = 0

    # Query ID


    def _qid(self) -> str:
        self._counter += 1
        ts = time.strftime("%H%M%S")
        return f"q_{ts}_{self._counter:04d}"

    # Context builder


    def _build_context(
        self,
        results: "List[RetrievalResult]",
    ) -> Tuple[str, List["Chunk"]]:
        """
        Build token-capped context string from results with [SOURCE:] headers.
        Returns (context_string, used_chunks).
        """
        from src.agents import _format_context
        ctx = _format_context(results, max_tokens=self.cfg.max_context_tokens)
        used_chunks = [r.chunk for r in results]
        return ctx, used_chunks

    # doc_versions collector — GAP-16


    @staticmethod
    def _collect_doc_versions(chunks: "List[Chunk]") -> Dict[str, str]:
        """
        GAP-16 fix: return {source_file: doc_version} for all used chunks.
        Only unique filenames; preserves first occurrence (chunk ordering matters).
        """
        versions: Dict[str, str] = {}
        for c in chunks:
            if c.source_file not in versions:
                versions[c.source_file] = c.doc_version
        return versions

    # Main answer pipeline


    def answer(self, query: str, verbose: bool = True) -> FinalAnswer:
        """
        Full answer pipeline with corrected agent order (GAP-03).

        CORRECTED ORDER (vs. notebook):
          Old: retrieval → precision → compliance → MANAGER → fact-check (post-synthesis)
          New: retrieval → precision → compliance → FACT-CHECK(draft) → MANAGER(+fc)

        Pipeline:
          1.  NeMo input check → route or block
          2.  L1 token importance
          3.  Precision type detection
          4.  Retrieval (hybrid FAISS+BM25+RRF+rerank+HyDE+MMR)
          5.  NeMo retrieval check (RRF confidence gate + freshness)
          6.  Route dispatch:
              a.  single_agent → single LLM call
              b.  crewai_hnw / crewai_standard → 5-agent pipeline (GAP-03 order)
          7.  NeMo output check
          8.  Citation extraction
          9.  Audit logging (audit_log.jsonl, crew_trace.jsonl, sentence_attribution.jsonl)
          10. Return FinalAnswer
        """
        t_start = time.perf_counter()
        qid = self._qid()

        if verbose:
            logger.info("=== Query [%s]: %s", qid, query[:80])

        nemo_input = self.nemo.check_input(query, qid)
        route      = nemo_input["route"]
        nemo_acts  = list(nemo_input["activations"])

        if verbose:
            logger.info("[%s] Route: %s", qid, route)

        # Hard fallback (blocked or OOS)
        if route in ("blocked", "fallback"):
            return self._make_fallback(
                qid=qid, query=query,
                route=route,
                nemo_acts=nemo_acts,
                t_start=t_start,
                answer_text=_CONTACT_CENTRE,
            )

        token_importance = self.index.emb.token_importance(query)

        ptype, params = self.precision_engine.detect_query_type(query)
        is_hnw = route in ("crewai_hnw", "crewai_standard")

        results = self.index.search(query, cfg=self.cfg)
        ctx, used_chunks = self._build_context(results)
        doc_versions = self._collect_doc_versions(used_chunks)  # GAP-16
        max_rrf = max((r.rrf_score for r in results), default=0.0)

        nemo_ret = self.nemo.check_retrieval(results, qid)
        nemo_acts.extend(nemo_ret["activations"])
        ret_warnings = nemo_ret.get("warnings", [])

        if nemo_ret.get("fallback") or not results:
            return self._make_fallback(
                qid=qid, query=query,
                route="fallback",
                nemo_acts=nemo_acts,
                t_start=t_start,
                answer_text=_CONTACT_CENTRE,
                max_rrf=max_rrf,
                doc_versions=doc_versions,
                warnings=ret_warnings,
            )

        crew_trace: Optional[CrewTrace] = None
        faith_report = None

        if route == "single_agent":
            answer_text, faith_report = self._run_single_agent(
                query, ctx, ptype, params, results, qid
            )
        else:
            # crewai_standard or crewai_hnw — full multi-agent pipeline (GAP-03 order)
            answer_text, faith_report, crew_trace = self._run_crew_pipeline(
                query, ctx, results, ptype, params, qid, route
            )

        faith_score = faith_report.overall_score if faith_report else 1.0
        nemo_out = self.nemo.check_output(
            answer=answer_text,
            qid=qid,
            faith_score=faith_score,
            is_hnw=is_hnw,
        )
        nemo_acts.extend(nemo_out["activations"])
        review_needed = nemo_out.get("review_needed", False)
        out_flags     = nemo_out.get("flags", [])

        # If the system has genuine doubt about accuracy — low faithfulness,
        # missing citation, or vague monetary language in an HNW answer —
        # it must NOT emit a potentially false answer. Instead it falls back
        # to the Contact Centre message to protect the adviser and client.
        #
        # Doubt is defined as ANY of:
        #   (a) faithfulness score < threshold (NLI says answer is unsupported)
        #   (b) no citation found in the answer AND no [Source:] in text
        #   (c) vague monetary language in an HNW answer (e.g., "approximately £")
        #
        # This implements the user requirement: "if there is doubt for the
        # single agent or crew ai agent in the answers, it must fall back to
        # a human operator and not produce false information."
        _in_doubt = False
        _doubt_reasons: List[str] = []
        original_answer = answer_text   # preserved for RAG Triad + audit

        if faith_score < self.cfg.nli_faithfulness_threshold:
            _in_doubt = True
            _doubt_reasons.append(
                f"faithfulness={faith_score:.2f} < threshold={self.cfg.nli_faithfulness_threshold}"
            )

        if "missing_citation" in out_flags:
            _in_doubt = True
            _doubt_reasons.append("no [Source:] citation in answer")

        if "vague_monetary" in out_flags:
            _in_doubt = True
            _doubt_reasons.append("vague monetary language in HNW answer")

        if _in_doubt:
            # Swap the answer for the human-operator referral message.
            # original_answer is already set above; preserved in crew_trace + RAG Triad.
            answer_text = (
                "I was unable to verify this answer with sufficient confidence from the "
                "approved source documents. "
                "Please refer this query to a human adviser or the Quilter Contact Centre "
                f"(0808 171 2626) for a verified response.\n\n"
                f"[Referral reasons: {'; '.join(_doubt_reasons)}]"
            )
            review_needed = True
            out_flags = list(set(out_flags + ["human_referral"]))
            if verbose:
                logger.warning(
                    "[%s] DOUBT-FALLBACK: answer replaced with human-operator referral. "
                    "Reasons: %s. Original answer (first 120 chars): %s",
                    qid, _doubt_reasons, original_answer[:120],
                )

        # Compute all three RAG Triad legs on the ORIGINAL answer text (before
        # doubt-fallback swap) so the scores reflect the answer quality, not the
        # referral boilerplate.  If _in_doubt, groundedness will already be low
        # (that's what caused the fallback), so the triad score is still meaningful.
        _answer_for_triad = original_answer if _in_doubt else answer_text
        _faith_for_triad  = faith_report or self._empty_faith()
        rag_triad = self.faith_eval.evaluate_rag_triad(
            query=query,
            answer=_answer_for_triad,
            results=results,
            faith_report=_faith_for_triad,
            query_id=qid,
            log_dir=self.cfg.log_dir,
        )

        # If the RAG Triad fails on Context Relevance only (retrieval pulled wrong
        # docs but the answer itself is reasonable), add a targeted warning.
        if "context_relevance" in rag_triad.weak_legs and not _in_doubt:
            all_warnings_extra = [
                f"RAG-TRIAD: context_relevance={rag_triad.context_relevance:.2f} "
                f"(retrieved chunks may not fully cover this query)"
            ]
        else:
            all_warnings_extra = []

        if verbose:
            logger.info("[%s] %s", qid, rag_triad.summary())

        citations = re.findall(r'\[Source:[^\]]+\]', answer_text)

        # Append staleness warnings
        all_warnings = ret_warnings + (
            [f"Output flags: {out_flags}"] if out_flags else []
        ) + all_warnings_extra

        # If no citations in non-referral answer, add from retrieved results
        if not citations and not _in_doubt:
            for r in results[:3]:
                c = r.chunk
                citations.append(f"[Source: {c.source_file}, p.{c.page_num}, §{c.section}]")

        latency_ms = (time.perf_counter() - t_start) * 1000

        fa = FinalAnswer(
            query_id         = qid,
            query            = query,
            answer           = answer_text,
            route_used       = route,
            citations        = citations,
            precision        = None,   # PrecisionResult stored in crew trace
            faithfulness     = faith_report or self._empty_faith(),
            crew_trace       = crew_trace,
            nemo_activations = nemo_acts,
            token_importance = token_importance,
            max_rrf_score    = max_rrf,
            latency_ms       = latency_ms,
            review_needed    = review_needed,
            warnings         = all_warnings,
            doc_versions     = doc_versions,  # GAP-16
            rag_triad        = rag_triad,     # RAG-TRIAD
        )

        self._log_audit(fa)
        self._log_crew_trace(fa)   # GAP-15

        if verbose:
            logger.info(
                "[%s] Done: route=%s faith=%.2f review=%s latency=%.0fms",
                qid, route, fa.faithfulness.overall_score, review_needed, latency_ms,
            )

        return fa

    # Single-agent route


    def _run_single_agent(
        self,
        query: str,
        context: str,
        ptype: str,
        params: Dict,
        results: "List[RetrievalResult]",
        qid: str,
    ):
        """
        Fast single-LLM-call path for simple operational queries.
        GAP-06 fix: uses call_fast (llama3.2:3b).
        BUG-FAITH-01 fix: passes `results` to faith_eval.evaluate() so each
          answer sentence is grounded to a real chunk (source_chunk_id,
          source_file, source_page are now populated in sentence_attribution.jsonl).
        """
        from src.llm_client import call_fast, SYS

        # Run precision engine even on single-agent path if applicable
        prec_context = ""
        if ptype != "none" and params:
            try:
                if ptype == "fee_calculation":
                    pr = self.precision_engine.compute_platform_fee(params["aum"], results)
                elif ptype == "chaps_fee":
                    pr = self.precision_engine.compute_chaps_fee(params["amount"], results)
                elif ptype == "ufpls_tax":
                    pr = self.precision_engine.compute_ufpls_tax(
                        params["amount"], params.get("marginal_rate", 0.20), results
                    )
                elif ptype == "threshold_check_db":
                    pr = self.precision_engine.check_db_threshold(params["tv"], results)
                else:
                    pr = None
                if pr:
                    prec_context = f"\n\nPrecision computation:\n{pr.format_for_answer()}"
            except Exception as exc:
                logger.warning("Precision engine failed on single-agent path: %s", exc)

        user_prompt = (
            f"Query: {query}\n\n"
            f"Document excerpts:\n{context}"
            f"{prec_context}"
        )
        answer_text = call_fast(SYS["single"], user_prompt, self.cfg, max_tokens=800)

        # BUG-FAITH-01 fix: pass `results` so faithfulness binds sentences to real chunks
        faith = self.faith_eval.evaluate(
            answer=answer_text,
            context=context,
            query_id=qid,
            log_dir=self.cfg.log_dir,
            results=results,
        )
        return answer_text, faith

    # Crew pipeline (GAP-03 corrected order)


    def _run_crew_pipeline(
        self,
        query: str,
        context: str,
        results: "List[RetrievalResult]",
        ptype: str,
        params: Dict,
        qid: str,
        route: str,
    ):
        """
        CORRECTED 5-agent pipeline (GAP-03 fix):
          a. Retrieval Agent (llama3.2:3b)
          b. Precision Agent (qwen2.5:7b)
          c. Compliance Agent (qwen2.5:7b)
          d. Fact-Check Agent (llama3.2:3b) ← on draft from b+c, BEFORE manager
          e. Manager Agent (qwen2.5:14b)    ← receives fact-check output

        Full CrewTrace written to crew_trace.jsonl (GAP-15).
        """
        from src.agents import (
            run_retrieval_agent,
            run_precision_agent,
            run_compliance_agent,
            run_factcheck_agent,
            run_manager_agent,
        )

        t0 = time.perf_counter()
        crew_trace = CrewTrace(query_id=qid, query=query)

        ret_output, ret_results = run_retrieval_agent(query, self.index, self.cfg)
        crew_trace.agent_steps.append(ret_output)
        # Use fresh retrieval results if available; fall back to main search results
        final_results = ret_results if ret_results else results
        fresh_ctx = context  # Use pre-built context (HyDE already applied)

        prec_output, precision_result = run_precision_agent(
            query=query,
            retrieved=fresh_ctx,
            ptype=ptype,
            params=params,
            results=final_results,
            precision_engine=self.precision_engine,
            cfg=self.cfg,
        )
        crew_trace.agent_steps.append(prec_output)

        comp_output = run_compliance_agent(
            query=query,
            retrieved=fresh_ctx,
            precision_out=prec_output.output,
            cfg=self.cfg,
        )
        crew_trace.agent_steps.append(comp_output)

        # GAP-03: Draft is built from precision + compliance, NOT from manager
        draft = (
            f"Based on precision analysis:\n{prec_output.output}\n\n"
            f"Regulatory considerations:\n{comp_output.output}"
        )

        # BUG-FAITH-01 fix: pass final_results so each draft sentence is
        # grounded to the correct source chunk (fixes source_chunk_id = "").
        fc_output, faith_report = run_factcheck_agent(
            query=query,
            draft_answer=draft,
            context=fresh_ctx,
            faith_eval=self.faith_eval,
            cfg=self.cfg,
            query_id=qid,
            log_dir=self.cfg.log_dir,
            results=final_results,
        )
        crew_trace.agent_steps.append(fc_output)

        mgr_output = run_manager_agent(
            query=query,
            retrieval_out=ret_output.output,
            precision_out=prec_output.output,
            compliance_out=comp_output.output,
            factcheck_out=fc_output.output,  # ← fact-check feeds manager
            context=fresh_ctx,
            cfg=self.cfg,
        )
        crew_trace.agent_steps.append(mgr_output)

        final_answer_text = mgr_output.output
        crew_trace.final_answer = final_answer_text
        crew_trace.total_ms = (time.perf_counter() - t0) * 1000

        return final_answer_text, faith_report, crew_trace

    # Fallback answer builder


    def _make_fallback(
        self,
        qid: str,
        query: str,
        route: str,
        nemo_acts,
        t_start: float,
        answer_text: str,
        max_rrf: float = 0.0,
        doc_versions: Optional[Dict] = None,
        warnings: Optional[List] = None,
    ) -> FinalAnswer:
        """Build a FinalAnswer for contact-centre fallback cases."""
        fa = FinalAnswer(
            query_id         = qid,
            query            = query,
            answer           = answer_text,
            route_used       = route,
            citations        = [],
            precision        = None,
            faithfulness     = self._empty_faith(),
            crew_trace       = None,
            nemo_activations = nemo_acts,
            token_importance = [],
            max_rrf_score    = max_rrf,
            latency_ms       = (time.perf_counter() - t_start) * 1000,
            review_needed    = False,
            warnings         = warnings or [],
            doc_versions     = doc_versions or {},
        )
        self._log_audit(fa)
        return fa

    # Audit logging — GAP-15, GAP-16


    def _log_audit(self, fa: FinalAnswer) -> None:
        """Write one-line scalar record to audit_log.jsonl (self._audit_path)."""
        try:
            self._audit_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._audit_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(fa.audit_dict()) + "\n")
        except Exception as exc:
            logger.error("Failed to write audit_log.jsonl: %s", exc)

    def _log_crew_trace(self, fa: FinalAnswer) -> None:
        """
        GAP-15 fix: Write full CrewTrace to separate crew_trace.jsonl.
        Linked to audit_log.jsonl via query_id.
        Only written for crew routes (not single_agent or fallback).
        """
        if fa.crew_trace is None:
            return
        try:
            self._crew_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._crew_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(fa.crew_trace.to_dict()) + "\n")
        except Exception as exc:
            logger.error("Failed to write crew_trace.jsonl: %s", exc)

    # Helpers


    @staticmethod
    def _empty_faith():
        from src.models import FaithfulnessReport
        return FaithfulnessReport(
            overall_score=1.0, sentence_scores=[], unsupported=[], is_fallback=True
        )
