"""
src/evaluation.py — Quilter HNW Advisor Assistant 

Evaluation framework with state-of-the-art metrics.

"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd  

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from src.models import Chunk
    from src.orchestrator import QuilterAdvisorSystem

try:
    from bert_score import score as _bert_score
    BERTSCORE_OK = True
except ImportError:
    BERTSCORE_OK = False
    logger.warning("bert-score not installed — BERTScore will use F1 placeholder")


# GAP-11: BERTScore


def compute_bertscore(
    predictions: List[str],
    references: List[str],
    model_type: str = "distilbert-base-uncased",
) -> Dict[str, float]:
    """
    GAP-11 fix: Compute BERTScore F1 between predictions and references.

    Replaces keyword coverage as the primary answer quality metric.
    BERTScore uses contextual embeddings — robust to paraphrasing and
    formatting differences (unlike exact string matching).

    model_type: 'distilbert-base-uncased' — lightweight, no internet required
                after first download (~260MB).

    Returns {"precision": mean_P, "recall": mean_R, "f1": mean_F1}.
    Falls back to character overlap ratio if bert_score not installed.
    """
    if not predictions or not references:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    if BERTSCORE_OK:
        try:
            P, R, F1 = _bert_score(
                cands=predictions,
                refs=references,
                lang="en",
                model_type=model_type,
                verbose=False,
            )
            return {
                "precision": float(P.mean().item()),
                "recall":    float(R.mean().item()),
                "f1":        float(F1.mean().item()),
            }
        except Exception as exc:
            logger.warning("BERTScore failed: %s — falling back to char overlap", exc)

    # Fallback: character n-gram F1
    def char_f1(pred: str, ref: str) -> float:
        pred_c = set(pred.lower().split())
        ref_c  = set(ref.lower().split())
        if not pred_c or not ref_c:
            return 0.0
        overlap = len(pred_c & ref_c)
        p = overlap / len(pred_c)
        r = overlap / len(ref_c)
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    f1_scores = [char_f1(p, r) for p, r in zip(predictions, references)]
    mean_f1 = float(np.mean(f1_scores))
    return {"precision": mean_f1, "recall": mean_f1, "f1": mean_f1}


# GAP-12: Recall@K


def recall_at_k(
    system: "QuilterAdvisorSystem",
    gold_set: List[Dict],
    k: int = 5,
) -> float:
    """
    GAP-12 fix: Recall@K with annotated ground truth.

    gold_set items must have 'relevant_chunk_ids': List[str]
    (SHA256[:12] of chunk text — matching Chunk.chunk_id).

    Returns mean Recall@K across all gold items with non-empty relevant_chunk_ids.

    Note: relevant_chunk_ids must be populated via annotate_gold_set() after
    first ingest. Items with empty lists are skipped.
    """
    recall_scores = []

    for item in gold_set:
        relevant_ids = set(item.get("relevant_chunk_ids", []))
        if not relevant_ids:
            continue  # Skip unannotated items

        query = item["query"]
        results = system.index.search(query, top_k=k, cfg=system.cfg)
        retrieved_ids = {r.chunk.chunk_id for r in results}

        recall_i = len(retrieved_ids & relevant_ids) / len(relevant_ids)
        recall_scores.append(recall_i)

    if not recall_scores:
        logger.warning(
            "recall_at_k: no annotated relevant_chunk_ids found. "
            "Run annotate_gold_set() to populate them."
        )
        return 0.0

    mean_recall = float(np.mean(recall_scores))
    logger.info("Recall@%d: %.3f (over %d annotated items)", k, mean_recall, len(recall_scores))
    return mean_recall


# GAP-13: OOS Detector F1


def oos_detector_f1(
    system: "QuilterAdvisorSystem",
    oos_queries: List[Dict],
    in_scope_queries: List[Dict],
) -> Dict[str, float]:
    """
    GAP-13 fix: OOS detector precision, recall, F1, and false positive rate.

    oos_queries:      list of {"query": str, "label": "oos"}
    in_scope_queries: list of {"query": str, "label": "in_scope"}

    For each query: check if system.nemo.check_input() returns route in
    ("blocked", "fallback") — if so, treated as detected OOS.

    Computes:
      True Positives  = OOS queries correctly routed to fallback/blocked
      False Negatives = OOS queries incorrectly routed to single/crew
      True Negatives  = In-scope queries correctly routed to single/crew
      False Positives = In-scope queries incorrectly blocked/falledback
    """
    import uuid

    tp = fn = tn = fp = 0

    for item in oos_queries:
        qid = f"oos_eval_{uuid.uuid4().hex[:6]}"
        result = system.nemo.check_input(item["query"], qid)
        if result["route"] in ("blocked", "fallback"):
            tp += 1
        else:
            fn += 1

    for item in in_scope_queries:
        qid = f"inscope_eval_{uuid.uuid4().hex[:6]}"
        result = system.nemo.check_input(item["query"], qid)
        if result["route"] in ("blocked", "fallback"):
            fp += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    fpr       = fp / (fp + tn) if (fp + tn) > 0 else 0.0  # False Positive Rate

    metrics = {
        "oos_precision":     precision,
        "oos_recall":        recall,
        "oos_f1":            f1,
        "false_positive_rate": fpr,
        "tp": tp, "fn": fn, "tn": tn, "fp": fp,
    }
    logger.info("OOS F1: %.3f (P=%.3f, R=%.3f, FPR=%.3f)", f1, precision, recall, fpr)
    return metrics


# GAP-14: Cross-Document Conflict Detection


def detect_cross_document_conflicts(chunks: "List[Chunk]") -> List[Dict]:
    """
    GAP-14 fix: Cross-document conflict detection.

    Algorithm:
      1. Group chunks by regulatory_keywords (e.g., all chunks with "mpaa")
      2. For each keyword group: extract all GBP amounts per source_file
      3. If the same keyword group has different GBP amounts across DIFFERENT
         source_files → flag as potential conflict
      4. Also checks for same-keyword, different-percentage conflicts

    Returns list of conflict records:
      {"keyword": str, "conflicting_values": List[str],
       "source_files": List[str], "chunk_ids": List[str], "type": "gbp"|"pct"}

    Example: quilter_pensions.pdf says "£30,000" but quilter_transfers.pdf
    says "£35,000" for "transfer value threshold" → flagged as conflict.
    """
    _GBP = re.compile(r'£\s*([\d,]+(?:\.\d+)?)', re.I)
    _PCT  = re.compile(r'([\d.]+)\s*%', re.I)

    # Group by keyword
    keyword_groups: Dict[str, List["Chunk"]] = {}
    for chunk in chunks:
        for kw in chunk.regulatory_keywords:
            keyword_groups.setdefault(kw, []).append(chunk)

    conflicts = []

    for keyword, kw_chunks in keyword_groups.items():
        # Group by source_file
        by_source: Dict[str, List[str]] = {}
        by_source_pct: Dict[str, List[str]] = {}

        for chunk in kw_chunks:
            src = chunk.source_file
            gbp_vals = _GBP.findall(chunk.text)
            pct_vals = _PCT.findall(chunk.text)
            by_source.setdefault(src, []).extend(
                [v.replace(",", "") for v in gbp_vals]
            )
            by_source_pct.setdefault(src, []).extend(pct_vals)

        # Only check if multiple source files mention this keyword
        if len(by_source) < 2:
            continue

        # Flatten unique values per source
        src_vals = {
            src: set(vals) for src, vals in by_source.items() if vals
        }

        # Check for conflicting GBP values across sources
        if len(src_vals) >= 2:
            # Get all unique values across all sources
            all_vals = set()
            for vals in src_vals.values():
                all_vals |= vals

            # If sources have DIFFERENT sets of values (not just subsets)
            sources = list(src_vals.keys())
            has_conflict = False
            for i in range(len(sources)):
                for j in range(i + 1, len(sources)):
                    v_i = src_vals[sources[i]]
                    v_j = src_vals[sources[j]]
                    # Conflict if they share keyword but differ in amounts
                    if v_i and v_j and not v_i.issubset(v_j) and not v_j.issubset(v_i):
                        # Further check: are the values numerically different?
                        try:
                            nums_i = {float(v) for v in v_i}
                            nums_j = {float(v) for v in v_j}
                            if not nums_i.issubset(nums_j) and not nums_j.issubset(nums_i):
                                has_conflict = True
                        except ValueError:
                            pass

            if has_conflict:
                chunk_ids = [c.chunk_id for c in kw_chunks[:4]]
                conflicts.append({
                    "keyword":            keyword,
                    "type":               "gbp",
                    "conflicting_values": {s: sorted(v) for s, v in src_vals.items()},
                    "source_files":       list(src_vals.keys()),
                    "chunk_ids":          chunk_ids,
                    "severity":           "HIGH" if keyword in (
                        "db pension", "mpaa", "annual allowance", "defined benefit"
                    ) else "MEDIUM",
                })

    logger.info(
        "Cross-document conflict scan: %d chunks, %d conflicts found",
        len(chunks), len(conflicts),
    )
    return conflicts


# Gold eval set management


def load_gold_set(eval_data_dir: str, cfg=None) -> List[Dict]:
    """
    Load the gold evaluation set.

    Args:
        eval_data_dir: Path to the eval_data directory.
        cfg: Optional Config instance. When provided, the filename is read
             from cfg.eval_gold rather than hardcoded.
    """
    name = cfg.eval_gold if cfg is not None else "gold_eval_set.json"
    path = Path(eval_data_dir) / name
    if not path.exists():
        logger.warning("Gold eval set not found at %s", path)
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def load_oos_set(eval_data_dir: str, cfg=None) -> List[Dict]:
    """
    Load the out-of-scope evaluation set.

    Args:
        eval_data_dir: Path to the eval_data directory.
        cfg: Optional Config instance. When provided, the filename is read
             from cfg.eval_oos rather than hardcoded.
    """
    name = cfg.eval_oos if cfg is not None else "oos_eval_set.json"
    path = Path(eval_data_dir) / name
    if not path.exists():
        logger.warning("OOS eval set not found at %s", path)
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def annotate_gold_set(
    system: "QuilterAdvisorSystem",
    gold_set: List[Dict],
    top_k: int = 5,
    eval_data_dir: str = "",
    cfg=None,
) -> List[Dict]:
    """
    Interactive helper to populate relevant_chunk_ids in the gold eval set.

    For each item with empty relevant_chunk_ids:
      1. Run retrieval on the query
      2. Show top-k chunks
      3. Prompt user to mark which are relevant (interactive)
    Returns updated gold_set.

    Args:
        system: QuilterAdvisorSystem instance.
        gold_set: List of gold evaluation items.
        top_k: Number of retrieved chunks to display per query.
        eval_data_dir: Path to the eval_data directory. Used for auto-save.
        cfg: Optional Config instance. When provided, filenames are read from cfg.

    Note: This is a one-time annotation task. Run once after first ingest.
    Chunk IDs are SHA256[:12] -- stable as long as chunk text does not change.
    """
    for i, item in enumerate(gold_set):
        if item.get("relevant_chunk_ids"):
            continue  # Already annotated

        query = item["query"]
        results = system.index.search(query, top_k=top_k, cfg=system.cfg)

        print(f"\n[{i+1}/{len(gold_set)}] Query: {query}")
        print("Top retrieved chunks:")
        for j, r in enumerate(results):
            c = r.chunk
            print(f"  [{j}] ({c.chunk_id}) {c.source_file} p.{c.page_num}: {c.text[:100]}...")

        relevant_str = input("Enter relevant chunk indices (e.g. '0,2') or Enter to skip: ").strip()
        if relevant_str:
            try:
                idxs = [int(x.strip()) for x in relevant_str.split(",")]
                item["relevant_chunk_ids"] = [results[i].chunk.chunk_id for i in idxs if i < len(results)]
            except (ValueError, IndexError):
                print("Invalid input - skipping")

    return gold_set


def save_gold_set(gold_set: List[Dict], eval_data_dir: str, cfg=None) -> None:
    """
    Save the annotated gold evaluation set.

    Args:
        gold_set: List of gold evaluation items.
        eval_data_dir: Path to the eval_data directory.
        cfg: Optional Config instance. When provided, the filename is read
             from cfg.eval_gold rather than hardcoded.
    """
    name = cfg.eval_gold if cfg is not None else "gold_eval_set.json"
    path = Path(eval_data_dir) / name
    path.write_text(json.dumps(gold_set, indent=2), encoding="utf-8")
    logger.info("Saved annotated gold eval set (%d items) -> %s", len(gold_set), path)


# Main evaluation runner


# NeMo Toolkit metrics


def nemo_toolkit_metrics(
    system: "QuilterAdvisorSystem",
    gold_set: List[Dict],
    audit_records: List[Dict],
) -> Dict[str, Any]:
    """
    NeMo Guardrails / NeMo Toolkit-aligned evaluation metrics.

    Computes seven NeMo-standard metrics that map directly to the Quilter
    7-layer defence stack and the NeMo Guardrails framework concepts:

    1. Rail Trigger Rate          — fraction of queries that activated ≥1 NeMo rail
    2. Rail Block Rate            — fraction of queries hard-blocked (injection/OOS)
    3. Rail Fallback Rate         — fraction of queries soft-fallbacked (Contact Centre)
    4. Hallucination Rate         — fraction of answers with faithfulness < 0.50
                                    (NLI-detected unsupported claims)
    5. Topic Adherence Rate       — fraction of in-scope queries correctly answered
                                    without Contact Centre referral
    6. Toxicity Pass Rate         — proxy: fraction of in-scope answers with no
                                    vague_monetary or missing_citation flags
    7. Rail Efficiency Score      — (blocked_correct + fallback_correct) /
                                    (total_blocked + total_fallback)
                                    i.e., are the rails blocking the right things?

    These map to NeMo Toolkit categories:
      - Input Rails:    Rail Block Rate, Rail Trigger Rate
      - Output Rails:   Hallucination Rate, Toxicity Pass Rate, Topic Adherence
      - Retrieval Rail: part of Hallucination Rate (RRF confidence gate)
      - Rail Accuracy:  Rail Efficiency Score

    Parameters
    ----------
    system        : QuilterAdvisorSystem (used for nemo.summary())
    gold_set      : List of gold eval set items (provides expected_route / should_fallback)
    audit_records : List of dicts from audit_log.jsonl (provides faithfulness, flags, route)

    Returns
    -------
    Dict with keys matching the seven metric names above, plus sub-metrics.
    """
    if not audit_records:
        logger.warning("nemo_toolkit_metrics: no audit records provided")
        return {}

    n_total = len(audit_records)

    n_rail_triggered = sum(
        1 for r in audit_records if r.get("nemo_rails_fired", 0) > 0
    )
    rail_trigger_rate = n_rail_triggered / n_total

    n_blocked = sum(1 for r in audit_records if r.get("route") == "blocked")
    rail_block_rate = n_blocked / n_total

    n_fallback = sum(1 for r in audit_records if r.get("route") == "fallback")
    rail_fallback_rate = n_fallback / n_total

    # Fraction of answered (non-fallback, non-blocked) queries where
    # faithfulness < 0.50 (NLI says answer is not grounded in source text).
    answered = [
        r for r in audit_records
        if r.get("route") not in ("fallback", "blocked")
    ]
    n_answered = len(answered)
    n_hallucinated = sum(
        1 for r in answered if r.get("faithfulness", 1.0) < 0.50
    )
    hallucination_rate = n_hallucinated / n_answered if n_answered > 0 else 0.0

    # Fraction of in-scope queries correctly handled (not incorrectly fallbacked).
    # A correctly handled in-scope query = route in (single_agent, crewai_*) and
    # faithfulness >= 0.50 OR human_referral flag present (genuine doubt).
    in_scope_answered = [
        r for r in audit_records
        if r.get("route") not in ("fallback", "blocked")
    ]
    n_topic_adherent = 0
    for r in in_scope_answered:
        warnings_str = " ".join(str(w) for w in r.get("warnings", []))
        is_human_referral = "human_referral" in warnings_str
        faith = r.get("faithfulness", 0.0)
        # Adherent if: either correctly answered (faith >= 0.50) OR legitimately
        # referred to human (which means the system correctly detected doubt).
        if faith >= 0.50 or is_human_referral:
            n_topic_adherent += 1
    topic_adherence_rate = (
        n_topic_adherent / len(in_scope_answered) if in_scope_answered else 0.0
    )

    # NeMo "toxicity" in this context = answers passing all output rail checks:
    # no missing_citation, no vague_monetary, no low_faithfulness flag.
    # Proxy metric: fraction of answered queries with NO output flags at all.
    n_clean_output = sum(
        1 for r in answered
        if not any("Output flags" in str(w) for w in r.get("warnings", []))
    )
    output_quality_pass_rate = n_clean_output / n_answered if n_answered > 0 else 0.0

    # Are the rails blocking the right things?
    # Uses gold_set to determine expected_route for each query.
    # Build query→expected map from gold set.
    gold_map = {item["query"].strip().lower()[:80]: item for item in gold_set}

    correct_blocks = 0
    incorrect_blocks = 0
    correct_fallbacks = 0
    incorrect_fallbacks = 0

    for r in audit_records:
        q_key = r.get("query", "").strip().lower()[:80]
        gold_item = gold_map.get(q_key)
        route = r.get("route", "")

        if route == "blocked":
            if gold_item and gold_item.get("should_fallback", False):
                correct_blocks += 1
            else:
                incorrect_blocks += 1
        elif route == "fallback":
            if gold_item and gold_item.get("should_fallback", False):
                correct_fallbacks += 1
            elif gold_item:
                incorrect_fallbacks += 1

    total_gated = n_blocked + n_fallback
    total_correct_gated = correct_blocks + correct_fallbacks
    rail_efficiency_score = (
        total_correct_gated / total_gated if total_gated > 0 else 0.0
    )

    metrics: Dict[str, Any] = {
        "n_total_queries":          n_total,
        "n_answered":               n_answered,
        # Input rail metrics
        "rail_trigger_rate":        round(rail_trigger_rate, 3),
        "rail_block_rate":          round(rail_block_rate, 3),
        "rail_fallback_rate":       round(rail_fallback_rate, 3),
        # Output rail / quality metrics
        "hallucination_rate":       round(hallucination_rate, 3),
        "topic_adherence_rate":     round(topic_adherence_rate, 3),
        "output_quality_pass_rate": round(output_quality_pass_rate, 3),
        # Efficiency
        "rail_efficiency_score":    round(rail_efficiency_score, 3),
        # Sub-metrics for transparency
        "n_blocked":                n_blocked,
        "n_fallback":               n_fallback,
        "n_hallucinated":           n_hallucinated,
        "n_clean_output":           n_clean_output,
        "correct_blocks":           correct_blocks,
        "incorrect_blocks":         incorrect_blocks,
        "correct_fallbacks":        correct_fallbacks,
        "incorrect_fallbacks":      incorrect_fallbacks,
    }

    # Incorporate live session summary from NeMo engine if available
    if hasattr(system, "nemo") and system.nemo is not None:
        nemo_session_summary = system.nemo.summary()
        metrics["nemo_session_summary"] = nemo_session_summary

    logger.info(
        "NeMo toolkit metrics: trigger=%.1f%% block=%.1f%% fallback=%.1f%% "
        "hallucination=%.1f%% adherence=%.1f%% quality=%.1f%% efficiency=%.1f%%",
        rail_trigger_rate * 100, rail_block_rate * 100, rail_fallback_rate * 100,
        hallucination_rate * 100, topic_adherence_rate * 100,
        output_quality_pass_rate * 100, rail_efficiency_score * 100,
    )
    return metrics


def print_nemo_scorecard(metrics: Dict[str, Any]) -> None:
    """
    Print a NeMo Toolkit-aligned scorecard with pass/fail against production targets.

    Targets are based on NeMo Guardrails recommended thresholds for a regulated
    financial services deployment:
      - Rail Trigger Rate:         informational (no pass/fail)
      - Rail Block Rate:           ≤ 10% (over-blocking harms usability)
      - Rail Fallback Rate:        ≤ 15% (excessive fallback = poor coverage)
      - Hallucination Rate:        ≤ 10% (NLI-detected unsupported claims)
      - Topic Adherence Rate:      ≥ 85% (in-scope queries correctly handled)
      - Output Quality Pass Rate:  ≥ 70% (answers with no output flags)
      - Rail Efficiency Score:     ≥ 80% (rails blocking the right queries)
    """
    print("\n" + "=" * 65)
    print("  NEMO TOOLKIT EVALUATION - QUILTER AGENTIC FRAMEWORK")
    print("=" * 65)
    print(f"  Total queries evaluated:  {metrics.get('n_total_queries', 0)}")
    print(f"  Queries answered:         {metrics.get('n_answered', 0)}")
    print()

    def _row(label: str, value: float, threshold: float, lower_better: bool,
             target_str: str) -> None:
        pct = f"{value * 100:.1f}%"
        if lower_better:
            status = "PASS" if value <= threshold else "FAIL"
        else:
            status = "PASS" if value >= threshold else "FAIL"
        print(f"  {label:<32} {pct:>7}  {target_str:<14} {status}")

    print(f"  {'Metric':<32} {'Value':>7}  {'Target':<14} {'Status'}")
    print("  " + "-" * 61)

    print(f"  {'Rail Trigger Rate':<32} {metrics.get('rail_trigger_rate', 0)*100:>6.1f}%  informational")
    _row("Rail Block Rate",          metrics.get("rail_block_rate", 0),          0.10, True,  "<= 10%")
    _row("Rail Fallback Rate",       metrics.get("rail_fallback_rate", 0),       0.15, True,  "<= 15%")
    _row("Hallucination Rate",       metrics.get("hallucination_rate", 0),       0.10, True,  "<= 10%")
    _row("Topic Adherence Rate",     metrics.get("topic_adherence_rate", 0),     0.85, False, ">= 85%")
    _row("Output Quality Pass Rate", metrics.get("output_quality_pass_rate", 0), 0.70, False, ">= 70%")
    _row("Rail Efficiency Score",    metrics.get("rail_efficiency_score", 0),    0.80, False, ">= 80%")

    print("  " + "-" * 61)
    print()
    print("  Detailed counts:")
    print(f"    Blocked queries:        {metrics.get('n_blocked', 0)}"
          f"  (correct: {metrics.get('correct_blocks', 0)}, "
          f"incorrect: {metrics.get('incorrect_blocks', 0)})")
    print(f"    Fallback queries:       {metrics.get('n_fallback', 0)}"
          f"  (correct: {metrics.get('correct_fallbacks', 0)}, "
          f"incorrect: {metrics.get('incorrect_fallbacks', 0)})")
    print(f"    Hallucinated answers:   {metrics.get('n_hallucinated', 0)}"
          f" / {metrics.get('n_answered', 0)}")
    print(f"    Clean output answers:   {metrics.get('n_clean_output', 0)}"
          f" / {metrics.get('n_answered', 0)}")

    if "nemo_session_summary" in metrics:
        s = metrics["nemo_session_summary"]
        print()
        print("  NeMo session summary (live rails):")
        print(f"    Total activations:  {s.get('total_activations', 0)}")
        print(f"    Hard blocks:        {s.get('blocked', 0)}")
        by_rail = s.get("by_rail", {})
        for rail_name, count in sorted(by_rail.items(), key=lambda x: -x[1]):
            print(f"    {rail_name:<30} {count} activations")

    print("=" * 65 + "\n")


# RAG Triad NeMo Metrics


def rag_triad_metrics(audit_records: List[Dict]) -> Dict[str, Any]:
    """
    Compute RAG Triad aggregate metrics from audit_log.jsonl records.

    Each audit record may contain (written by FinalAnswer.audit_dict()):
      rag_context_relevance  float
      rag_groundedness       float
      rag_answer_relevance   float
      rag_triad_score        float
      rag_passed             bool
      rag_weak_legs          List[str]

    Returns a dict with mean scores, pass rates, weak-leg distribution,
    and per-leg failure analysis — everything needed for the client scorecard.
    """
    triad_records = [
        r for r in audit_records
        if "rag_triad_score" in r
        and r.get("route") not in ("fallback", "blocked")
    ]

    if not triad_records:
        logger.warning(
            "rag_triad_metrics: no RAG Triad records in audit log. "
            "Run queries through the updated pipeline first."
        )
        return {
            "n_triad_evaluated": 0,
            "note": "No RAG Triad data yet — run the updated pipeline.",
        }

    n = len(triad_records)

    mean_ctx_rel  = float(np.mean([r["rag_context_relevance"] for r in triad_records]))
    mean_ground   = float(np.mean([r["rag_groundedness"]      for r in triad_records]))
    mean_ans_rel  = float(np.mean([r["rag_answer_relevance"]  for r in triad_records]))
    mean_triad    = float(np.mean([r["rag_triad_score"]       for r in triad_records]))

    ctx_pass_rate   = sum(1 for r in triad_records if r["rag_context_relevance"] >= 0.60) / n
    ground_pass_rate= sum(1 for r in triad_records if r["rag_groundedness"]      >= 0.60) / n
    ans_pass_rate   = sum(1 for r in triad_records if r["rag_answer_relevance"]  >= 0.60) / n
    triad_pass_rate = sum(1 for r in triad_records if r.get("rag_passed", False)) / n

    weak_leg_counts: Dict[str, int] = {
        "context_relevance": 0,
        "groundedness":      0,
        "answer_relevance":  0,
    }
    for r in triad_records:
        for leg in r.get("rag_weak_legs", []):
            if leg in weak_leg_counts:
                weak_leg_counts[leg] += 1

    def _bucket(scores: List[float]) -> Dict[str, int]:
        return {
            "excellent (0.80 and above)":   sum(1 for s in scores if s >= 0.80),
            "good (0.60 to 0.79)":          sum(1 for s in scores if 0.60 <= s < 0.80),
            "borderline (0.40 to 0.59)":    sum(1 for s in scores if 0.40 <= s < 0.60),
            "fail (below 0.40)":            sum(1 for s in scores if s < 0.40),
        }

    ctx_scores   = [r["rag_context_relevance"] for r in triad_records]
    grnd_scores  = [r["rag_groundedness"]      for r in triad_records]
    ans_scores   = [r["rag_answer_relevance"]  for r in triad_records]
    triad_scores = [r["rag_triad_score"]       for r in triad_records]

    sorted_records = sorted(triad_records, key=lambda r: r["rag_triad_score"])
    worst_5 = [
        {
            "query":           r.get("query", "")[:80],
            "route":           r.get("route", ""),
            "triad_score":     round(r["rag_triad_score"], 3),
            "ctx_rel":         round(r["rag_context_relevance"], 3),
            "groundedness":    round(r["rag_groundedness"], 3),
            "ans_rel":         round(r["rag_answer_relevance"], 3),
            "weak_legs":       r.get("rag_weak_legs", []),
        }
        for r in sorted_records[:5]
    ]

    best_5 = [
        {
            "query":       r.get("query", "")[:80],
            "route":       r.get("route", ""),
            "triad_score": round(r["rag_triad_score"], 3),
        }
        for r in sorted_records[-5:][::-1]
    ]

    metrics: Dict[str, Any] = {
        "n_triad_evaluated":      n,
        # Mean scores per leg
        "mean_context_relevance": round(mean_ctx_rel, 3),
        "mean_groundedness":      round(mean_ground, 3),
        "mean_answer_relevance":  round(mean_ans_rel, 3),
        "mean_triad_score":       round(mean_triad, 3),
        # Pass rates (per leg, per triad)
        "context_relevance_pass_rate":  round(ctx_pass_rate, 3),
        "groundedness_pass_rate":       round(ground_pass_rate, 3),
        "answer_relevance_pass_rate":   round(ans_pass_rate, 3),
        "triad_overall_pass_rate":      round(triad_pass_rate, 3),
        # Weak-leg distribution
        "weak_leg_counts":        weak_leg_counts,
        # Score distributions
        "context_relevance_dist":  _bucket(ctx_scores),
        "groundedness_dist":       _bucket(grnd_scores),
        "answer_relevance_dist":   _bucket(ans_scores),
        "triad_score_dist":        _bucket(triad_scores),
        # Exemplars
        "worst_5_queries":         worst_5,
        "best_5_queries":          best_5,
    }

    logger.info(
        "RAG Triad (%d records): ctx=%.2f ground=%.2f ans=%.2f → triad=%.2f "
        "pass_rate=%.1f%%",
        n, mean_ctx_rel, mean_ground, mean_ans_rel, mean_triad, triad_pass_rate * 100,
    )
    return metrics


def print_rag_triad_scorecard(triad_metrics: Dict[str, Any]) -> None:
    """
    Print a formatted RAG Triad scorecard with per-leg analysis,
    score distributions, and worst-performing query examples.

    Production targets (NeMo / TruLens aligned, financial services):
      Context Relevance pass rate:  ≥ 75%  (retrieval correctly covers query)
      Groundedness pass rate:       ≥ 80%  (answers grounded in source docs)
      Answer Relevance pass rate:   ≥ 85%  (answers address the question)
      Overall Triad pass rate:      ≥ 70%  (all three legs pass together)
    """
    if not triad_metrics or triad_metrics.get("n_triad_evaluated", 0) == 0:
        print("\n  [RAG Triad] No data yet - run the updated pipeline first.\n")
        return

    n = triad_metrics["n_triad_evaluated"]

    print("\n" + "=" * 70)
    print("  RAG TRIAD EVALUATION - QUILTER AGENTIC FRAMEWORK")
    print("  Context Relevance | Groundedness | Answer Relevance")
    print("=" * 70)
    print(f"  Queries evaluated:  {n}")
    print()

    def _row(label: str, value: float, target: float, higher_better: bool,
             target_str: str) -> None:
        pct = f"{value * 100:.1f}%"
        status = "PASS" if (value >= target if higher_better else value <= target) else "FAIL"
        print(f"  {label:<38} {pct:>7}   {target_str:<12}  {status}")

    print(f"  {'Metric':<38} {'Value':>7}   {'Target':<12}  {'Status'}")
    print("  " + "-" * 66)

    # Mean scores
    print(f"  {'Mean Context Relevance score':<38} "
          f"{triad_metrics['mean_context_relevance']*100:>6.1f}%")
    print(f"  {'Mean Groundedness score':<38} "
          f"{triad_metrics['mean_groundedness']*100:>6.1f}%")
    print(f"  {'Mean Answer Relevance score':<38} "
          f"{triad_metrics['mean_answer_relevance']*100:>6.1f}%")
    print(f"  {'Mean Triad Score (harmonic mean)':<38} "
          f"{triad_metrics['mean_triad_score']*100:>6.1f}%")
    print()

    # Pass rates with targets
    _row("Context Relevance pass rate (>=0.60)",
         triad_metrics["context_relevance_pass_rate"],  0.75, True, ">= 75%")
    _row("Groundedness pass rate (>=0.60)",
         triad_metrics["groundedness_pass_rate"],       0.80, True, ">= 80%")
    _row("Answer Relevance pass rate (>=0.60)",
         triad_metrics["answer_relevance_pass_rate"],   0.85, True, ">= 85%")
    _row("Overall Triad pass rate",
         triad_metrics["triad_overall_pass_rate"],      0.70, True, ">= 70%")

    print()
    print("  Weak-leg distribution (queries where leg score < 0.60):")
    wlc = triad_metrics.get("weak_leg_counts", {})
    for leg, count in wlc.items():
        pct = count / n * 100 if n > 0 else 0
        print(f"    {leg:<24}  {count:>3} queries ({pct:4.1f}%)")

    print()
    print("  Score distribution - Context Relevance:")
    for bucket, cnt in triad_metrics.get("context_relevance_dist", {}).items():
        print(f"    {bucket:<25}  {cnt:>3} queries")

    print()
    print("  Score distribution - Groundedness:")
    for bucket, cnt in triad_metrics.get("groundedness_dist", {}).items():
        print(f"    {bucket:<25}  {cnt:>3} queries")

    print()
    print("  Score distribution - Answer Relevance:")
    for bucket, cnt in triad_metrics.get("answer_relevance_dist", {}).items():
        print(f"    {bucket:<25}  {cnt:>3} queries")

    print()
    print("  Top 5 queries (highest triad score):")
    for item in triad_metrics.get("best_5_queries", []):
        print(f"    [{item['triad_score']:.2f}] [{item['route']:<17}]  {item['query']}")

    print()
    print("  Bottom 5 queries (lowest triad score):")
    for item in triad_metrics.get("worst_5_queries", []):
        weak = ", ".join(item.get("weak_legs", [])) or "none"
        print(f"    [{item['triad_score']:.2f}] [{item['route']:<17}]  "
              f"ctx={item['ctx_rel']:.2f} gnd={item['groundedness']:.2f} "
              f"ans={item['ans_rel']:.2f}  weak={weak}")
        print(f"           {item['query']}")

    print("=" * 70 + "\n")


# Routing Intelligence Report


def routing_intelligence_report(audit_records: List[Dict]) -> Dict[str, Any]:
    """
    Detailed analysis of intelligent routing decisions across all 58 audit queries.

    Answers the question: "How intelligently is the system routing queries —
    are simple questions going to single_agent and complex ones to crewai?"

    Computes:
      1. Route distribution (count + %) for all 5 routes
      2. Per-route quality metrics (mean faithfulness, mean RRF, mean latency)
      3. Routing correctness indicators:
           - single_agent with high RRF (≥0.03) + low latency → CORRECT (fast + confident)
           - crewai_hnw with monetary + HNW pattern → CORRECT (complex, right path)
           - single_agent with very low RRF (<0.02) → OVER-SIMPLIFIED (should fallback)
           - crewai_standard on simple definitional queries → OVER-COMPLEX
      4. Agent efficiency: mean agent_steps per route, latency per step
      5. RAG Triad per route (if rag_triad data available)
    """
    if not audit_records:
        return {}

    route_counts: Dict[str, int] = {}
    for r in audit_records:
        route = r.get("route", "unknown")
        route_counts[route] = route_counts.get(route, 0) + 1

    n_total = len(audit_records)

    route_stats: Dict[str, Dict] = {}
    for route in route_counts:
        recs = [r for r in audit_records if r.get("route") == route]
        faiths    = [r.get("faithfulness", 0.0)  for r in recs]
        rrfs      = [r.get("max_rrf", 0.0)       for r in recs]
        latencies = [r.get("latency_ms", 0.0)    for r in recs]
        steps     = [r.get("agent_steps", 1)     for r in recs]
        rails     = [r.get("nemo_rails_fired", 0) for r in recs]
        review    = [r.get("review_needed", False) for r in recs]

        # RAG Triad scores (if present)
        ctx_rels = [r["rag_context_relevance"] for r in recs if "rag_context_relevance" in r]
        grounds  = [r["rag_groundedness"]      for r in recs if "rag_groundedness" in r]
        ans_rels = [r["rag_answer_relevance"]  for r in recs if "rag_answer_relevance" in r]
        triads   = [r["rag_triad_score"]       for r in recs if "rag_triad_score" in r]

        route_stats[route] = {
            "count":               len(recs),
            "pct":                 round(len(recs) / n_total * 100, 1),
            "mean_faithfulness":   round(float(np.mean(faiths)), 3)   if faiths    else 0.0,
            "mean_rrf":            round(float(np.mean(rrfs)), 4)      if rrfs      else 0.0,
            "mean_latency_ms":     round(float(np.mean(latencies)), 1) if latencies else 0.0,
            "max_latency_ms":      round(float(np.max(latencies)), 1)  if latencies else 0.0,
            "mean_agent_steps":    round(float(np.mean(steps)), 1)     if steps     else 0.0,
            "mean_rails_fired":    round(float(np.mean(rails)), 1)     if rails     else 0.0,
            "review_needed_pct":   round(sum(review) / len(review) * 100, 1) if review else 0.0,
            "mean_ctx_relevance":  round(float(np.mean(ctx_rels)), 3) if ctx_rels  else None,
            "mean_groundedness":   round(float(np.mean(grounds)), 3)  if grounds   else None,
            "mean_ans_relevance":  round(float(np.mean(ans_rels)), 3) if ans_rels  else None,
            "mean_triad_score":    round(float(np.mean(triads)), 3)   if triads    else None,
        }

    routing_issues: List[Dict] = []

    # Over-simplified: single_agent on low-RRF queries (should have been fallback)
    for r in audit_records:
        if r.get("route") == "single_agent" and r.get("max_rrf", 1.0) < 0.02:
            routing_issues.append({
                "issue":   "over-simplified: low-RRF single_agent (should fallback)",
                "query":   r.get("query", "")[:80],
                "max_rrf": r.get("max_rrf", 0),
                "faith":   r.get("faithfulness", 0),
            })

    # Timeout risk: crewai with latency > 120s
    for r in audit_records:
        if r.get("route", "").startswith("crewai") and r.get("latency_ms", 0) > 120_000:
            routing_issues.append({
                "issue":      "timeout-risk: crewai latency > 120s",
                "query":      r.get("query", "")[:80],
                "latency_ms": r.get("latency_ms", 0),
                "route":      r.get("route", ""),
            })

    # Latency efficiency: ms per agent step
    latency_per_step: Dict[str, float] = {}
    for route, stats in route_stats.items():
        steps = stats["mean_agent_steps"]
        if steps > 0:
            latency_per_step[route] = round(stats["mean_latency_ms"] / steps, 1)

    # For each answered route: is the routing decision "efficient"?
    # Efficient = (high RRF AND high faithfulness AND reasonable latency)
    routing_quality: Dict[str, str] = {}
    for route, stats in route_stats.items():
        if route in ("fallback", "blocked"):
            routing_quality[route] = "CORRECT — safety gate"
            continue
        faith = stats["mean_faithfulness"]
        rrf   = stats["mean_rrf"]
        lat   = stats["mean_latency_ms"]
        steps = stats["mean_agent_steps"]
        if rrf >= 0.025 and faith >= 0.50 and lat < 15_000:
            routing_quality[route] = "EFFICIENT"
        elif rrf >= 0.025 and steps <= 1 and lat < 15_000:
            routing_quality[route] = "FAST - awaiting faith fix"
        elif lat > 90_000:
            routing_quality[route] = "SLOW - SLA breach"
        else:
            routing_quality[route] = "BORDERLINE"

    return {
        "n_total":           n_total,
        "route_distribution": {
            r: {"count": c, "pct": round(c / n_total * 100, 1)}
            for r, c in sorted(route_counts.items(), key=lambda x: -x[1])
        },
        "route_stats":        route_stats,
        "routing_quality":    routing_quality,
        "latency_per_step_ms": latency_per_step,
        "routing_issues":     routing_issues,
        "n_routing_issues":   len(routing_issues),
    }


def print_routing_report(report: Dict[str, Any]) -> None:
    """
    Print a complete routing intelligence report with per-route breakdown,
    RAG Triad per route, latency efficiency, and routing issue flags.
    """
    if not report:
        print("\n  [Routing Report] No data.\n")
        return

    n = report["n_total"]
    print("\n" + "=" * 75)
    print("  ROUTING INTELLIGENCE REPORT - QUILTER AGENTIC FRAMEWORK")
    print("  How intelligently are queries distributed across agent routes?")
    print("=" * 75)
    print(f"  Total queries in audit log: {n}")
    print()

    print(f"  {'Route':<22} {'Count':>6} {'%':>7}  {'Quality'}")
    print("  " + "-" * 70)
    rq = report.get("routing_quality", {})
    for route, info in report["route_distribution"].items():
        quality = rq.get(route, "")
        print(f"  {route:<22} {info['count']:>6}  {info['pct']:>5.1f}%  {quality}")

    print()
    print(f"  {'Route':<22} {'RRF':>6} {'Faith':>6} {'Steps':>6} "
          f"{'Latency(ms)':>12} {'Rails':>6} {'Review%':>8}")
    print("  " + "-" * 70)
    stats = report.get("route_stats", {})
    for route in report["route_distribution"]:
        s = stats.get(route, {})
        print(
            f"  {route:<22} "
            f"{s.get('mean_rrf', 0):>6.4f} "
            f"{s.get('mean_faithfulness', 0):>6.2f} "
            f"{s.get('mean_agent_steps', 0):>6.1f} "
            f"{s.get('mean_latency_ms', 0):>12,.0f} "
            f"{s.get('mean_rails_fired', 0):>6.1f} "
            f"{s.get('review_needed_pct', 0):>7.0f}%"
        )

    has_triad = any(
        stats.get(r, {}).get("mean_triad_score") is not None
        for r in report["route_distribution"]
    )
    if has_triad:
        print()
        print(f"  {'Route':<22} {'CtxRel':>8} {'Ground':>8} {'AnsRel':>8} "
              f"{'Triad':>8}")
        print("  " + "-" * 58)
        for route in report["route_distribution"]:
            s = stats.get(route, {})
            ct = s.get("mean_ctx_relevance")
            gr = s.get("mean_groundedness")
            ar = s.get("mean_ans_relevance")
            tr = s.get("mean_triad_score")
            if ct is not None:
                print(f"  {route:<22} {ct:>8.2f} {gr:>8.2f} {ar:>8.2f} {tr:>8.2f}")

    print()
    print("  Latency efficiency (ms per agent step):")
    for route, ms_per_step in sorted(
        report.get("latency_per_step_ms", {}).items(), key=lambda x: x[1]
    ):
        print(f"    {route:<22}  {ms_per_step:>8,.0f} ms/step")

    issues = report.get("routing_issues", [])
    if issues:
        print()
        print(f"  Routing issues detected: {len(issues)}")
        for issue in issues:
            print(f"    [{issue['issue']}]")
            print(f"       {issue.get('query', '')}")
    else:
        print()
        print("  No routing issues detected.")

    print("=" * 75 + "\n")


def run_eval(
    system: "QuilterAdvisorSystem",
    gold_set: List[Dict],
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Extended evaluation runner.

    GAP-02 fix: uses explicitly imported pandas.
    GAP-11 fix: BERTScore F1 column added (replaces keyword coverage as primary metric).

    gold_set item format:
      {
        "query":               str,
        "expected_route":      str,
        "expected_keywords":   List[str],   # retained for backwards compat
        "reference_answer":    str,         # for BERTScore
        "relevant_chunk_ids":  List[str],   # for Recall@K
        "should_fallback":     bool,
        "is_hnw_query":        bool,
      }

    Returns pd.DataFrame with one row per query.
    """
    rows = []

    for item in gold_set:
        query   = item["query"]
        exp_route = item.get("expected_route", "")
        exp_kws   = item.get("expected_keywords", [])
        ref_ans   = item.get("reference_answer", "")
        is_hnw    = item.get("is_hnw_query", False)
        should_fb = item.get("should_fallback", False)

        t0 = time.perf_counter()
        fa = system.answer(query, verbose=verbose)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        answer = fa.answer

        # Route accuracy
        actual_route = fa.route_used
        if should_fb:
            route_ok = actual_route in ("fallback", "blocked")
        elif "crewai" in exp_route:
            route_ok = "crewai" in actual_route
        else:
            route_ok = actual_route == exp_route

        # Keyword coverage (backwards compat)
        if exp_kws:
            kw_hits = sum(1 for kw in exp_kws if kw.lower() in answer.lower())
            kw_coverage = kw_hits / len(exp_kws)
        else:
            kw_coverage = 1.0

        # BERTScore (GAP-11)
        if ref_ans:
            bs = compute_bertscore([answer], [ref_ans])
            bertscore_f1 = bs["f1"]
        else:
            bertscore_f1 = None

        if verbose:
            status = "OK" if route_ok else "FAIL"
            print(f"  [{status}] {query[:50]:<50} route={actual_route} "
                  f"faith={fa.faithfulness.overall_score:.2f} "
                  f"lat={elapsed_ms:.0f}ms")

        rows.append({
            "query":          query[:80],
            "expected_route": exp_route,
            "actual_route":   actual_route,
            "route_ok":       route_ok,
            "kw_coverage":    round(kw_coverage, 3),
            "bertscore_f1":   round(bertscore_f1, 3) if bertscore_f1 is not None else None,
            "faithfulness":   round(fa.faithfulness.overall_score, 3),
            "max_rrf":        round(fa.max_rrf_score, 4),
            "latency_ms":     round(elapsed_ms, 1),
            "review_needed":  fa.review_needed,
            "nemo_rails":     len(fa.nemo_activations),
            "agent_steps":    len(fa.crew_trace.agent_steps) if fa.crew_trace else 1,
            "is_hnw":         is_hnw,
        })

    df = pd.DataFrame(rows)

    print("\n" + "=" * 60)
    print("  EVALUATION RESULTS")
    print("=" * 60)

    n = len(df)
    n_hnw = df["is_hnw"].sum()
    n_std = n - n_hnw

    print(f"\n  Total queries: {n} ({n_hnw} HNW, {n_std} standard)")
    print(f"\n  Core metrics:")
    print(f"    Route accuracy:      {df['route_ok'].mean()*100:.1f}%")
    print(f"    Mean faithfulness:   {df['faithfulness'].mean():.3f}")

    bs_vals = df["bertscore_f1"].dropna()
    if not bs_vals.empty:
        print(f"    Mean BERTScore F1:  {bs_vals.mean():.3f}  (primary quality metric)")
    else:
        print(f"    BERTScore F1:       N/A (no reference answers provided)")

    print(f"    Keyword coverage:    {df['kw_coverage'].mean()*100:.1f}%  (secondary)")
    print(f"    Mean latency:        {df['latency_ms'].mean():.0f}ms")
    print(f"    P95 latency:         {df['latency_ms'].quantile(0.95):.0f}ms")
    print(f"    Review-flagged:      {df['review_needed'].sum()} / {n}")

    if n_hnw > 0:
        hnw_df = df[df["is_hnw"]]
        print(f"\n  HNW queries ({n_hnw}):")
        print(f"    Route accuracy:      {hnw_df['route_ok'].mean()*100:.1f}%")
        print(f"    Mean faithfulness:   {hnw_df['faithfulness'].mean():.3f}")

    print("=" * 60 + "\n")

    return df


# Executive scorecard


def print_scorecard(df: pd.DataFrame, system: "QuilterAdvisorSystem") -> None:
    """
    Executive scorecard with pass/fail against targets.
    Updated targets: BERTScore F1 ≥ 0.65 replaces keyword coverage ≥ 70%.
    """
    n = len(df)

    metrics = {
        "Route Accuracy":        (df["route_ok"].mean(),           0.90,  ">= 90%"),
        "Mean Faithfulness":     (df["faithfulness"].mean(),        0.50,  ">= 0.50"),
        "Review Flag Rate":      (df["review_needed"].mean(),       None,  "<= 10%"),  # lower=better
    }

    bs_vals = df["bertscore_f1"].dropna()
    if not bs_vals.empty:
        metrics["BERTScore F1"] = (bs_vals.mean(), 0.65, ">= 0.65")

    fallback_acc = df[df["expected_route"].str.contains("fallback", na=False)]["route_ok"].mean()
    if not np.isnan(fallback_acc):
        metrics["Fallback Accuracy"] = (fallback_acc, 0.90, ">= 90%")

    print("\n" + "=" * 55)
    print("  EXECUTIVE SCORECARD")
    print("=" * 55)
    print(f"  {'Metric':<25} {'Value':>8}  {'Target':<10} {'Status'}")
    print("  " + "-" * 51)

    all_pass = True
    for metric_name, (value, threshold, target_str) in metrics.items():
        if np.isnan(value):
            status = "N/A"
        elif metric_name == "Review Flag Rate":
            passed = value <= 0.10
            status = "PASS" if passed else "FAIL"
            all_pass = all_pass and passed
        elif threshold is not None:
            passed = value >= threshold
            status = "PASS" if passed else "FAIL"
            all_pass = all_pass and passed
        else:
            status = "N/A"

        fmt_value = f"{value*100:.1f}%" if "Rate" in metric_name or "Accuracy" in metric_name else f"{value:.3f}"
        print(f"  {metric_name:<25} {fmt_value:>8}  {target_str:<10} {status}")

    print("  " + "-" * 51)
    overall = "ALL PASS" if all_pass else "SOME FAILURES"
    print(f"  Overall: {overall}")
    print("=" * 55 + "\n")
