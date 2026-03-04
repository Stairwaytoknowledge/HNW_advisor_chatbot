"""
src/display.py — Quilter HNW Advisor Assistant 

display_answer() — rich console renderer for FinalAnswer.
Extracted from notebook cell 25 with no logic changes.

Displays:
  - Answer text with route badge
  - L1 Token importance bar chart
  - L2 Agent trace (if crew route)
  - L3 Sentence NLI faithfulness breakdown
  - NeMo rail activations
  - Precision computation working (if applicable)
  - Citations and warnings
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.models import FinalAnswer


def display_answer(
    fa: "FinalAnswer",
    show_agent_trace: bool = True,
    show_l1: bool = True,
    show_l3: bool = True,
) -> None:
    """
    Rich console display of a FinalAnswer.

    Parameters:
      show_agent_trace -- show L2 CrewAI agent trace (default: True)
      show_l1          -- show L1 token importance (default: True)
      show_l3          -- show L3 sentence NLI breakdown (default: True)
    """
    SEP = "-" * 70
    THICK = "=" * 70

    route_label = {
        "crewai_hnw":      "HNW Multi-Agent (5 Agents)",
        "crewai_standard": "Standard Multi-Agent",
        "single_agent":    "Single Agent",
        "fallback":        "Contact Centre Fallback",
        "blocked":         "Blocked (Injection/OOS)",
    }.get(fa.route_used, fa.route_used)

    review_label = "YES - REVIEW NEEDED" if fa.review_needed else "NO"

    print(f"\n{THICK}")
    print(f"  Query [{fa.query_id}]: {fa.query}")
    print(f"  Route: {route_label}  |  Latency: {fa.latency_ms:.0f}ms  |  "
          f"Faith: {fa.faithfulness.overall_score:.2f}  |  "
          f"Review: {review_label}")
    print(THICK)

    # -- Answer --
    print(f"\nANSWER\n{SEP}")
    print(fa.answer)

    # -- Warnings --
    if fa.warnings:
        print(f"\nWARNINGS")
        for w in fa.warnings:
            print(f"  - {w}")

    # -- L1: Token Importance --
    if show_l1 and fa.token_importance:
        print(f"\nL1 TOKEN IMPORTANCE (retrieval drivers)\n{SEP}")
        for token, score in fa.token_importance[:8]:
            bar_width = int(score * 40)
            bar = "#" * bar_width
            print(f"  {token:<20} {bar:<40} {score:.3f}")

    # -- L2: Agent Trace --
    if show_agent_trace and fa.crew_trace:
        print(f"\nL2 AGENT TRACE\n{SEP}")
        for i, step in enumerate(fa.crew_trace.agent_steps, 1):
            print(f"\n  [{i}] {step.agent_name}  ({step.latency_ms:.0f}ms)")
            if step.reasoning:
                print(f"      Reasoning: {step.reasoning}")
            if step.tool_calls:
                for tc in step.tool_calls:
                    print(f"      Tool: {tc.get('tool','')}({tc.get('input','')[:60]})")
                    out = tc.get('output', '')
                    if out:
                        print(f"        -> {out[:100]}")
            out_preview = step.output[:200].replace("\n", " ")
            print(f"      Output: {out_preview}...")

    # -- L3: Sentence NLI --
    if show_l3 and fa.faithfulness.sentence_scores:
        print(f"\nL3 SENTENCE FAITHFULNESS (NLI)\n{SEP}")
        print(f"  Overall: {fa.faithfulness.summary()}")
        print()
        for detail in fa.faithfulness.sentence_scores:
            icon  = "OK  " if detail.supported else "FAIL"
            label = detail.label
            conf  = detail.confidence
            sent  = detail.sentence[:90]
            print(f"  [{icon}] {label:<14} {conf:.2f}  {sent}")
        if fa.faithfulness.unsupported:
            print(f"\n  Unsupported sentences ({len(fa.faithfulness.unsupported)}) "
                  f"-- hallucination risk:")
            for u in fa.faithfulness.unsupported:
                print(f"    - {u[:100]}")

    # -- NeMo Rail Activations --
    if fa.nemo_activations:
        print(f"\nNEMO RAIL ACTIVATIONS ({len(fa.nemo_activations)})\n{SEP}")
        for act in fa.nemo_activations:
            blocked = "BLOCKED" if act.was_blocked else "logged"
            print(f"  [{act.rail_type}] {act.rail_name}  ->  {act.action_taken}  [{blocked}]")
            print(f"    Trigger: {act.trigger[:80]}")

    # -- Citations --
    if fa.citations:
        print(f"\nCITATIONS\n{SEP}")
        for cit in fa.citations[:8]:
            print(f"  {cit}")

    # -- Doc versions (audit) --
    if fa.doc_versions:
        print(f"\nDOCUMENT VERSIONS (audit trail)\n{SEP}")
        for fname, sha in fa.doc_versions.items():
            sha_preview = sha[:16] + "..." if len(sha) > 16 else sha
            print(f"  {fname:<35} sha256:{sha_preview}")

    print(f"\n{THICK}\n")
