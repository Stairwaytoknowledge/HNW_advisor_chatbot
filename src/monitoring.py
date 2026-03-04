"""
src/monitoring.py — Quilter HNW Advisor Assistant 

MonitoringDashboard — reads JSONL log files and produces operational metrics.
Extended to read crew_trace.jsonl  for per-agent latency breakdown.

Alert thresholds:
  fallback rate       > 30%
  mean faithfulness   < 0.40
  review flag rate    > 25%
  P95 latency         > 8000ms
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class MonitoringDashboard:
    """
    Reads JSONL log files from logs_v3/ and produces an operational report.

    Log files read:
      audit_log.jsonl          — one record per query (scalars)
      nemo_rail_log.jsonl      — one record per rail activation
      crew_trace.jsonl         — one record per crew query (full agent trace, GAP-15)
      sentence_attribution.jsonl — one record per sentence (GAP-15)
      update_log.jsonl         — document update history
    """

    def __init__(self, log_dir: str, cfg=None) -> None:
        """
        Args:
            log_dir: Path to the log directory.
            cfg: Optional Config instance. When provided, log filenames are read
                 from cfg (e.g. cfg.log_audit) rather than hardcoded strings.
        """
        self.log_dir = Path(log_dir)
        self._cfg = cfg

    # Log readers


    def _load_jsonl(self, filename: str) -> List[Dict]:
        path = self.log_dir / filename
        if not path.exists():
            return []
        records = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return records

    def _load_audit(self) -> List[Dict]:
        return self._load_jsonl(self._cfg.log_audit if self._cfg else "audit_log.jsonl")

    def _load_nemo(self) -> List[Dict]:
        return self._load_jsonl(self._cfg.log_nemo if self._cfg else "nemo_rail_log.jsonl")

    def _load_crew_traces(self) -> List[Dict]:  # GAP-15
        return self._load_jsonl(self._cfg.log_crew if self._cfg else "crew_trace.jsonl")

    def _load_attributions(self) -> List[Dict]:
        return self._load_jsonl(self._cfg.log_attribution if self._cfg else "sentence_attribution.jsonl")

    def _load_updates(self) -> List[Dict]:
        return self._load_jsonl(self._cfg.log_update if self._cfg else "update_log.jsonl")

    # Report


    def report(self) -> None:
        """Print comprehensive operational report to stdout."""
        audit  = self._load_audit()
        nemo   = self._load_nemo()
        traces = self._load_crew_traces()
        attrs  = self._load_attributions()
        updates = self._load_updates()

        print("\n" + "=" * 60)
        print("  QUILTER HNW ADVISOR — MONITORING DASHBOARD")
        print("=" * 60)

        if not audit:
            print("  No audit records found. Run some queries first.")
            print("=" * 60)
            return

        n = len(audit)
        print(f"\n  Total queries: {n}\n")

        routes = Counter(r.get("route", "unknown") for r in audit)
        print("  Route distribution:")
        for route, count in sorted(routes.items(), key=lambda x: -x[1]):
            pct = count / n * 100
            print(f"    {route:<20} {count:>4}  ({pct:.1f}%)")

        latencies    = [r.get("latency_ms", 0) for r in audit]
        faithfulness = [r.get("faithfulness", 0) for r in audit]
        review_flags = [r for r in audit if r.get("review_needed")]
        fallback_q   = [r for r in audit if r.get("route") in ("fallback", "blocked")]
        nemo_counts  = [r.get("nemo_rails_fired", 0) for r in audit]

        mean_faith = float(np.mean(faithfulness)) if faithfulness else 0
        p50_lat    = float(np.percentile(latencies, 50)) if latencies else 0
        p95_lat    = float(np.percentile(latencies, 95)) if latencies else 0

        print(f"\n  Key metrics:")
        print(f"    Mean faithfulness:   {mean_faith:.3f}")
        print(f"    P50 latency:         {p50_lat:.0f} ms")
        print(f"    P95 latency:         {p95_lat:.0f} ms")
        print(f"    Review-flagged:      {len(review_flags)} / {n} ({len(review_flags)/n*100:.1f}%)")
        print(f"    Fallback rate:       {len(fallback_q)} / {n} ({len(fallback_q)/n*100:.1f}%)")
        print(f"    Mean rails fired:    {float(np.mean(nemo_counts)):.1f}" if nemo_counts else "")

        alerts = []
        if len(fallback_q) / n > 0.30:
            alerts.append(f"HIGH FALLBACK RATE: {len(fallback_q)/n*100:.1f}% > 30% threshold")
        if mean_faith < 0.40:
            alerts.append(f"LOW FAITHFULNESS: {mean_faith:.3f} < 0.40 threshold")
        if len(review_flags) / n > 0.25:
            alerts.append(f"HIGH REVIEW RATE: {len(review_flags)/n*100:.1f}% > 25% threshold")
        if p95_lat > 8000:
            alerts.append(f"HIGH P95 LATENCY: {p95_lat:.0f}ms > 8000ms threshold")

        if alerts:
            print("\n  ALERTS:")
            for alert in alerts:
                print(f"    - {alert}")
        else:
            print("\n  All metrics within thresholds.")

        if nemo:
            print(f"\n  NeMo rail activations ({len(nemo)} total):")
            rail_counts = Counter(r.get("rail_name", "unknown") for r in nemo)
            for rail, count in sorted(rail_counts.items(), key=lambda x: -x[1]):
                print(f"    {rail:<30} {count:>4}")
            blocked = sum(1 for r in nemo if r.get("was_blocked"))
            print(f"    Blocked queries:               {blocked:>4}")

        if traces:
            print(f"\n  Agent latency breakdown ({len(traces)} crew queries):")
            agent_lats: Dict[str, List[float]] = {}
            for trace in traces:
                for step in trace.get("agent_steps", []):
                    agent = step.get("agent_name", "unknown")
                    lat   = step.get("latency_ms", 0)
                    agent_lats.setdefault(agent, []).append(lat)
            for agent, lats in sorted(agent_lats.items()):
                mean_lat = float(np.mean(lats))
                print(f"    {agent:<25} mean={mean_lat:.0f}ms  n={len(lats)}")

        if attrs:
            labels = Counter(r.get("nli_label", "?") for r in attrs)
            total  = len(attrs)
            print(f"\n  Sentence attribution (L3 NLI) - {total} sentences:")
            for label, count in sorted(labels.items()):
                print(f"    {label:<15} {count:>5}  ({count/total*100:.1f}%)")

        all_sources = []
        for r in audit:
            all_sources.extend(r.get("sources", []))
        if all_sources:
            source_counts = Counter(all_sources)
            print(f"\n  Most-cited sources:")
            for src, cnt in source_counts.most_common(5):
                print(f"    {src:<35} {cnt:>4} queries")

        if updates:
            recent = [u for u in updates if u.get("status") in ("added", "updated")]
            print(f"\n  Document updates: {len(recent)} changed files in update_log")
            for u in recent[-5:]:
                print(f"    {u.get('ts', '')[:10]}  {u.get('filename', '')}  {u.get('status', '')}")

        print("\n" + "=" * 60 + "\n")

    # Query-level audit lookup


    def lookup_query(self, query_id: str) -> Dict[str, Any]:
        """
        Return all audit artefacts for a given query_id.
        Answers compliance questions like:
          "Was this query answered from current documents?"
          "Which agent produced the fee calculation?"
          "Is the MPAA figure in the answer supported by source text?"
        """
        audit  = [r for r in self._load_audit()       if r.get("qid") == query_id]
        nemo   = [r for r in self._load_nemo()        if r.get("query_id") == query_id]
        traces = [r for r in self._load_crew_traces() if r.get("query_id") == query_id]
        attrs  = [r for r in self._load_attributions() if r.get("query_id") == query_id]

        return {
            "query_id":            query_id,
            "audit_record":        audit[0] if audit else None,
            "nemo_activations":    nemo,
            "crew_trace":          traces[0] if traces else None,
            "sentence_attributions": attrs,
        }
