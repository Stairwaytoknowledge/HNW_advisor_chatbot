"""
src/guardrails.py — Quilter HNW Advisor Assistant 

NeMo Guardrails engine with Ollama backend.
Falls back to Python-regex implementation with identical interface if NeMo unavailable.

Three checkpoint methods applied to every query:
  check_input(query, qid)     — injection block, OOS detection, HNW routing
  check_retrieval(results, qid) — RRF confidence gate, doc freshness
  check_output(answer, qid, faith_score, is_hnw) — citation check, precision, faithfulness

Colang files: rails/main.co and rails/hnw_rails.co
NeMo config:  rails/config.yml (Ollama backend: qwen2.5:7b)
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from src.models import RailActivation

logger = logging.getLogger(__name__)

try:
    from nemoguardrails import RailsConfig, LLMRails
    NEMO_OK = True
except ImportError:
    RailsConfig = None  # type: ignore[assignment]
    LLMRails    = None  # type: ignore[assignment]
    NEMO_OK = False
    logger.info("nemoguardrails not installed — using Python-regex fallback")

if TYPE_CHECKING:
    from src.config import Config
    from src.models import RetrievalResult


# Rail pattern definitions


# Out-of-scope patterns (clear non-Quilter queries)
# NOTE: Order matters — more specific patterns first.
_OOS_PAT = [
    re.compile(r'\bweather\b', re.I),
    re.compile(r'\brestaurant\b|\bdining\b', re.I),
    # Stock/share prices — catches "stock prices", "share price", "FTSE", "stock price of"
    re.compile(r'\bshare\s+prices?\b|\bstock\s+prices?\b|\bftse\b', re.I),
    # "current stock price", "what is the stock price" — real-time financial data
    re.compile(r'(?:current|today.?s?|latest|live)\s+stock\s+price', re.I),
    re.compile(r'\bjoke\b|\bfunny\b', re.I),
    # Sports — catches any sports query, not just results/scores
    re.compile(r'\b(?:premier\s+league|champions\s+league|world\s+cup|football|cricket|rugby|tennis|golf)\b', re.I),
    re.compile(r'\bsports?\s+(?:result|score|news|team|club|match|game)\b', re.I),
    re.compile(r'\bcrypto(?:currency)?\b|\bbitcoin\b|\bethereum\b', re.I),
    re.compile(r'\bnews\s+today\b', re.I),
    # Programming / coding requests unrelated to Quilter
    re.compile(r'\b(?:write|create|generate)\s+(?:me\s+)?(?:a\s+)?(?:python|javascript|java|sql|bash|script|code|program)\b', re.I),
    # Central bank policy rates (OOS — Quilter doesn't set or advise on base rates)
    re.compile(r'\bbank\s+of\s+england\s+(?:base\s+)?rate\b', re.I),
    re.compile(r'\bbase\s+rate\s+(?:today|current|now|forecast)\b', re.I),
    # Real-time / live data requests — "current X", "today's X", "right now", "as of today"
    # These are always OOS because Quilter's knowledge base is static document-based.
    re.compile(r'\bcurrent\s+(?:interest|base|inflation|mortgage)\s+rate\b', re.I),
    re.compile(r'\bright\s+now\b|\bas\s+of\s+today\b|\btoday.s\s+rate\b', re.I),
    re.compile(r'\bwhat\s+is\s+the\s+(?:current|latest|live|today.?s?)\s+\w+\s+(?:rate|price|level|index)\b', re.I),
    # Generic investment opinion (not platform-specific advice queries)
    re.compile(r'\b(?:should\s+i|shall\s+i)\s+invest\s+in\b', re.I),
    re.compile(r'\b(?:gold|silver|commodities?)\s+(?:vs?\.?|or|versus)\s+(?:bonds?|equities|stocks?|shares?)\b', re.I),
    # Generic arithmetic / percentage maths with "investment decision" framing
    re.compile(r'\bwhat\s+is\s+\d+%\s+of\b', re.I),
    re.compile(r'\binvestment\s+decision\b', re.I),
]

# Prompt injection patterns — covers all injection test cases including sophisticated jailbreaks
# BUG-OOS-01 fix: added patterns for "DAN", "SYSTEM:", persona-swap, and gradual escalation.
_INJ_PAT = [
    # "Ignore [your/my/all/previous/any] instructions" — flexible word-order
    re.compile(r'ignore\s+(?:\w+\s+){0,2}instructions?', re.I),
    # "You are now X" — catches "You are now DAN", "You are now a helpful bot", etc.
    re.compile(r'you\s+are\s+now\s+\w', re.I),
    # DAN / jailbreak keywords
    re.compile(r'\b(?:DAN|do\s+anything\s+now|jailbreak|jail\s+break)\b', re.I),
    # Disregard / forget directives
    re.compile(r'disregard\s+(?:your\s+)?(?:system\s+)?(?:prompt|restrictions?|rules?|instructions?)', re.I),
    re.compile(r'forget\s+everything\s+(?:above|I\s+told\s+you)', re.I),
    # act as / pretend — catches "pretend you have no restrictions", "act as"
    re.compile(r'\bact\s+as\b', re.I),
    re.compile(r'\bpretend\s+(?:you\s+(?:are|have)|to\s+be)', re.I),
    # SYSTEM: prefix injection (catches "SYSTEM: Override compliance mode")
    re.compile(r'^SYSTEM\s*:', re.I | re.MULTILINE),
    re.compile(r'override\s+(?:compliance|safety|restrictions?|mode|system)', re.I),
    # "no content restrictions" / "no restrictions" phrasing
    re.compile(r'no\s+(?:content\s+)?restrictions?', re.I),
    # Evade/circumvent regulatory/compliance
    re.compile(r'evade\s+(?:fca|regulatory|compliance|rules?)\b', re.I),
    # BUG-OOS-01 fix: reveal / expose system prompt attempts
    re.compile(r'(?:reveal|show|tell\s+me|expose|output)\s+(?:your\s+)?(?:system\s+)?(?:prompt|instructions?|rules?|config)', re.I),
    # BUG-OOS-01 fix: "from now on" persona-swap directive
    re.compile(r'from\s+now\s+on\s+(?:you\s+are|act|respond|behave)', re.I),
    # BUG-OOS-01 fix: "answer freely" / "answer without restrictions"
    re.compile(r'answer\s+(?:freely|without\s+(?:restrictions?|limits?|filters?))', re.I),
    # BUG-OOS-01 fix: "developer mode" / "god mode" / "admin mode"
    re.compile(r'\b(?:developer|admin|god|unrestricted|unlocked)\s+mode\b', re.I),
    # BUG-OOS-01 fix: hypothetical frame that asks for restricted content
    re.compile(r'hypothetically\s+(?:speaking\s+)?(?:if\s+you\s+(?:had|could|were)|how\s+would\s+you)', re.I),
    # BUG-OOS-01 fix: "in a story" / "in a roleplay" evasion frames
    re.compile(r'(?:in\s+a\s+(?:story|roleplay|novel|game|simulation|hypothetical))[^.]{0,50}(?:tell|explain|describe|say)', re.I),
]

# HNW escalation patterns (route to crewai_hnw)
_HNW_PAT = [
    # Monetary amount adjacent to a pension/fee/transfer action
    re.compile(r'£\s*[\d,]+.*?(?:fee|charge|pension|transfer|withdraw)', re.I),
    re.compile(r'(?:fee|charge|platform)\s+(?:for|of|on)\s+.*?£\s*[\d,]+', re.I),
    # DB schemes — catches "DB transfer", "DB pension", "defined benefit"
    re.compile(r'\bdefined\s+benefit\b|\bdb\s+(?:pension|transfer|scheme|plan)\b', re.I),
    re.compile(r'\bmpaa\b|\bmoney\s+purchase\s+annual\s+allowance\b', re.I),
    re.compile(r'\bufpls\b|\buncrystallised\s+funds\b', re.I),
    re.compile(r'\bannual\s+allowance\s+(?:charge|limit|breach)\b', re.I),
    re.compile(r'\bcobs\s*19\b', re.I),
    re.compile(r'\bcarry.?forward\b', re.I),
    re.compile(r'\btransfer\s+value\b', re.I),
    # Flexi-access drawdown — client-specific contribution limit scenarios
    re.compile(r'\bflexi.?access\s+drawdown\b', re.I),
    re.compile(r'\bmoney\s+purchase\s+(?:pension|annual)\b', re.I),
    # Politically Exposed Person — compliance-heavy onboarding/monitoring process
    re.compile(r'\bpolitically\s+exposed\b|\bpep\s+(?:client|status|check|screening|risk|onboard)\b', re.I),
]

# HNW monetary value pattern (used in routing logic)
_MONETARY_PAT = re.compile(r'£\s*[\d,]+', re.I)

# Informational / lookup query pattern — no client-specific calculation needed.
# Matches "what is/are/does", "explain", "how does/do/many/long", "when does/is",
# "define", "describe", etc.  When hnw_hits > 0 but no monetary £ value, routes
# to single_agent (regulatory lookup, not client calculation).
_INFORMATIONAL_PAT = re.compile(
    # Matches pure definitional / threshold-lookup queries ONLY.
    # Deliberately excludes procedural / compliance questions ("what documents",
    # "what happens if", "can X and Y be combined", "what are the consequences").
    #
    # Triggers single_agent when combined with hnw_hits > 0 and no £ amount:
    #   "What is the MPAA?"  "When is MPAA triggered?"  "What triggers the MPAA?"
    #   "How many years for carry-forward?"  "Does UFPLS trigger the MPAA?"
    #   "Explain the MPAA"   "Define carry-forward"
    #
    # Does NOT match (→ crewai_standard):
    #   "What documents are needed for DB transfer?"  (procedural)
    #   "What happens if DB transfer proceeds without advice?"  (consequence)
    #   "Can MPAA be used with carry-forward?"  (combined-rules question)
    r'^(?:'
    # "What is/are/was the MPAA/UFPLS/etc?" — pure definition (NOT "what documents", "what happens")
    r'what\s+(?:is|are|was)\s+(?:the\s+)?(?:current\s+)?(?:mpaa|ufpls|db\b|carry.?forward|annual\s+allowance|threshold|limit|rule|trigger)|'
    # "What triggers the X?"
    r'what\s+triggers?\b|'
    # "When does/is X triggered/applied?"
    r'when\s+(?:does|is|do)\s+(?:the\s+)?(?:mpaa|ufpls|db\b|carry.?forward)|'
    # "When does X apply / trigger / kick in?"
    r'when\s+(?:does|do)\s+\w+(?:\s+\w+)?\s+(?:apply|trigger|kick)\b|'
    # "How many / how long" — quantity lookup (NOT "how much tax", "how do I")
    r'how\s+(?:many|long)\b|'
    # "Does [anything] trigger/apply/count/mean?" — binary factual yes/no
    r'does\s+(?:\w+\s+){0,4}(?:trigger|apply|count|affect|mean)\b|'
    # "Explain the MPAA/UFPLS/carry-forward/DB" — specific regulatory term
    r'explain\s+(?:the\s+)?(?:mpaa|ufpls|carry.?forward|db\b|annual\s+allowance|money\s+purchase)|'
    r'define\b'
    r')',
    re.I,
)

# In-scope keywords (override OOS detection if present).
# IMPORTANT: exclude generic words like "investment" and "fund" that appear in
# clearly OOS queries (e.g. "invest in gold", "investment decision") — only
# Quilter-specific platform terminology qualifies as an in-scope override.
_IN_SCOPE_KEYWORDS = {
    # Quilter-specific platform terms — deliberately narrow to avoid false negatives
    "quilter",
    "platform fee", "platform charge",
    "isa", "sipp",
    "pension transfer", "pension drawdown",
    "onboarding", "kyc",
    "drawdown pension", "flexi-access drawdown",
    "origo", "re-registration",
    "lump sum", "tax-free cash",
    "expression of wishes",
    "consumer duty",
    "politically exposed",
    "defined benefit", "db pension",
    "mpaa", "money purchase annual allowance",
    "ufpls", "uncrystallised funds",
    "carry forward", "annual allowance",
    "chaps", "safe custody",
    "death benefit",
}

# Vague monetary language patterns (precision rail)
_MON_PAT = [
    re.compile(r'approximately\s+£', re.I),
    re.compile(r'roughly\s+£', re.I),
    re.compile(r'around\s+£', re.I),
    re.compile(r'about\s+£', re.I),
    re.compile(r'estimated\s+£', re.I),
    re.compile(r'circa\s+£', re.I),
]


class NeMoEngine:
    """
    Guardrail engine with NeMo Guardrails (Ollama backend) + Python-regex fallback.

    Real NeMo is used when nemoguardrails is installed and rails/config.yml is valid.
    Python fallback implements identical logic and is fully auditable.

    Every rail activation produces a RailActivation record written to nemo_rail_log.jsonl.
    """

    def __init__(self, cfg: "Config") -> None:
        self.cfg    = cfg
        self._real  = None
        self._log: List[RailActivation] = []
        self._log_path = cfg.log_path(cfg.log_nemo)

        if not cfg.nemo_enabled:
            logger.info("NeMo guardrails disabled in config")
            return

        if NEMO_OK:
            try:
                rc = RailsConfig.from_path(cfg.rails_dir)
                self._real = LLMRails(rc)
                logger.info("NeMo Guardrails loaded from %s (Ollama backend)", cfg.rails_dir)
            except Exception as exc:
                logger.warning(
                    "NeMo Guardrails init failed: %s — using Python-regex fallback", exc
                )
                self._real = None
        else:
            logger.info("NeMo package not installed — Python-regex guardrails active")

    # Internal activation logger


    def _act(
        self,
        query_id: str,
        rail_type: str,
        rail_name: str,
        trigger: str,
        action: str,
        blocked: bool,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> RailActivation:
        activation = RailActivation(
            timestamp    = datetime.now(timezone.utc).isoformat(),
            query_id     = query_id,
            rail_type    = rail_type,
            rail_name    = rail_name,
            trigger      = trigger,
            action_taken = action,
            was_blocked  = blocked,
            metadata     = metadata or {},
        )
        self._log.append(activation)
        self._write_activation(activation)
        return activation

    def _write_activation(self, activation: RailActivation) -> None:
        try:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(activation)) + "\n")
        except Exception as exc:
            logger.error("Failed to write nemo_rail_log.jsonl: %s", exc)

    # Input rails


    def check_input(self, query: str, qid: str) -> Dict[str, Any]:
        """
        Input rail checkpoint. Returns:
          {"route": str, "activations": List[RailActivation], "blocked": bool}

        Routes:
          "blocked"        — injection or hard OOS detected
          "fallback"       — soft OOS (contact centre)
          "crewai_hnw"     — HNW complex query (multi-domain, monetary + regulatory)
          "crewai_standard"— monetary query without explicit regulatory pattern
          "single_agent"   — simple operational query
        """
        activations: List[RailActivation] = []

        for pat in _INJ_PAT:
            m = pat.search(query)
            if m:
                act = self._act(
                    qid, "input", "injection_block",
                    trigger=m.group(0), action="block_query", blocked=True,
                    metadata={"pattern": pat.pattern},
                )
                activations.append(act)
                return {"route": "blocked", "activations": activations, "blocked": True}

        q_lower = query.lower()

        oos_hits = sum(1 for p in _OOS_PAT if p.search(query))
        in_scope_hits = sum(1 for kw in _IN_SCOPE_KEYWORDS if kw in q_lower)

        # Monetary value alone is NOT enough to pull an OOS query in-scope.
        # Only genuine Quilter-platform keyword hits override OOS detection.
        if oos_hits > 0 and in_scope_hits == 0:
            act = self._act(
                qid, "input", "oos_detection",
                trigger=f"{oos_hits} OOS pattern(s), 0 in-scope keywords",
                action="contact_centre_fallback", blocked=False,
                metadata={"oos_hits": oos_hits, "in_scope_hits": in_scope_hits},
            )
            activations.append(act)
            return {"route": "fallback", "activations": activations, "blocked": False}

        hnw_hits = sum(1 for p in _HNW_PAT if p.search(query))
        has_monetary = bool(_MONETARY_PAT.search(query))

        # Informational-only queries that mention a regulatory concept but
        # carry NO client monetary value are single_agent (simple lookup).
        # Examples: "What is the MPAA limit?", "What is the DB threshold?",
        #           "What are the carry-forward rules?", "Explain UFPLS taxation."
        is_informational = _INFORMATIONAL_PAT.search(query) and not has_monetary

        if hnw_hits >= 1 and has_monetary:
            # Any HNW concept + client £ amount → full HNW pipeline
            act = self._act(
                qid, "input", "hnw_escalation",
                trigger=f"{hnw_hits} HNW pattern(s), monetary={has_monetary}",
                action="route_crewai_hnw", blocked=False,
                metadata={"hnw_hits": hnw_hits, "has_monetary": has_monetary},
            )
            activations.append(act)
            return {"route": "crewai_hnw", "activations": activations, "blocked": False}

        if hnw_hits >= 1 and not is_informational:
            # Single HNW concept without £ but NOT a pure lookup → crewai_standard
            act = self._act(
                qid, "input", "hnw_moderate",
                trigger=f"{hnw_hits} HNW pattern(s), monetary={has_monetary}",
                action="route_crewai_standard", blocked=False,
                metadata={"hnw_hits": hnw_hits, "has_monetary": has_monetary},
            )
            activations.append(act)
            return {"route": "crewai_standard", "activations": activations, "blocked": False}

        if has_monetary and not hnw_hits:
            # Pure monetary query (fee calculation, CHAPS, ISA amount) → crewai_standard
            act = self._act(
                qid, "input", "monetary_query",
                trigger=f"monetary=True, hnw_hits=0",
                action="route_crewai_standard", blocked=False,
                metadata={"has_monetary": True},
            )
            activations.append(act)
            return {"route": "crewai_standard", "activations": activations, "blocked": False}

        return {"route": "single_agent", "activations": activations, "blocked": False}

    # Retrieval rails


    def check_retrieval(
        self,
        results: "List[RetrievalResult]",
        qid: str,
    ) -> Dict[str, Any]:
        """
        Retrieval rail checkpoint.

        1. RRF confidence gate: if max_rrf < threshold → contact centre fallback
        2. Document freshness: if top chunks > 90 days old → staleness warning
        """
        activations: List[RailActivation] = []
        warnings: List[str] = []
        fallback = False

        if not results:
            act = self._act(
                qid, "retrieval", "no_results",
                trigger="0 results returned", action="contact_centre_fallback", blocked=False,
            )
            activations.append(act)
            return {"fallback": True, "warnings": [], "activations": activations}

        max_rrf = max(r.rrf_score for r in results)

        threshold = self.cfg.rrf_contact_centre_threshold
        if max_rrf < threshold:
            act = self._act(
                qid, "retrieval", "rrf_confidence_gate",
                trigger=f"max_rrf={max_rrf:.4f} < threshold={threshold}",
                action="contact_centre_fallback", blocked=False,
                metadata={"max_rrf": max_rrf, "threshold": threshold},
            )
            activations.append(act)
            fallback = True

        from datetime import datetime as _dt
        now = _dt.now(timezone.utc)
        for r in results[:3]:
            try:
                ingested = _dt.fromisoformat(r.chunk.ingestion_ts)
                if ingested.tzinfo is None:
                    from datetime import timezone as _tz
                    ingested = ingested.replace(tzinfo=_tz.utc)
                age_days = (now - ingested).days
                if age_days > self.cfg.doc_freshness_threshold_days:
                    warning = (
                        f"Document '{r.chunk.source_file}' is {age_days} days old "
                        f"(threshold: {self.cfg.doc_freshness_threshold_days} days). "
                        f"Values may be stale — verify against current Quilter documentation."
                    )
                    warnings.append(warning)
                    act = self._act(
                        qid, "retrieval", "freshness_check",
                        trigger=f"{r.chunk.source_file} age={age_days}d",
                        action="append_staleness_warning", blocked=False,
                        metadata={"source_file": r.chunk.source_file, "doc_age_days": age_days},
                    )
                    activations.append(act)
                    break  # One warning is enough
            except Exception:
                pass

        return {"fallback": fallback, "warnings": warnings, "activations": activations}

    # Output rails


    def check_output(
        self,
        answer: str,
        qid: str,
        faith_score: float,
        is_hnw: bool = False,
    ) -> Dict[str, Any]:
        """
        Output rail checkpoint.

        1. Citation check: [Source:] tag must be present in all answers
        2. Precision rail (HNW only): vague monetary language blocked
        3. Faithfulness gate: NLI score < threshold → review queue
        """
        activations: List[RailActivation] = []
        flags: List[str] = []
        review_needed = False

        if "[Source:" not in answer:
            act = self._act(
                qid, "output", "citation_check",
                trigger="no [Source:] tag found in answer",
                action="flag_missing_citation", blocked=False,
                metadata={"is_hnw": is_hnw},
            )
            activations.append(act)
            flags.append("missing_citation")

        if is_hnw:
            for pat in _MON_PAT:
                m = pat.search(answer)
                if m:
                    act = self._act(
                        qid, "output", "precision_enforcement",
                        trigger=m.group(0),
                        action="flag_vague_monetary", blocked=False,
                        metadata={"is_hnw": True, "pattern": pat.pattern},
                    )
                    activations.append(act)
                    flags.append("vague_monetary")
                    break  # One flag per answer

        threshold = self.cfg.nli_faithfulness_threshold
        if faith_score < threshold:
            act = self._act(
                qid, "output", "faithfulness_gate",
                trigger=f"faithfulness={faith_score:.3f} < threshold={threshold}",
                action="route_human_review", blocked=False,
                metadata={"faith_score": faith_score, "threshold": threshold},
            )
            activations.append(act)
            flags.append("low_faithfulness")
            review_needed = True

        return {
            "flags":         flags,
            "review_needed": review_needed,
            "activations":   activations,
        }

    # Monitoring summary


    def summary(self) -> Dict[str, Any]:
        """Aggregate counts by rail name and route type from session log."""
        by_rail: Dict[str, int] = {}
        blocked_count = 0
        for act in self._log:
            by_rail[act.rail_name] = by_rail.get(act.rail_name, 0) + 1
            if act.was_blocked:
                blocked_count += 1
        return {
            "total_activations": len(self._log),
            "blocked":           blocked_count,
            "by_rail":           by_rail,
        }
