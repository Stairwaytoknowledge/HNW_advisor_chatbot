"""
src/thresholds_store.py — Quilter HNW Advisor Assistant

Extracts regulatory threshold values from indexed Chunk objects
at index-build time and stores them in thresholds.json.
HNWPrecisionEngine reads from this store instead of hard-coded class constants.

If a value cannot be extracted from documents, the dataclass default (correct
as of 2024/25 tax year) is used as the fallback, with a warning logged.
"""

from __future__ import annotations

import json
import re
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from src.models import Chunk

logger = logging.getLogger(__name__)


# Regex patterns for threshold extraction


# Each pattern captures a numeric group (£ amount or percentage)
THRESHOLD_PATTERNS: Dict[str, re.Pattern] = {
    "db_threshold": re.compile(
        r'(?:defined\s+benefit|db)\s+(?:pension\s+)?transfer\s+value[^£\n]{0,60}£\s*([\d,]+)',
        re.I,
    ),
    "mpaa": re.compile(
        r'money\s+purchase\s+annual\s+allowance[^£\n]{0,60}£\s*([\d,]+)',
        re.I,
    ),
    "mpaa_short": re.compile(
        r'\bmpaa\b[^£\n]{0,40}£\s*([\d,]+)',
        re.I,
    ),
    "annual_allowance": re.compile(
        r'(?:pension\s+)?annual\s+allowance[^£:\n]{0,50}£\s*([\d,]+)',
        re.I,
    ),
    "chaps_fee": re.compile(
        r'chaps[^£\n]{0,40}(?:fee|charge)[^£\n]{0,30}£\s*([\d,]+)',
        re.I,
    ),
    "chaps_threshold": re.compile(
        r'chaps[^£\n]{0,60}(?:amounts?\s+(?:of\s+)?(?:over|exceeding|above)|over)\s+£\s*([\d,]+)',
        re.I,
    ),
    "adviser_init_cap_pct": re.compile(
        r'(?:initial|adviser)\s+(?:service\s+)?(?:fee|charge)[^%\n]{0,60}capped?\s+(?:at\s+)?([\d.]+)\s*%',
        re.I,
    ),
    "quarterly_minimum": re.compile(
        r'minimum\s+quarterly\s+(?:platform\s+)?(?:charge|fee)[^£\n]{0,30}£\s*([\d,]+)',
        re.I,
    ),
}

# Fee tier: "X% on/for/up to the first/next £Y" — captures (rate_pct, amount)
FEE_TIER_PATTERN = re.compile(
    r'([\d.]+)\s*%[^%\n]{0,80}(?:first|next|on|up\s+to)[^£]{0,20}£\s*([\d,]+)',
    re.I,
)

# Residual tier: "X% on/of the remainder / balance / remaining"
FEE_RESIDUAL_PATTERN = re.compile(
    r'([\d.]+)\s*%[^%\n]{0,80}(?:remainder|remaining|balance|thereafter|above)',
    re.I,
)


def _parse_gbp(raw: str) -> Optional[float]:
    """Convert '£30,000' or '30000' string to float. Returns None on failure."""
    try:
        return float(raw.replace(",", "").strip())
    except (ValueError, AttributeError):
        return None


def _parse_pct(raw: str) -> Optional[float]:
    """Convert percentage string '0.30' → 0.003, or '30' → 0.30 (cap check)."""
    try:
        v = float(raw.strip())
        # Values < 1 are already in decimal form; > 1 are in percentage form
        return v / 100 if v >= 1 else v
    except (ValueError, AttributeError):
        return None


# ThresholdsStore dataclass


@dataclass
class ThresholdsStore:
    """
    Regulatory threshold values extracted from document corpus.
    Defaults are correct for 2024/25 tax year (used as fallback only).
    """
    # Core regulatory thresholds
    db_threshold:      float = 30_000.0   # FCA COBS 19 mandatory advice threshold
    mpaa:              float = 10_000.0   # Money Purchase Annual Allowance (2024/25)
    annual_allowance:  float = 60_000.0   # Pension Annual Allowance (2024/25)
    chaps_fee:         float = 25.0       # Quilter CHAPS payment fee
    chaps_threshold:   float = 10_000.0   # Threshold above which CHAPS applies
    adviser_init_cap:  float = 0.05       # 5% adviser initial charge cap
    quarterly_minimum: float = 12.0       # Minimum quarterly platform charge

    # Tiered fee schedule: [(band_size, rate_decimal, label), ...]
    # band_size=inf for the residual tier
    fee_tiers: List[Tuple[float, float, str]] = field(default_factory=lambda: [
        (250_000.0, 0.0030, "First £250,000 at 0.30% p.a."),
        (250_000.0, 0.0025, "Next £250,000 at 0.25% p.a."),
        (float("inf"), 0.0020, "Remainder at 0.20% p.a."),
    ])

    # Source traceability: {field_name: chunk_id}
    source_chunks: Dict[str, str] = field(default_factory=dict)

    # Extraction metadata
    extracted_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    extraction_warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        d = asdict(self)
        # Convert inf to string for JSON serialisation
        d["fee_tiers"] = [
            (str(b) if b == float("inf") else b, r, lbl)
            for b, r, lbl in self.fee_tiers
        ]
        return d

    @classmethod
    def from_dict(cls, d: Dict) -> "ThresholdsStore":
        tiers_raw = d.pop("fee_tiers", [])
        tiers: List[Tuple[float, float, str]] = []
        for row in tiers_raw:
            b, r, lbl = row
            tiers.append((float("inf") if str(b) == "inf" else float(b), float(r), lbl))
        store = cls(**{k: v for k, v in d.items()
                       if k in cls.__dataclass_fields__})  # type: ignore[attr-defined]
        store.fee_tiers = tiers
        return store


# Extraction logic


def extract_thresholds_from_chunks(chunks: "List[Chunk]") -> ThresholdsStore:
    """
    Scan all Chunk objects for regulatory values using THRESHOLD_PATTERNS.

    For each pattern match:
      - Parse captured group to float
      - Sanity-check value (plausibility bounds)
      - Override store default only if value is plausible
      - Record source_chunk_id for traceability

    Fee tiers are extracted separately using FEE_TIER_PATTERN.

    RISK-01: This is the sole mechanism for populating regulatory constants.
    HNWPrecisionEngine receives a ThresholdsStore, not hardcoded literals.
    """
    store = ThresholdsStore()
    warnings: List[str] = []

    for chunk in chunks:
        text = chunk.text + " " + chunk.parent_context
        chunk_id = chunk.chunk_id

        # DB transfer threshold
        for pat_name in ("db_threshold",):
            m = THRESHOLD_PATTERNS[pat_name].search(text)
            if m:
                v = _parse_gbp(m.group(1))
                if v and 20_000 <= v <= 100_000:  # sanity bounds
                    store.db_threshold = v
                    store.source_chunks["db_threshold"] = chunk_id
                    break

        # MPAA
        for pat_name in ("mpaa", "mpaa_short"):
            m = THRESHOLD_PATTERNS[pat_name].search(text)
            if m:
                v = _parse_gbp(m.group(1))
                if v and 5_000 <= v <= 30_000:
                    store.mpaa = v
                    store.source_chunks["mpaa"] = chunk_id
                    break

        # Annual allowance
        m = THRESHOLD_PATTERNS["annual_allowance"].search(text)
        if m:
            v = _parse_gbp(m.group(1))
            if v and 20_000 <= v <= 120_000:
                store.annual_allowance = v
                store.source_chunks["annual_allowance"] = chunk_id

        # CHAPS fee
        m = THRESHOLD_PATTERNS["chaps_fee"].search(text)
        if m:
            v = _parse_gbp(m.group(1))
            if v and 5 <= v <= 100:
                store.chaps_fee = v
                store.source_chunks["chaps_fee"] = chunk_id

        # CHAPS threshold
        m = THRESHOLD_PATTERNS["chaps_threshold"].search(text)
        if m:
            v = _parse_gbp(m.group(1))
            if v and 1_000 <= v <= 50_000:
                store.chaps_threshold = v
                store.source_chunks["chaps_threshold"] = chunk_id

        # Adviser initial charge cap
        m = THRESHOLD_PATTERNS["adviser_init_cap_pct"].search(text)
        if m:
            v = _parse_pct(m.group(1))
            if v and 0.01 <= v <= 0.10:
                store.adviser_init_cap = v
                store.source_chunks["adviser_init_cap"] = chunk_id

        # Quarterly minimum
        m = THRESHOLD_PATTERNS["quarterly_minimum"].search(text)
        if m:
            v = _parse_gbp(m.group(1))
            if v and 1 <= v <= 100:
                store.quarterly_minimum = v
                store.source_chunks["quarterly_minimum"] = chunk_id

    tier_candidates: List[Tuple[float, float, str]] = []  # (band_size, rate, label)

    for chunk in chunks:
        text = chunk.text + " " + chunk.parent_context

        for m in FEE_TIER_PATTERN.finditer(text):
            rate = _parse_pct(m.group(1))
            band = _parse_gbp(m.group(2))
            if rate and band and 0.0001 <= rate <= 0.05 and band >= 10_000:
                label = f"{m.group(1)}% on next £{m.group(2)}"
                tier_candidates.append((band, rate, label))

        # Residual tier
        for m in FEE_RESIDUAL_PATTERN.finditer(text):
            rate = _parse_pct(m.group(1))
            if rate and 0.0001 <= rate <= 0.05:
                # Only add if we have at least one band tier already
                if tier_candidates:
                    tier_candidates.append((float("inf"), rate, f"{m.group(1)}% on remainder"))

    # Deduplicate tiers: keep unique (band_size, rate) pairs, sort by band
    seen: set = set()
    unique_tiers: List[Tuple[float, float, str]] = []
    for band, rate, label in sorted(tier_candidates, key=lambda x: (x[0], x[1])):
        key = (band, round(rate, 5))
        if key not in seen:
            seen.add(key)
            unique_tiers.append((band, rate, label))

    if len(unique_tiers) >= 2:
        # Ensure exactly one residual tier (inf) at the end
        non_inf = [(b, r, l) for b, r, l in unique_tiers if b != float("inf")]
        inf_tiers = [(b, r, l) for b, r, l in unique_tiers if b == float("inf")]
        if non_inf:
            store.fee_tiers = non_inf + (inf_tiers[:1] if inf_tiers else [
                (float("inf"), 0.0020, "Remainder at 0.20% p.a.")
            ])
            store.source_chunks["fee_tiers"] = "extracted"
    else:
        warnings.append(
            "Fee tiers could not be extracted from documents — using 2024/25 defaults "
            "(0.30%/0.25%/0.20%). Verify against quilter_charges.pdf."
        )

    defaults_used = [
        f for f in ("db_threshold", "mpaa", "annual_allowance",
                     "chaps_fee", "chaps_threshold")
        if f not in store.source_chunks
    ]
    if defaults_used:
        warnings.append(
            f"Fields using 2024/25 regulatory defaults (not found in documents): "
            f"{defaults_used}. Update documents to extract live values."
        )

    store.extraction_warnings = warnings
    if warnings:
        for w in warnings:
            logger.warning("[ThresholdsStore] %s", w)

    return store


# Persistence


def save_thresholds(store: ThresholdsStore, path: Path) -> None:
    """Write ThresholdsStore as JSON. Path is cfg.index_path(cfg.index_thresholds)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(store.to_dict(), indent=2), encoding="utf-8")
    tmp.replace(path)
    logger.info("ThresholdsStore saved to %s", path)


def load_thresholds(path: Path) -> ThresholdsStore:
    """
    Load ThresholdsStore from JSON.
    Returns default ThresholdsStore with a warning if file is missing.
    """
    if not path.exists():
        logger.warning(
            "Thresholds file not found at %s — using 2024/25 regulatory defaults. "
            "Run ingest to extract values from documents.", path
        )
        return ThresholdsStore()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        store = ThresholdsStore.from_dict(data)
        logger.info(
            "ThresholdsStore loaded: db=£%.0f mpaa=£%.0f aa=£%.0f tiers=%d",
            store.db_threshold, store.mpaa, store.annual_allowance, len(store.fee_tiers),
        )
        return store
    except Exception as exc:
        logger.error("Failed to load thresholds file: %s — using defaults", exc)
        return ThresholdsStore()
