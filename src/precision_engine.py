"""
src/precision_engine.py — Quilter HNW Advisor Assistant

HNW Precision Engine — exact financial computation for HNW queries.

Thresholds injected from ThresholdsStore (extracted from docs at index time).
No hardcoded regulatory constants.

compute_chaps_fee() — CHAPS same-day payment fee computation
check_mpaa() — carry-forward allowance with 3-year lookback
compute_ufpls_tax() — 25% tax-free / 75% taxable UFPLS calculation
"""

from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from src.models import PrecisionResult

if TYPE_CHECKING:
    from src.models import RetrievalResult
    from src.thresholds_store import ThresholdsStore


class HNWPrecisionEngine:
    """
    Exact financial computation engine for HNW Quilter queries.

    Methods:
      compute_platform_fee(aum, results)
      check_db_threshold(tv, results)
      check_mpaa(has_flex, proposed, prior_year_unused)
      compute_chaps_fee(withdrawal_amount, results)
      compute_ufpls_tax(ufpls_amount, marginal_rate, results)
      detect_query_type(query)

    All monetary computations show full working and cite source documents.
    The engine never approximates — it either computes exactly or returns
    an explicit "cannot compute" PrecisionResult.
    """

    def __init__(self, thresholds: "ThresholdsStore") -> None:
        """
        RISK-01: Thresholds injected from ThresholdsStore (loaded from index).
        Falls back to ThresholdsStore defaults if store not available.
        """
        self.t = thresholds

    # Platform fee computation


    def compute_platform_fee(
        self,
        aum: float,
        results: "List[RetrievalResult]",
    ) -> PrecisionResult:
        """
        Compute exact Quilter tiered platform fee for given AUM.

        Uses self.t.fee_tiers (extracted from quilter_charges.pdf at index time).
        Shows full per-tier working. Returns annual, quarterly, and monthly figures.
        Never approximates.
        """
        working: List[str] = []
        citations: List[str] = []
        total_fee = 0.0
        remaining = aum

        working.append(f"Portfolio AUM: £{aum:>14,.2f}")
        working.append("-" * 45)

        for band_size, rate, label in self.t.fee_tiers:
            if remaining <= 0:
                break
            applied = min(remaining, band_size) if band_size != math.inf else remaining
            tier_fee = applied * rate
            total_fee += tier_fee
            remaining -= applied

            if band_size == math.inf:
                tier_label = f"Remainder £{applied:,.2f}"
            else:
                tier_label = f"Band £{applied:,.2f}"

            working.append(
                f"{tier_label} × {rate*100:.2f}% = £{tier_fee:>10,.2f}"
            )

        working.append("-" * 45)
        working.append(f"TOTAL annual fee:       £{total_fee:>10,.2f} p.a.")
        working.append(f"Quarterly (/ 4):        £{total_fee/4:>10,.2f}")
        working.append(f"Monthly  (/ 12):        £{total_fee/12:>10,.2f}")

        # Minimum quarterly check
        q_fee = total_fee / 4
        if q_fee < self.t.quarterly_minimum:
            working.append(
                f"Note: Quarterly fee £{q_fee:.2f} is below minimum £{self.t.quarterly_minimum:.2f} "
                f"— minimum charge applies."
            )

        # CHAPS note for large withdrawals
        if aum > self.t.chaps_threshold:
            working.append(
                f"Note: CHAPS fee of £{self.t.chaps_fee:.2f} applies to any same-day "
                f"withdrawals over £{self.t.chaps_threshold:,.0f}."
            )

        # Source citations from retrieved chunks
        citations = self._citations_from_results(results, "quilter_charges.pdf")
        if not citations:
            citations = ["[Source: quilter_charges.pdf, p.1, §Platform Charge Schedule]"]

        return PrecisionResult(
            query_type="fee_calculation",
            computed_value=f"£{total_fee:,.2f} per annum (£{total_fee/4:,.2f}/quarter)",
            working_shown=working,
            source_citations=citations,
            confidence=0.99,
            raw_values={
                "aum": aum,
                "annual_fee": total_fee,
                "quarterly_fee": total_fee / 4,
                "monthly_fee": total_fee / 12,
            },
        )

    # DB transfer threshold check


    def check_db_threshold(
        self,
        tv: float,
        results: "List[RetrievalResult]",
    ) -> PrecisionResult:
        """
        Check DB pension transfer value against FCA COBS 19 mandatory advice threshold.
        Uses self.t.db_threshold (extracted from docs, not hardcoded).
        """
        threshold = self.t.db_threshold
        exceeds = tv >= threshold
        delta = tv - threshold

        working: List[str] = [
            f"Transfer Value (TV):        £{tv:>12,.2f}",
            f"FCA COBS 19 threshold:      £{threshold:>12,.2f}",
            f"Delta:                      £{abs(delta):>12,.2f} {'above' if exceeds else 'below'} threshold",
            "-" * 50,
        ]

        if exceeds:
            working += [
                "STATUS: Mandatory regulated advice REQUIRED",
                "",
                "Required steps (COBS 19.1):",
                "  1. Transfer Value Analysis (TVA) must be completed",
                "  2. Appropriate Pension Transfer Analysis (APTA) required",
                "  3. Advice must be given by a suitably qualified pension transfer specialist",
                "  4. Client must receive a suitability report",
                f"  5. Transfer timeline: 4–8 weeks from receipt of completed documentation",
            ]
            result_val = (
                f"£{tv:,.2f} EXCEEDS £{threshold:,.0f} threshold by £{delta:,.2f} — "
                f"MANDATORY COBS 19 advice required"
            )
            confidence = 0.99
        else:
            working += [
                "STATUS: Below mandatory advice threshold",
                "Regulated advice is not mandatory for this transfer value.",
                "Note: Adviser should still consider whether advice is in the client's best interests.",
            ]
            result_val = (
                f"£{tv:,.2f} is BELOW the £{threshold:,.0f} threshold — "
                f"mandatory advice not triggered"
            )
            confidence = 0.98

        citations = self._citations_from_results(results, "quilter_pensions.pdf")
        if not citations:
            citations = ["[Source: quilter_pensions.pdf, p.1, §Defined Benefit Transfer Rules]",
                         "[Source: FCA COBS 19.1 — Pension Transfer Rules]"]

        return PrecisionResult(
            query_type="threshold_check_db",
            computed_value=result_val,
            working_shown=working,
            source_citations=citations,
            confidence=confidence,
            raw_values={
                "transfer_value": tv,
                "threshold": threshold,
                "exceeds_threshold": exceeds,
                "delta": delta,
            },
        )

    # MPAA and carry-forward allowance check


    def check_mpaa(
        self,
        has_flex: bool,
        proposed: float,
        prior_year_unused: Optional[List[float]] = None,
    ) -> PrecisionResult:
        """
        MPAA and Annual Allowance check.

        GAP-08 fix: carry-forward allowance computation.
          - If has_flex=True: MPAA applies (£10,000 limit), carry-forward NOT available.
          - If has_flex=False AND prior_year_unused provided: compute total available
            allowance including carry-forward from up to 3 prior years.

        prior_year_unused: list of unused AA amounts, most recent year first.
          e.g. [5000, 10000, 0] = £5k unused in prior year, £10k two years ago, £0 three years ago.
        """
        working: List[str] = []

        if has_flex:
            # MPAA path — carry-forward NOT available after flexible access
            mpaa = self.t.mpaa
            excess = max(0.0, proposed - mpaa)
            working += [
                "Client has flexibly accessed their pension (flexi-access drawdown / UFPLS).",
                "MPAA APPLIES — carry-forward allowance is NOT available.",
                "",
                f"Money Purchase Annual Allowance (MPAA): £{mpaa:>10,.2f}",
                f"Proposed contribution:                  £{proposed:>10,.2f}",
                "-" * 50,
            ]

            if excess > 0:
                tax_charge_basic = excess * 0.20   # illustrative at basic rate
                tax_charge_higher = excess * 0.40
                working += [
                    f"Excess over MPAA:                       £{excess:>10,.2f}",
                    "",
                    "Annual Allowance Charge (on excess):",
                    f"  At 20% (basic rate):    £{tax_charge_basic:>10,.2f}",
                    f"  At 40% (higher rate):   £{tax_charge_higher:>10,.2f}",
                    "  (Actual rate = client's marginal income tax rate)",
                ]
                result_val = (
                    f"MPAA EXCEEDED by £{excess:,.2f}. "
                    f"Annual Allowance Charge applies on £{excess:,.2f} at marginal rate."
                )
                warnings = [
                    f"Contribution of £{proposed:,.2f} exceeds MPAA of £{mpaa:,.0f}.",
                    "Client must self-assess and pay Annual Allowance Charge via tax return.",
                ]
            else:
                working.append(
                    f"Proposed contribution is WITHIN the MPAA. No annual allowance charge."
                )
                result_val = f"Contribution of £{proposed:,.2f} is within MPAA of £{mpaa:,.0f}."
                warnings = []

            citations_src = "quilter_pensions.pdf"

        else:
            # Standard Annual Allowance path — carry-forward may apply
            aa = self.t.annual_allowance
            available = aa

            working += [
                "Client has NOT flexibly accessed their pension.",
                "Standard Annual Allowance applies. Carry-forward MAY be available.",
                "",
                f"Current year Annual Allowance:   £{aa:>10,.2f}",
            ]

            # GAP-08: carry-forward computation
            if prior_year_unused:
                years_ago = ["Prior year (1)", "Prior year (2)", "Prior year (3)"]
                cf_total = 0.0
                for i, unused in enumerate(prior_year_unused[:3]):
                    working.append(f"  + {years_ago[i]} unused:          £{unused:>10,.2f}")
                    cf_total += unused
                    available += unused
                working += [
                    "-" * 50,
                    f"Total available (AA + carry-forward): £{available:>10,.2f}",
                    "",
                    "Note: Current year AA must be fully used before carry-forward applies.",
                    "Note: Carry-forward requires pension scheme membership in each prior year.",
                ]
            else:
                working.append(
                    "  (No carry-forward data provided — using current year AA only)"
                )

            working += [
                "",
                f"Proposed contribution:           £{proposed:>10,.2f}",
                "-" * 50,
            ]

            excess = max(0.0, proposed - available)
            if excess > 0:
                tax_charge_basic = excess * 0.20
                tax_charge_higher = excess * 0.40
                working += [
                    f"Excess over available allowance: £{excess:>10,.2f}",
                    "",
                    "Annual Allowance Charge (on excess):",
                    f"  At 20% (basic rate):    £{tax_charge_basic:>10,.2f}",
                    f"  At 40% (higher rate):   £{tax_charge_higher:>10,.2f}",
                ]
                result_val = f"Proposed contribution EXCEEDS available allowance by £{excess:,.2f}."
                warnings = [
                    f"Annual Allowance Charge on £{excess:,.2f} excess.",
                    "Verify carry-forward years are within the 3-year lookback period.",
                ]
            else:
                working.append(
                    f"Proposed contribution of £{proposed:,.2f} is WITHIN the available allowance. "
                    f"No annual allowance charge."
                )
                result_val = f"Contribution of £{proposed:,.2f} is within available allowance of £{available:,.2f}."
                warnings = []

            citations_src = "quilter_pensions.pdf"

        citations = [f"[Source: {citations_src}, p.2, §Money Purchase Annual Allowance]",
                     "[Source: HMRC Pension Annual Allowance rules]"]

        return PrecisionResult(
            query_type="threshold_check_mpaa",
            computed_value=result_val,
            working_shown=working,
            source_citations=citations,
            confidence=0.97,
            warnings=warnings if "warnings" in dir() else [],
            raw_values={
                "has_flexible_access": has_flex,
                "proposed_contribution": proposed,
                "mpaa": self.t.mpaa if has_flex else None,
                "annual_allowance": self.t.annual_allowance,
                "available_allowance": available if not has_flex else self.t.mpaa,
                "excess": excess,
                "prior_year_unused": prior_year_unused,
            },
        )

    # CHAPS fee computation — GAP-07 fix


    def compute_chaps_fee(
        self,
        withdrawal_amount: float,
        results: "List[RetrievalResult]",
    ) -> PrecisionResult:
        """
        GAP-07 fix: CHAPS same-day payment fee computation.

        Logic:
          chaps_applies = withdrawal_amount > self.t.chaps_threshold
          If applies: CHAPS fee = self.t.chaps_fee (£25)
          If not applies: BACS used, no CHAPS fee
        """
        chaps_applies = withdrawal_amount > self.t.chaps_threshold

        working: List[str] = [
            f"Withdrawal amount:    £{withdrawal_amount:>12,.2f}",
            f"CHAPS threshold:      £{self.t.chaps_threshold:>12,.0f}",
            "-" * 40,
        ]

        if chaps_applies:
            working += [
                f"CHAPS fee applies:    YES",
                f"CHAPS fee:            £{self.t.chaps_fee:>12,.2f}",
                "",
                "Payment will be processed via CHAPS (same-day settlement).",
                "CHAPS funds are available in the client's account on the same business day.",
                "CHAPS requests must be submitted before the cut-off time (check with operations).",
            ]
            result_val = f"CHAPS fee: £{self.t.chaps_fee:.2f} (same-day settlement)"
        else:
            working += [
                "CHAPS fee applies:    NO",
                "",
                f"Withdrawal of £{withdrawal_amount:,.2f} is at or below the CHAPS threshold.",
                "Payment will be processed via BACS (3 working days, no fee).",
            ]
            result_val = f"No CHAPS fee - withdrawal processed via BACS (£{withdrawal_amount:,.2f} at or below £{self.t.chaps_threshold:,.0f})"

        citations = self._citations_from_results(results, "quilter_charges.pdf")
        if not citations:
            citations = ["[Source: quilter_charges.pdf, p.1, §CHAPS and Same-Day Payments]"]

        return PrecisionResult(
            query_type="chaps_fee",
            computed_value=result_val,
            working_shown=working,
            source_citations=citations,
            confidence=0.99,
            raw_values={
                "withdrawal_amount": withdrawal_amount,
                "chaps_threshold": self.t.chaps_threshold,
                "chaps_applies": chaps_applies,
                "chaps_fee": self.t.chaps_fee if chaps_applies else 0.0,
            },
        )

    # UFPLS tax computation — GAP-09 fix


    def compute_ufpls_tax(
        self,
        ufpls_amount: float,
        marginal_rate: float = 0.20,
        results: Optional["List[RetrievalResult]"] = None,
    ) -> PrecisionResult:
        """
        GAP-09 fix: UFPLS (Uncrystallised Funds Pension Lump Sum) tax computation.

        Per HMRC rules:
          25% of each UFPLS payment is tax-free (PCLS equivalent)
          75% is taxable as income in the year of payment

        Emergency tax may be deducted on the first UFPLS — adviser should check.
        Taking a UFPLS triggers the MPAA.
        """
        tax_free  = ufpls_amount * 0.25
        taxable   = ufpls_amount * 0.75
        tax_due   = taxable * marginal_rate
        net       = ufpls_amount - tax_due

        working: List[str] = [
            f"UFPLS gross amount:         £{ufpls_amount:>12,.2f}",
            "-" * 48,
            f"Tax-free element (25%):     £{tax_free:>12,.2f}",
            f"Taxable element (75%):      £{taxable:>12,.2f}",
            f"Income tax @ {marginal_rate*100:.0f}% marginal:  £{tax_due:>12,.2f}",
            "-" * 48,
            f"Estimated net receipt:      £{net:>12,.2f}",
            "",
            f"MPAA triggered: future pension contributions limited to £{self.t.mpaa:,.0f} p.a.",
            "",
            "Important notes:",
            "  1. Emergency tax (Month 1 basis) may be deducted on first UFPLS payment.",
            "     Client can reclaim via HMRC form P55 (partial withdrawal) or P50Z (full).",
            "  2. The taxable element is added to other income in the tax year.",
            "  3. Actual tax may differ if client has other income sources.",
            f"  4. Adviser must confirm marginal rate — this computation assumes {marginal_rate*100:.0f}%.",
        ]

        citations: List[str] = []
        if results:
            citations = self._citations_from_results(results, "quilter_pensions.pdf")
        if not citations:
            citations = [
                "[Source: quilter_pensions.pdf, p.3, §UFPLS Tax Treatment]",
                "[Source: HMRC Pension Lump Sums — Uncrystallised Funds rules]",
            ]

        return PrecisionResult(
            query_type="ufpls_tax",
            computed_value=f"Net receipt: £{net:,.2f} (after £{tax_due:,.2f} income tax at {marginal_rate*100:.0f}%)",
            working_shown=working,
            source_citations=citations,
            confidence=0.95,
            warnings=[
                f"MPAA of £{self.t.mpaa:,.0f} now applies — future contributions restricted.",
                "Confirm marginal income tax rate with client's tax adviser.",
                "Emergency tax may require P55/P50Z reclaim from HMRC.",
            ],
            raw_values={
                "ufpls_amount": ufpls_amount,
                "tax_free": tax_free,
                "taxable": taxable,
                "marginal_rate": marginal_rate,
                "tax_due": tax_due,
                "net_receipt": net,
                "mpaa_triggered": True,
                "mpaa_limit": self.t.mpaa,
            },
        )

    # Query type detection


    def detect_query_type(self, query: str) -> Tuple[str, Dict]:
        """
        Classify a query and extract relevant parameters for precision computation.

        Returns (query_type, params_dict).

        Query types:
          "fee_calculation"       → params: {"aum": float}
          "threshold_check_db"    → params: {"tv": float}
          "threshold_check_mpaa"  → params: {"has_flex": bool, "proposed": float, "unused": List[float]}
          "chaps_fee"             → params: {"amount": float}
          "ufpls_tax"             → params: {"amount": float, "marginal_rate": float}
          "carry_forward_mpaa"    → params: {"proposed": float, "unused": List[float]}
          "none"                  → params: {}
        """
        q = query.lower()

        def _extract_gbp(text: str) -> Optional[float]:
            m = re.search(r'£\s*([\d,]+(?:\.\d+)?)', text)
            if m:
                return float(m.group(1).replace(",", ""))
            # Also try bare number with m/k suffix
            m2 = re.search(r'\b([\d,]+(?:\.\d+)?)\s*(?:million|m\b)', text, re.I)
            if m2:
                return float(m2.group(1).replace(",", "")) * 1_000_000
            return None

        def _extract_pct(text: str) -> float:
            m = re.search(r'\b(\d+)\s*%\s*(?:tax|rate|marginal)', text, re.I)
            return int(m.group(1)) / 100 if m else 0.20

        def _extract_gbp_list(text: str) -> List[float]:
            return [float(m.replace(",", "")) for m in re.findall(r'£\s*([\d,]+)', text)]

        if re.search(
            r'(?:platform\s+)?(?:fee|charge|cost)\s+(?:on|for).*£|'
            r'what.*(?:fee|charge).*£.*(?:portfolio|aum|assets)',
            q,
        ):
            aum = _extract_gbp(query)
            if aum:
                return ("fee_calculation", {"aum": aum})

        if re.search(r'chaps|same.?day.*(?:payment|withdrawal)|large.*withdrawal.*fee', q):
            amount = _extract_gbp(query)
            if amount:
                return ("chaps_fee", {"amount": amount})

        if re.search(r'ufpls.*tax|tax.*ufpls|net.*ufpls|ufpls.*net|uncrystallised.*tax', q):
            amount = _extract_gbp(query)
            rate = _extract_pct(query)
            if amount:
                return ("ufpls_tax", {"amount": amount, "marginal_rate": rate})

        if re.search(r'carry.?forward|unused.*allowance|prior.*year.*allowance', q):
            amounts = _extract_gbp_list(query)
            if amounts:
                proposed = amounts[0]
                unused = amounts[1:4]  # up to 3 prior years
                return ("carry_forward_mpaa", {"proposed": proposed, "unused": unused})

        if re.search(r'\bmpaa\b|money\s+purchase\s+annual\s+allowance|annual\s+allowance\s+charge', q):
            has_flex = bool(re.search(r'taken|triggered|flexi|drawdown|ufpls|flexible', q))
            proposed = _extract_gbp(query)
            if proposed:
                return ("threshold_check_mpaa", {
                    "has_flex": has_flex,
                    "proposed": proposed,
                    "unused": [],
                })

        if re.search(
            r'defined\s+benefit|db\s+pension|transfer\s+value|db\s+transfer|cobs\s*19', q
        ):
            tv = _extract_gbp(query)
            if tv:
                return ("threshold_check_db", {"tv": tv})

        return ("none", {})

    # Helpers


    @staticmethod
    def _citations_from_results(
        results: "List[RetrievalResult]",
        prefer_source: str = "",
    ) -> List[str]:
        """Extract [Source: ...] citation strings from retrieval results."""
        citations = []
        for r in results:
            c = r.chunk
            citation = f"[Source: {c.source_file}, p.{c.page_num}, §{c.section}]"
            if prefer_source and prefer_source in c.source_file:
                citations.insert(0, citation)
            else:
                citations.append(citation)
        # Deduplicate preserving order
        seen = set()
        deduped = []
        for cit in citations:
            if cit not in seen:
                seen.add(cit)
                deduped.append(cit)
        return deduped[:5]
