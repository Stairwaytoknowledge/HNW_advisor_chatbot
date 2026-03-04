"""
src/pdf_ingestion.py — Quilter HNW Advisor Assistant 

 Real PDF ingestion pipeline using pypdf (pure Python, no DLL).
pymupdf/fitz was blocked by Windows Application Control; pypdf is pure Python.
Implements sliding-window chunking at 400 tokens / 80 token overlap.
Falls back to DEMO_CORPUS if pdf_dir is empty (preserves demo mode).

Pipeline per PDF:
  1. load_pdf_pages     — pypdf extraction (pdfplumber fallback)
  2. extract_section_heading — heuristic section header detection
  3. split_into_sentences — regex sentence splitter
  4. sliding_window_chunks — 400-token windows with 80-token overlap
  5. make_chunk          — Chunk dataclass with SHA256 IDs and entity extraction
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from src.models import Chunk

logger = logging.getLogger(__name__)

try:
    from pypdf import PdfReader as _PdfReader
    PYPDF_OK = True
except ImportError:
    _PdfReader = None  # type: ignore[assignment,misc]
    PYPDF_OK = False

try:
    import pdfplumber as _pdfplumber
    PDFPLUMBER_OK = True
except ImportError:
    _pdfplumber = None  # type: ignore[assignment]
    PDFPLUMBER_OK = False

if not PYPDF_OK and not PDFPLUMBER_OK:
    logger.warning("Neither pypdf nor pdfplumber available — PDF loading unavailable")

if TYPE_CHECKING:
    from src.config import Config


# Demo corpus (fallback when no PDFs are present)
# Extracted from notebook cell 6 — preserves demo/test mode


DEMO_CORPUS: List[Dict] = [
    {
        "source": "quilter_charges.pdf", "page": 1,
        "section": "Platform Charge Schedule",
        "text": (
            "The Quilter platform charge is tiered based on total assets under management. "
            "The first £250,000 is charged at 0.30% per annum. "
            "The next £250,000 (£250,001 to £500,000) is charged at 0.25% per annum. "
            "Assets above £500,000 are charged at 0.20% per annum. "
            "A minimum quarterly charge of £12 applies. "
            "VAT is not chargeable on platform fees."
        ),
    },
    {
        "source": "quilter_charges.pdf", "page": 1,
        "section": "CHAPS and Same-Day Payments",
        "text": (
            "A CHAPS fee of £25 applies to same-day (CHAPS) payment requests. "
            "CHAPS processing is available for withdrawal amounts over £10,000. "
            "Standard withdrawals below £10,000 are processed via BACS within 3 working days."
        ),
    },
    {
        "source": "quilter_charges.pdf", "page": 2,
        "section": "Adviser Charges",
        "text": (
            "Adviser initial charges are capped at 5% of the investment amount under Consumer Duty. "
            "Ongoing adviser charges must represent fair value as assessed annually. "
            "All adviser charges must be agreed in writing with the client before application."
        ),
    },
    {
        "source": "quilter_pensions.pdf", "page": 1,
        "section": "Defined Benefit Transfer Rules",
        "text": (
            "Clients with a defined benefit (DB) pension transfer value of £30,000 or more "
            "must receive regulated advice from a suitably qualified adviser before transferring. "
            "This is a mandatory requirement under FCA COBS 19.1. "
            "A Transfer Value Analysis (TVA) and an Appropriate Pension Transfer Analysis (APTA) "
            "are both required. "
            "The transfer timeline is typically 4–8 weeks from receipt of completed paperwork."
        ),
    },
    {
        "source": "quilter_pensions.pdf", "page": 2,
        "section": "Money Purchase Annual Allowance",
        "text": (
            "Once a client has flexibly accessed their pension (e.g., by taking flexi-access "
            "drawdown or an Uncrystallised Funds Pension Lump Sum (UFPLS)), the Money Purchase "
            "Annual Allowance (MPAA) applies. "
            "The MPAA for the 2024/25 tax year is £10,000. "
            "Contributions above £10,000 to money purchase pensions in the same tax year "
            "will incur an annual allowance charge at the client's marginal income tax rate. "
            "The standard Annual Allowance of £60,000 is replaced by the MPAA for money purchase inputs."
        ),
    },
    {
        "source": "quilter_pensions.pdf", "page": 3,
        "section": "UFPLS Tax Treatment",
        "text": (
            "An Uncrystallised Funds Pension Lump Sum (UFPLS) is a flexible way to take pension "
            "benefits without entering drawdown. "
            "25% of each UFPLS payment is tax-free; the remaining 75% is taxable as income "
            "in the year of payment. "
            "Taking a UFPLS triggers the Money Purchase Annual Allowance (MPAA). "
            "Emergency tax may be deducted at source on the first UFPLS payment; reclaim via HMRC."
        ),
    },
    {
        "source": "quilter_pensions.pdf", "page": 4,
        "section": "Carry Forward Allowance",
        "text": (
            "Clients who have not triggered the MPAA may use pension annual allowance carry forward. "
            "Unused pension annual allowance from the previous 3 tax years may be carried forward "
            "provided the client was a member of a registered pension scheme in those years. "
            "Carry forward is not available once the MPAA has been triggered. "
            "The current year's annual allowance must be used in full before carry forward applies."
        ),
    },
    {
        "source": "quilter_transfers.pdf", "page": 1,
        "section": "ISA Transfer Requirements",
        "text": (
            "To complete an ISA transfer, the client's name (including spelling and middle names), "
            "date of birth, full address, and National Insurance number must exactly match the "
            "ceding scheme records. "
            "The account number for the ceding scheme must be accurate. "
            "For current year ISAs, the subscription amount must be correctly stated. "
            "Most ISA transfers complete within 20 working days via ORIGO. "
            "Use Quilter's Transfer Tracker to monitor progress."
        ),
    },
    {
        "source": "quilter_transfers.pdf", "page": 2,
        "section": "Re-registration of Assets",
        "text": (
            "Re-registration (in-specie transfer) allows assets to be moved between platforms "
            "without selling and repurchasing. "
            "Re-registration typically completes within 15–20 working days. "
            "Not all assets are eligible for re-registration; check Quilter's permitted assets list. "
            "The client must sign the re-registration authority form with a wet signature."
        ),
    },
    {
        "source": "quilter_onboarding.pdf", "page": 1,
        "section": "KYC and Client Verification",
        "text": (
            "All new clients must complete Know Your Client (KYC) verification before account opening. "
            "Acceptable ID documents include: valid passport, UK driving licence (full or provisional), "
            "or national identity card. "
            "Proof of address must be dated within 3 months: utility bill, bank statement, or "
            "HMRC correspondence. "
            "Enhanced Due Diligence (EDD) applies to Politically Exposed Persons (PEPs) and clients "
            "with assets above £1,000,000."
        ),
    },
    {
        "source": "quilter_onboarding.pdf", "page": 2,
        "section": "Politically Exposed Persons",
        "text": (
            "Politically Exposed Persons (PEPs) require enhanced due diligence under the "
            "Money Laundering Regulations 2017. "
            "Senior management approval is required before onboarding a PEP client. "
            "Ongoing monitoring of PEP accounts must be conducted at least annually. "
            "PEP status must be re-assessed whenever there is a material change in the client's "
            "circumstances or public profile."
        ),
    },
]


# Entity extraction (regulatory values and keywords)


_GBP_PAT  = re.compile(r'£\s*[\d,]+(?:\.\d+)?', re.I)
_PCT_PAT  = re.compile(r'\b\d+(?:\.\d+)?\s*%', re.I)
_DAYS_PAT = re.compile(r'\b\d+\s+(?:working\s+)?days?\b', re.I)
_YEAR_PAT = re.compile(r'\b(?:20\d{2}|\d{4}/\d{2,4})\b')

REG_KEYWORDS = [
    "mpaa", "annual allowance", "db pension", "defined benefit",
    "transfer value", "cobs 19", "ufpls", "uncrystallised",
    "carry forward", "drawdown", "flexi-access", "chaps",
    "re-registration", "re-reg", "pep", "politically exposed",
    "consumer duty", "enhanced due diligence", "edd",
    "money laundering", "mandatory advice", "tva", "apta",
    "isa transfer", "pension transfer",
]

THRESHOLD_KEYWORDS = [
    "£30,000", "£10,000", "£60,000", "£250,000", "£500,000",
    "30000", "10000", "60000", "threshold", "mandatory",
]


def extract_entities(text: str) -> Dict:
    """Extract numerical entities and regulatory keywords from chunk text."""
    return {
        "gbp_amounts":   _GBP_PAT.findall(text),
        "percentages":   _PCT_PAT.findall(text),
        "day_mentions":  _DAYS_PAT.findall(text),
        "year_mentions": _YEAR_PAT.findall(text),
    }


def detect_regulatory_keywords(text: str) -> List[str]:
    """Return list of regulatory keywords found in text."""
    tl = text.lower()
    return [kw for kw in REG_KEYWORDS if kw in tl]


def contains_threshold(text: str) -> bool:
    """True if text contains a known threshold value or keyword."""
    tl = text.lower()
    return any(kw.lower() in tl for kw in THRESHOLD_KEYWORDS)


def make_chunk(
    source: str,
    page: int,
    section: str,
    text: str,
    cfg: "Config",
    doc_version: str = "",
) -> Chunk:
    """
    Factory function for Chunk dataclass.
    Computes SHA256 IDs, extracts entities, detects regulatory content.
    doc_version (SHA256 of source PDF) passed through from manifest — RISK-01 fix.
    """
    chunk_id = hashlib.sha256(text.encode()).hexdigest()[:12]
    doc_id   = hashlib.sha256(source.encode()).hexdigest()[:8]
    parent_context = f"{section}\n\n{text}"
    tokens = len(text.split())

    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        source_file=source,
        page_num=page,
        section=section,
        parent_context=parent_context,
        text=text,
        token_count=tokens,
        doc_version=doc_version,
        ingestion_ts=datetime.now(timezone.utc).isoformat(),
        numerical_entities=extract_entities(text),
        contains_regulatory_threshold=contains_threshold(text),
        regulatory_keywords=detect_regulatory_keywords(text),
    )


# PDF loading (pymupdf)


def load_pdf_pages(pdf_path: Path) -> List[Tuple[int, str]]:
    """
    Open PDF with pypdf (pure Python, no DLL requirement).
    Falls back to pdfplumber if pypdf extracts no text (scanned PDFs).
    Returns list of (page_num, cleaned_text) — 1-indexed page numbers.
    """
    if PYPDF_OK:
        pages = _load_with_pypdf(pdf_path)
        if pages:
            logger.info("Loaded %d pages from %s (pypdf)", len(pages), pdf_path.name)
            return pages

    if PDFPLUMBER_OK:
        pages = _load_with_pdfplumber(pdf_path)
        if pages:
            logger.info("Loaded %d pages from %s (pdfplumber)", len(pages), pdf_path.name)
            return pages

    logger.error("Could not extract text from %s — no PDF backend available", pdf_path)
    return []


def _load_with_pypdf(pdf_path: Path) -> List[Tuple[int, str]]:
    """Extract text from PDF using pypdf (pure Python)."""
    try:
        reader = _PdfReader(str(pdf_path))
    except Exception as exc:
        logger.warning("pypdf failed to open %s: %s", pdf_path, exc)
        return []

    pages: List[Tuple[int, str]] = []
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
            text = _clean_pdf_text(text)
            if text:
                pages.append((i + 1, text))
        except Exception as exc:
            logger.warning("pypdf failed page %d of %s: %s", i + 1, pdf_path, exc)
    return pages


def _load_with_pdfplumber(pdf_path: Path) -> List[Tuple[int, str]]:
    """Extract text from PDF using pdfplumber (fallback for complex layouts)."""
    try:
        with _pdfplumber.open(str(pdf_path)) as pdf:
            pages: List[Tuple[int, str]] = []
            for i, page in enumerate(pdf.pages):
                try:
                    text = page.extract_text() or ""
                    text = _clean_pdf_text(text)
                    if text:
                        pages.append((i + 1, text))
                except Exception as exc:
                    logger.warning("pdfplumber failed page %d of %s: %s", i + 1, pdf_path, exc)
            return pages
    except Exception as exc:
        logger.warning("pdfplumber failed to open %s: %s", pdf_path, exc)
        return []


def _clean_pdf_text(text: str) -> str:
    """
    Clean raw PDF text:
      - Collapse runs of whitespace within lines
      - Remove lines that are only numbers (page numbers)
      - Remove lines shorter than 3 chars
      - Normalise unicode dashes/quotes
    """
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line or len(line) < 3:
            continue
        if re.fullmatch(r'\d{1,4}', line):   # bare page numbers
            continue
        line = line.replace('\u2019', "'").replace('\u2018', "'")
        line = line.replace('\u201c', '"').replace('\u201d', '"')
        line = line.replace('\u2013', '-').replace('\u2014', '-')
        line = re.sub(r' {2,}', ' ', line)
        lines.append(line)
    return "\n".join(lines).strip()


def extract_section_heading(page_text: str, page_num: int, source_file: str) -> str:
    """
    Heuristic section heading extraction from page text.

    Checks in order:
      1. ALL CAPS line of 3+ words (common in Quilter docs)
      2. Line followed by a colon
      3. Short line (< 60 chars) at start of text
      4. Fallback: "{source_file} p.{page_num}"
    """
    lines = [ln.strip() for ln in page_text.split("\n") if ln.strip()]
    if not lines:
        return f"{source_file} p.{page_num}"

    # All-caps heading
    for line in lines[:5]:
        words = line.split()
        if len(words) >= 2 and all(w.isupper() or not w.isalpha() for w in words):
            return line.title()

    # Line ending with colon
    for line in lines[:5]:
        if line.endswith(":") and 5 < len(line) < 80:
            return line[:-1]

    # Short line at top (likely a heading)
    if lines and len(lines[0]) < 60 and len(lines[0]) > 4:
        return lines[0]

    return f"{source_file} p.{page_num}"


# Sentence splitting


# Abbreviations that should NOT be treated as sentence boundaries
_ABBREVS = re.compile(
    r'\b(?:e\.g|i\.e|p\.a|p\.m|a\.m|Mr|Mrs|Ms|Dr|vs|etc|COBS|FCA|ISA|SIPP|TVA|APTA)\.',
    re.I,
)

def split_into_sentences(text: str) -> List[str]:
    """
    Regex sentence splitter that respects financial abbreviations.
    Splits on '. ', '! ', '? ', '\n\n' boundaries.
    Filters out sentences shorter than 4 words.
    """
    # Protect abbreviations
    protected = _ABBREVS.sub(lambda m: m.group(0).replace(".", "<<<DOT>>>"), text)

    # Split on sentence boundaries
    raw_sentences = re.split(r'(?<=[.!?])\s+|\n{2,}', protected)

    # Restore abbreviations and clean up
    sentences = []
    for s in raw_sentences:
        s = s.replace("<<<DOT>>>", ".").strip()
        if len(s.split()) >= 4:
            sentences.append(s)
    return sentences


# Sliding window chunker


def sliding_window_chunks(
    sentences: List[str],
    chunk_size_tokens: int = 400,
    overlap_tokens: int = 80,
) -> List[str]:
    """
    Token-aware sliding window chunker.

    Algorithm:
      - Accumulate sentences until chunk_size_tokens reached
      - Record chunk text
      - Step back by overlap_tokens worth of sentences for the next window
      - Repeat until all sentences covered

    Whitespace tokenisation (len(s.split())) for token counting.
    Ensures no chunk exceeds chunk_size_tokens (splits oversized sentences at word boundaries).
    """
    if not sentences:
        return []

    chunks: List[str] = []
    i = 0

    while i < len(sentences):
        current_tokens = 0
        window: List[str] = []

        j = i
        while j < len(sentences):
            s_tokens = len(sentences[j].split())
            if current_tokens + s_tokens > chunk_size_tokens and window:
                break
            # Handle single sentence that exceeds chunk_size_tokens
            if s_tokens > chunk_size_tokens:
                words = sentences[j].split()
                for start in range(0, len(words), chunk_size_tokens - overlap_tokens):
                    sub = " ".join(words[start:start + chunk_size_tokens])
                    if sub:
                        chunks.append(sub)
                j += 1
                i = j
                break
            window.append(sentences[j])
            current_tokens += s_tokens
            j += 1

        if window:
            chunks.append(" ".join(window))

            # Step back by overlap: find how many trailing sentences fit in overlap_tokens
            overlap_so_far = 0
            step_back = 0
            for k in range(len(window) - 1, -1, -1):
                overlap_so_far += len(window[k].split())
                if overlap_so_far >= overlap_tokens:
                    break
                step_back += 1

            i = j - step_back if step_back > 0 else j

    return chunks


# Full ingestion pipeline


def ingest_pdf(
    pdf_path: Path,
    cfg: "Config",
    doc_version: str = "",
) -> List[Chunk]:
    """
    Full ingestion pipeline for one PDF. GAP-01 fix.

    Steps:
      1. load_pdf_pages → raw page texts
      2. For each page: extract_section_heading
      3. split_into_sentences on page text
      4. sliding_window_chunks (400 tokens / 80 overlap)
      5. make_chunk for each window
      6. Return List[Chunk]

    doc_version is the SHA256 from the download manifest (RISK-01 fix).
    """
    pages = load_pdf_pages(pdf_path)
    if not pages:
        logger.warning("No text extracted from %s", pdf_path)
        return []

    chunks: List[Chunk] = []
    source_name = pdf_path.name

    for page_num, page_text in pages:
        section = extract_section_heading(page_text, page_num, source_name)
        sentences = split_into_sentences(page_text)
        windows = sliding_window_chunks(sentences, cfg.chunk_size, cfg.chunk_overlap)

        for window_text in windows:
            if not window_text.strip():
                continue
            chunk = make_chunk(
                source=source_name,
                page=page_num,
                section=section,
                text=window_text,
                cfg=cfg,
                doc_version=doc_version,
            )
            chunks.append(chunk)

    logger.info("Ingested %s: %d chunks from %d pages", source_name, len(chunks), len(pages))
    return chunks


def ingest_directory(
    pdf_dir: Path,
    cfg: "Config",
    manifest_path: Optional[Path] = None,
) -> List[Chunk]:
    """
    Ingest all .pdf files in pdf_dir.
    Reads doc_version from manifest if provided.
    Falls back to DEMO_CORPUS if pdf_dir is empty or no PDFs found.
    Logs per-file chunk counts.
    """
    # Load manifest for doc_version lookup
    manifest: Dict[str, str] = {}
    if manifest_path and manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Could not load manifest: %s", exc)

    pdf_files = sorted(pdf_dir.glob("*.pdf"))

    if not pdf_files:
        logger.warning(
            "No PDFs found in %s — using DEMO_CORPUS. "
            "Run download_quilter_docs.py to fetch real documents.",
            pdf_dir,
        )
        return _load_demo_corpus(cfg)

    all_chunks: List[Chunk] = []
    for pdf_path in pdf_files:
        doc_version = manifest.get(pdf_path.name, "")
        try:
            file_chunks = ingest_pdf(pdf_path, cfg, doc_version=doc_version)
            all_chunks.extend(file_chunks)
        except Exception as exc:
            logger.error("Failed to ingest %s: %s", pdf_path.name, exc)
            continue

    logger.info(
        "Directory ingestion complete: %d PDFs → %d total chunks",
        len(pdf_files), len(all_chunks),
    )
    return all_chunks


def _load_demo_corpus(cfg: "Config") -> List[Chunk]:
    """Convert DEMO_CORPUS list into Chunk objects for demo/test mode."""
    chunks = []
    for item in DEMO_CORPUS:
        chunk = make_chunk(
            source=item["source"],
            page=item["page"],
            section=item["section"],
            text=item["text"],
            cfg=cfg,
            doc_version="demo",
        )
        chunks.append(chunk)
    logger.info("Loaded %d demo chunks", len(chunks))
    return chunks
