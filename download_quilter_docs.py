"""
download_quilter_docs.py 

Downloads publicly available Quilter PDF docs with SHA256 change detection
On each run: only re-downloads files whose hash has changed (or are missing)
Maintains a manifest.json in quilter_docs/ for version tracking

Usage:
    python download_quilter_docs.py    # normal sync
    python download_quilter_docs.py --force   # re-download all
    python download_quilter_docs.py --dry-run    # show what would change

URLs must be verified against https://www.quilter.com/adviser/help-support-documents/
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import httpx
    HTTPX_OK = True
except ImportError:
    HTTPX_OK = False
    print("WARNING-> httpx not installed.Run: pip install httpx")


# The downloader will gracefully handle 404s (marks file as "failed")
QUILTER_URLS: Dict[str, str] = {
    # Platform charges
    "quilter_charges.pdf": (
        "https://www.quilter.com/siteassets/documents/adviser-documents/"
        "platform-charges-and-fees.pdf"
    ),
    # Pension rules and SIPP guide
    "quilter_pensions.pdf": (
        "https://www.quilter.com/siteassets/documents/adviser-documents/"
        "sipp-technical-guide.pdf"
    ),
    # ISA and transfer 
    "quilter_transfers.pdf": (
        "https://www.quilter.com/siteassets/documents/adviser-documents/"
        "isa-and-re-registration-guide.pdf"
    ),
    # Onbroading or  KYC 
    "quilter_onboarding.pdf": (
        "https://www.quilter.com/siteassets/documents/adviser-documents/"
        "client-onboarding-guide.pdf"
    ),
    # Withdrawal or income options
    "quilter_withdrawals.pdf": (
        "https://www.quilter.com/siteassets/documents/adviser-documents/"
        "income-and-withdrawal-options.pdf"
    ),
    # Asset types,restrictions
    "quilter_assets.pdf": (
        "https://www.quilter.com/siteassets/documents/adviser-documents/"
        "permitted-assets-guide.pdf"
    ),
    # Death benefits,expression of wish
    "quilter_death_benefits.pdf": (
        "https://www.quilter.com/siteassets/documents/adviser-documents/"
        "death-benefits-guide.pdf"
    ),
    # Consumer Duty statement
    "quilter_consumer_duty.pdf": (
        "https://www.quilter.com/siteassets/documents/adviser-documents/"
        "consumer-duty-statement.pdf"
    ),
}

# Default paths
DEFAULT_PDF_DIR      = Path("C:/Code/quilter/quilter_docs")
DEFAULT_MANIFEST     = DEFAULT_PDF_DIR / "manifest.json"
DEFAULT_LOG_DIR      = Path("C:/Code/quilter/logs_v3")
DEFAULT_UPDATE_LOG   = DEFAULT_LOG_DIR / "update_log.jsonl"

DOWNLOAD_TIMEOUT     = 60   # seconds
HEADERS = {
    "User-Agent": "QuilterHNWAdvisor/3.0 (research; adviser-support-docs)",
    "Accept": "application/pdf,*/*",
}


# SHA256 utilities check
def sha256_file(path: Path) -> str:
    #Return hex SHA256 of a file. Reads in 64KB chunks for large PDFs
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# version tracking
def load_manifest(manifest_path: Path) -> Dict[str, str]:
    #Load {filename: sha256} manifest. Returns {} if file missing
    if not manifest_path.exists():
        return {}
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def save_manifest(manifest: Dict[str, str], manifest_path: Path) -> None:
    #Atomically write updated manifest
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = manifest_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    tmp.replace(manifest_path)


# Download
def download_pdf(url: str, dest: Path, timeout: int = DOWNLOAD_TIMEOUT) -> Tuple[bool, str]:
    """
    Download a PDF from url to dest using httpx with streaming.
    Returns (success, sha256_or_error_message).
    Validates Content-Type contains 'pdf' or 'octet-stream'
    """
    if not HTTPX_OK:
        return False, "httpx not installed"

    try:
        with httpx.Client(follow_redirects=True, timeout=timeout) as client:
            with client.stream("GET", url, headers=HEADERS) as resp:
                if resp.status_code == 404:
                    return False, f"HTTP 404 — URL not found"
                if resp.status_code != 200:
                    return False, f"HTTP {resp.status_code}"

                ct = resp.headers.get("content-type", "").lower()
                if "pdf" not in ct and "octet-stream" not in ct and "binary" not in ct:
                    # Some CDNs return text/html for missing docs — catch this
                    if "html" in ct:
                        return False, f"Got HTML instead of PDF (content-type: {ct})"

                dest.parent.mkdir(parents=True, exist_ok=True)
                h = hashlib.sha256()
                with open(dest, "wb") as f:
                    for chunk in resp.iter_bytes(chunk_size=65536):
                        f.write(chunk)
                        h.update(chunk)
                return True, h.hexdigest()

    except httpx.TimeoutException:
        return False, "Download timeout"
    except Exception as exc:
        return False, str(exc)


# Audit logging
def log_update(log_dir: Path, entries: List[Dict]) -> None:
    #Append update entries to logs_v3/update_log.jsonl
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "update_log.jsonl"
    with open(log_path, "a", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")


# Main sync function
def sync_quilter_docs(
    pdf_dir: Path = DEFAULT_PDF_DIR,
    manifest_path: Path = DEFAULT_MANIFEST,
    log_dir: Path = DEFAULT_LOG_DIR,
    force: bool = False,
    dry_run: bool = False,
    urls: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """
    Sync Quilter PDFs with SHA256 change detection.
    Algorithm for each URL:
      1. Download to a temp file
      2. Compute SHA256 of downloaded content
      3. Compare to manifest SHA256
      4. If changed (or missing or force=True): move to dest
      5. If unchanged: discard temp file
      6. Log result to update_log.jsonl
    Returns dict: {filename: "added" | "updated" | "unchanged" | "failed" | "dry_run"}
    """
    if urls is None:
        urls = QUILTER_URLS

    pdf_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(manifest_path)
    results: Dict[str, str] = {}
    log_entries: List[Dict] = []
    ts = datetime.now(timezone.utc).isoformat()

    print(f"Syncing {len(urls)} Quilter documents → {pdf_dir}")
    print(f"Manifest: {len(manifest)} known files | force={force} | dry_run={dry_run}\n")

    for filename, url in urls.items():
        dest = pdf_dir / filename
        known_hash = manifest.get(filename, "")

        if dry_run:
            exists = dest.exists()
            status = "present" if exists else "missing"
            print(f"  [DRY-RUN] {filename}: {status} (hash={'known' if known_hash else 'unknown'})")
            results[filename] = "dry_run"
            continue

        print(f"  {filename} ... ", end="", flush=True)

        # Download to temp file
        with tempfile.NamedTemporaryFile(
            dir=pdf_dir, prefix=f".tmp_{filename}_", suffix=".pdf", delete=False
        ) as tmp_f:
            tmp_path = Path(tmp_f.name)

        success, sha256_or_err = download_pdf(url, tmp_path)

        if not success:
            tmp_path.unlink(missing_ok=True)
            print(f"FAILED ({sha256_or_err})")
            results[filename] = "failed"
            log_entries.append({
                "ts": ts, "filename": filename, "status": "failed",
                "error": sha256_or_err, "url": url,
            })
            continue

        new_hash = sha256_or_err
        file_size = tmp_path.stat().st_size

        if not force and new_hash == known_hash:
            tmp_path.unlink(missing_ok=True)
            print(f"unchanged ({file_size/1024:.0f} KB)")
            results[filename] = "unchanged"
            continue

        # Changed (or new / forced): move to destination
        action = "added" if not dest.exists() else "updated"
        tmp_path.replace(dest)
        manifest[filename] = new_hash
        print(f"{action.upper()} ({file_size/1024:.0f} KB, sha256:{new_hash[:12]}...)")
        results[filename] = action
        log_entries.append({
            "ts": ts,
            "filename": filename,
            "status": action,
            "doc_version": new_hash,
            "file_size_bytes": file_size,
            "url": url,
        })

    # Persist updated manifest and logs
    if not dry_run:
        save_manifest(manifest, manifest_path)
        if log_entries:
            log_update(log_dir, log_entries)

    # Summary
    counts = {s: sum(1 for v in results.values() if v == s)
              for s in ("added", "updated", "unchanged", "failed")}
    print(f"\nSync complete: {counts}")
    return results


# CLI entry point
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download/sync Quilter adviser support PDFs"
    )
    parser.add_argument(
        "--pdf-dir", default=str(DEFAULT_PDF_DIR),
        help="Destination directory for PDFs"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-download all files even if SHA256 unchanged"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be downloaded without downloading"
    )
    args = parser.parse_args()

    pdf_dir = Path(args.pdf_dir)
    manifest_path = pdf_dir / "manifest.json"
    log_dir = Path(str(pdf_dir).replace("quilter_docs", "logs_v3"))

    results = sync_quilter_docs(
        pdf_dir=pdf_dir,
        manifest_path=manifest_path,
        log_dir=log_dir,
        force=args.force,
        dry_run=args.dry_run,
    )

    failed = [f for f, s in results.items() if s == "failed"]
    if failed:
        print(f"\nFailed downloads ({len(failed)}): {failed}")
        print("Check the URLs in QUILTER_URLS — Quilter may have moved documents.")
        sys.exit(1)


if __name__ == "__main__":
    main()
