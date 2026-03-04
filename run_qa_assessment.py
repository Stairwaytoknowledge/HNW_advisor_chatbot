"""
run_qa_assessment.py
====================
Full production-readiness assessment using 200 QA pairs
(100 yes/no + 100 descriptive) against the 4 Quilter source PDFs.

Steps:
  1. Download the 4 PDFs (or reuse cached copies)
  2. Extract and chunk PDF text
  3. Embed corpus with BGE-large
  4. Load NLI and BERTScore models
  5. Load QA datasets
  6. Generate answers via Ollama
  7. Run evaluation (200 queries)
  8. Save results CSV + produce readiness report

Usage:
    python run_qa_assessment.py
    python run_qa_assessment.py --yesno path/to/yesno.csv --detail path/to/detail.csv
"""
import sys, os, re, csv, json, time, math, textwrap, argparse
from pathlib import Path
from collections import defaultdict

sys.stdout.reconfigure(encoding="utf-8")

_HERE = Path(__file__).resolve().parent
os.chdir(_HERE)
sys.path.insert(0, str(_HERE))

parser = argparse.ArgumentParser(description="Quilter QA production-readiness assessment")
parser.add_argument(
    "--yesno",
    default=None,
    help=(
        "Path to the Yes/No QA CSV file. "
        "Defaults to eval_data/gold_eval_set.json or the Downloads folder."
    ),
)
parser.add_argument(
    "--detail",
    default=None,
    help=(
        "Path to the Detailed QA CSV file. "
        "Defaults to eval_data/gold_eval_set.json or the Downloads folder."
    ),
)
args = parser.parse_args()

def _find_csv(arg_val, fallback_name):
    """Resolve a CSV path: CLI arg -> eval_data/ -> Downloads."""
    if arg_val and Path(arg_val).exists():
        return Path(arg_val)
    local = _HERE / "eval_data" / fallback_name
    if local.exists():
        return local
    dl = Path.home() / "Downloads" / fallback_name
    if dl.exists():
        return dl
    return None

YESNO_CSV_PATH  = _find_csv(args.yesno,  "Question_Answer_Source Document.csv")
DETAIL_CSV_PATH = _find_csv(args.detail, "Question_Answer_Source_Document_detailed.csv")

PDF_DIR    = _HERE / "eval_data" / "quilter_pdfs"
OUTPUT_CSV = _HERE / "logs_v3" / "qa_assessment_results.csv"
REPORT_OUT = _HERE / "logs_v3" / "qa_production_readiness_report.txt"
PDF_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

PDFS = {
    "6593_pension_pot.pdf":     "https://www.quilter.com/4a943f/siteassets/documents/platform/guides-and-brochures/6593_how_to_use_the_money_in_your_pension_pot.pdf",
    "6600_tax_voucher.pdf":     "https://www.quilter.com/495e6c/siteassets/documents/platform/guides-and-brochures/6600_a_guide_to_your_tax_voucher.pdf",
    "qip23731_flexible_isa.pdf":"https://www.quilter.com/49d8ff/siteassets/documents/platform/guides-and-brochures/qip23731-discover-the-freedom-of-flexible-isas-customer-flyer.pdf",
    "18179_cia_kfd.pdf":        "https://www.quilter.com/4a5cb4/siteassets/documents/platform/kfd/18179_cia_kfd.pdf",
}

DOC_MAP = {
    "How to use the money in your pension pot (6593)":         "6593_pension_pot.pdf",
    "A guide to your tax voucher (6600)":                      "6600_tax_voucher.pdf",
    "Discover the freedom of flexible ISAs (qip23731)":        "qip23731_flexible_isa.pdf",
    "Collective Investment Account KFD (18179_cia_kfd)":       "18179_cia_kfd.pdf",
}

print("=" * 68)
print("  STEP 1 — Downloading PDFs")
print("=" * 68)
import urllib.request, urllib.error

headers = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    )
}

pdf_paths = {}
for fname, url in PDFS.items():
    dest = PDF_DIR / fname
    pdf_paths[fname] = dest
    if dest.exists() and dest.stat().st_size > 10_000:
        print(f"  [cached] {fname} ({dest.stat().st_size:,} bytes)")
        continue
    print(f"  [downloading] {fname} ...", end="", flush=True)
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
        dest.write_bytes(data)
        print(f" {len(data):,} bytes OK")
    except Exception as e:
        print(f" FAILED: {e}")


print()
print("=" * 68)
print("  STEP 2 — Extracting PDF text")
print("=" * 68)

try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False
    print("  [warn] pdfplumber not installed — trying pypdf")

try:
    import pypdf
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False

def extract_text(pdf_path):
    text = ""
    if HAS_PDFPLUMBER:
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    t = page.extract_text()
                    if t:
                        text += t + "\n"
            return text
        except Exception:
            pass
    if HAS_PYPDF:
        try:
            reader = pypdf.PdfReader(str(pdf_path))
            for page in reader.pages:
                t = page.extract_text()
                if t:
                    text += t + "\n"
            return text
        except Exception:
            pass
    return ""

pdf_texts = {}
for fname, dest in pdf_paths.items():
    if dest.exists():
        txt = extract_text(dest)
        pdf_texts[fname] = txt
        print(f"  {fname}: {len(txt):,} chars, {txt.count(chr(10))} lines")
    else:
        pdf_texts[fname] = ""
        print(f"  {fname}: MISSING — skipping")

def chunk_text(text, size=400, overlap=80):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i+size])
        chunks.append(chunk)
        i += size - overlap
    return chunks

all_chunks = {}
for fname, text in pdf_texts.items():
    all_chunks[fname] = chunk_text(text)
    print(f"  {fname}: {len(all_chunks[fname])} chunks")

flat_chunks = []
flat_texts  = []
for fname, chunks in all_chunks.items():
    for c in chunks:
        flat_chunks.append((fname, c))
        flat_texts.append(c)

print(f"\n  Total corpus: {len(flat_chunks)} chunks across {len(all_chunks)} docs")


print()
print("=" * 68)
print("  STEP 3 — Embedding corpus (BAAI/bge-large-en-v1.5)")
print("=" * 68)

import numpy as np
from sentence_transformers import SentenceTransformer

embed_model = SentenceTransformer("BAAI/bge-large-en-v1.5")
print("  Model loaded. Encoding corpus ...", flush=True)
corpus_embs = embed_model.encode(flat_texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
print(f"  Corpus embeddings: {corpus_embs.shape}")

def retrieve(query, k=5):
    q_emb  = embed_model.encode([query], normalize_embeddings=True)[0]
    scores = corpus_embs @ q_emb
    top_k  = np.argsort(scores)[::-1][:k]
    return [(flat_chunks[i][0], flat_chunks[i][1], float(scores[i])) for i in top_k]


print()
print("=" * 68)
print("  STEP 4 — Loading NLI + BERTScore models")
print("=" * 68)

from transformers import pipeline as hf_pipeline
import bert_score as bslib

nli_pipe = hf_pipeline(
    "zero-shot-classification",
    model="cross-encoder/nli-deberta-v3-large",
    device=-1,
)
print("  NLI model ready.")


print()
print("=" * 68)
print("  STEP 5 — Loading QA datasets")
print("=" * 68)

def load_csv(path):
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))

if YESNO_CSV_PATH is None or not YESNO_CSV_PATH.exists():
    print("  ERROR: Yes/No CSV not found.")
    print("  Pass it with: python run_qa_assessment.py --yesno /path/to/file.csv")
    sys.exit(1)

if DETAIL_CSV_PATH is None or not DETAIL_CSV_PATH.exists():
    print("  ERROR: Detailed CSV not found.")
    print("  Pass it with: python run_qa_assessment.py --detail /path/to/file.csv")
    sys.exit(1)

yesno_qs  = load_csv(YESNO_CSV_PATH)
detail_qs = load_csv(DETAIL_CSV_PATH)
print(f"  Yes/No   : {len(yesno_qs)} questions  (from {YESNO_CSV_PATH.name})")
print(f"  Detailed : {len(detail_qs)} questions  (from {DETAIL_CSV_PATH.name})")


print()
print("=" * 68)
print("  STEP 6 — Generating answers via Ollama (llama3.2:3b)")
print("=" * 68)

import subprocess

def ask_ollama(prompt, model="llama3.2:3b", timeout=60):
    try:
        result = subprocess.run(
            ["ollama", "run", model],
            input=prompt, capture_output=True, text=True, timeout=timeout,
            encoding="utf-8", errors="replace"
        )
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        return "[TIMEOUT]"
    except Exception as e:
        return f"[ERROR: {e}]"

def build_prompt(question, chunks, qa_type="detailed"):
    context = "\n---\n".join(c[1] for c in chunks[:3])
    if qa_type == "yesno":
        return (
            f"You are a Quilter financial services assistant. Answer ONLY 'Yes' or 'No' "
            f"based on the following context. Do not explain.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\nAnswer (Yes/No):"
        )
    else:
        return (
            f"You are a Quilter financial services assistant. Answer concisely and accurately "
            f"based on the following context. Use only information from the context.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\nAnswer:"
        )


print()
print("=" * 68)
print("  STEP 7 — Running evaluation (200 queries)")
print("=" * 68)

results = []

def eval_batch(questions, qa_type):
    for i, row in enumerate(questions):
        q   = row["Question"].strip()
        ref = row["Answer"].strip()
        src = row["Source Document"].strip()
        src_fname = DOC_MAP.get(src, "")

        t0   = time.time()
        hits = retrieve(q, k=5)
        top_fname = hits[0][0] if hits else ""
        top_score = hits[0][2] if hits else 0.0

        top3_fnames   = [h[0] for h in hits[:3]]
        ret_prec_at3  = 1.0 if src_fname in top3_fnames else 0.0

        answer  = ask_ollama(build_prompt(q, hits, qa_type), timeout=90)
        elapsed = time.time() - t0

        context_text = " ".join(h[1] for h in hits[:3])[:2000]
        try:
            nli_out = nli_pipe(
                answer[:512],
                candidate_labels=["entailment", "contradiction", "neutral"],
                hypothesis_template="{}",
            )
            faith = dict(zip(nli_out["labels"], nli_out["scores"])).get("entailment", 0.0)
        except Exception:
            faith = 0.0

        try:
            P, R, F1 = bslib.score(
                [answer], [ref],
                lang="en", model_type="microsoft/deberta-v3-large",
                verbose=False, rescale_with_baseline=True,
            )
            bs_f1 = float(F1[0])
        except Exception:
            bs_f1 = 0.0

        ans_clean = re.sub(r'[^a-z0-9 ]', '', answer.lower().strip())
        ref_clean = re.sub(r'[^a-z0-9 ]', '', ref.lower().strip())
        if qa_type == "yesno":
            yn = re.search(r'\b(yes|no)\b', ans_clean)
            gen_yn = yn.group(1) if yn else "?"
            ref_yn = "yes" if ref_clean.startswith("yes") else ("no" if ref_clean.startswith("no") else ref_clean[:3])
            exact_match = 1.0 if gen_yn == ref_yn else 0.0
            fuzzy_match = exact_match
        else:
            ref_toks    = set(ref_clean.split())
            ans_toks    = set(ans_clean.split())
            tok_recall  = len(ref_toks & ans_toks) / max(len(ref_toks), 1)
            exact_match = 1.0 if ref_clean == ans_clean else 0.0
            fuzzy_match = tok_recall

        results.append({
            "qa_type":       qa_type,
            "qid":           f"{qa_type[0].upper()}{i+1:03d}",
            "source_doc":    src,
            "question":      q,
            "reference":     ref,
            "generated":     answer[:300],
            "ret_prec_at3":  ret_prec_at3,
            "top_ret_score": round(top_score, 3),
            "faithfulness":  round(faith, 3),
            "bertscore_f1":  round(bs_f1, 3),
            "exact_match":   round(exact_match, 3),
            "fuzzy_match":   round(fuzzy_match, 3),
            "latency_s":     round(elapsed, 1),
        })

        ok_sym = "Y" if fuzzy_match >= 0.5 else "N"
        print(
            f"  [{qa_type[:1].upper()}{i+1:03d}] {ok_sym}  "
            f"ret@3={ret_prec_at3:.0f}  faith={faith:.2f}  "
            f"bs={bs_f1:.2f}  fuzz={fuzzy_match:.2f}  {elapsed:.1f}s  "
            f"q={q[:55]}",
            flush=True,
        )

print("\n  --- Yes/No questions ---")
eval_batch(yesno_qs, "yesno")

print("\n  --- Detailed questions ---")
eval_batch(detail_qs, "detailed")


fieldnames = list(results[0].keys())
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(results)
print(f"\n  [Saved] {OUTPUT_CSV}")


print()
print("=" * 68)
print("  STEP 8 — Generating production-readiness report")
print("=" * 68)

def stats(vals):
    if not vals: return 0, 0, 0
    return sum(vals)/len(vals), min(vals), max(vals)

def pct_pass(vals, threshold):
    return sum(1 for v in vals if v >= threshold) / max(len(vals), 1) * 100

yn_rows  = [r for r in results if r["qa_type"] == "yesno"]
det_rows = [r for r in results if r["qa_type"] == "detailed"]

def report_section(rows, label):
    faith_vals = [r["faithfulness"] for r in rows]
    bs_vals    = [r["bertscore_f1"] for r in rows]
    ret_vals   = [r["ret_prec_at3"] for r in rows]
    fuzz_vals  = [r["fuzzy_match"]  for r in rows]
    em_vals    = [r["exact_match"]  for r in rows]
    lat_vals   = [r["latency_s"]    for r in rows]
    return {
        "label":         label,
        "n":             len(rows),
        "ret_prec_at3":  stats(ret_vals),
        "faithfulness":  stats(faith_vals),
        "bertscore_f1":  stats(bs_vals),
        "fuzzy_match":   stats(fuzz_vals),
        "exact_match":   stats(em_vals),
        "latency":       stats(lat_vals),
        "faith_pass_40": pct_pass(faith_vals, 0.40),
        "bs_pass_65":    pct_pass(bs_vals, 0.65),
        "ret_pass":      pct_pass(ret_vals, 1.0),
        "fuzz_pass_50":  pct_pass(fuzz_vals, 0.50),
        "fuzz_pass_70":  pct_pass(fuzz_vals, 0.70),
    }

yn_s  = report_section(yn_rows,  "Yes/No (100 q)")
det_s = report_section(det_rows, "Detailed (100 q)")
all_s = report_section(results,  "Combined (200 q)")

doc_breakdown = {}
for src, fname in DOC_MAP.items():
    doc_rows = [r for r in results if r["source_doc"] == src]
    if doc_rows:
        doc_breakdown[src] = report_section(doc_rows, src)

lines = []
def w(s=""): lines.append(s)

w("=" * 72)
w("  QUILTER HNW ADVISOR — PRODUCTION READINESS ASSESSMENT")
w("  200 Ground-Truth QA Pairs across 4 Source Documents")
w(f"  Generated: {time.strftime('%Y-%m-%d %H:%M')} UTC")
w("=" * 72)

w()
w("  CORPUS")
for fname, chunks in all_chunks.items():
    pdf_p = PDF_DIR / fname
    sz = f"{pdf_p.stat().st_size:,}" if pdf_p.exists() else "missing"
    w(f"    {fname:<36}  {len(chunks):>4} chunks  ({sz} bytes)")
w(f"    Total: {len(flat_chunks)} chunks")

w()
w("  EVALUATION DATASETS")
w(f"    Yes/No questions      : {len(yesno_qs):>4}  (binary correctness, factual)")
w(f"    Descriptive questions : {len(detail_qs):>4}  (open-ended, explanatory)")
w(f"    Source documents      : 4 official Quilter PDFs")

w()
w("  METRICS SUMMARY")
w(f"  {'Metric':<36} {'YesNo':>9} {'Detailed':>9} {'Combined':>9}  Target")
w("  " + "-" * 68)

rows_meta = [
    ("Retrieval Precision@3",   "ret_prec_at3",  ">=80%"),
    ("NLI Faithfulness (mean)", "faithfulness",  ">=40%"),
    ("  % queries >=0.40",      "faith_pass_40", ">=60%"),
    ("BERTScore F1 (mean)",     "bertscore_f1",  ">=65%"),
    ("  % queries >=0.65",      "bs_pass_65",    ">=75%"),
    ("Fuzzy Match >=50% tokens","fuzz_pass_50",  ">=70%"),
    ("Fuzzy Match >=70% tokens","fuzz_pass_70",  ">=50%"),
    ("Exact Match",             "exact_match",   "--"),
    ("Mean Latency (s)",        "latency",       "<=10s"),
]

for label, key, target in rows_meta:
    def fmt(s, key):
        v = s[key]
        if isinstance(v, tuple):
            mean, mn, mx = v
            return f"{mean*100:6.1f}%"
        else:
            return f"{v:6.1f}%"
    yn_v  = fmt(yn_s,  key)
    det_v = fmt(det_s, key)
    all_v = fmt(all_s, key)
    if key == "latency":
        yn_v  = f"{yn_s['latency'][0]:6.1f}s"
        det_v = f"{det_s['latency'][0]:6.1f}s"
        all_v = f"{all_s['latency'][0]:6.1f}s"
    w(f"  {label:<36} {yn_v:>9} {det_v:>9} {all_v:>9}  {target}")

w()
w("  PER-DOCUMENT BREAKDOWN (Combined yes/no + detailed)")
w("  " + "-" * 68)
w(f"  {'Document':<44} {'N':>3}  {'Ret@3':>6}  {'Faith':>6}  {'BS-F1':>6}  {'Fuzz':>6}")
w("  " + "-" * 68)
for src, s in doc_breakdown.items():
    short = src[:42]
    w(
        f"  {short:<44} {s['n']:>3}  "
        f"{s['ret_prec_at3'][0]*100:>5.0f}%  "
        f"{s['faithfulness'][0]*100:>5.1f}%  "
        f"{s['bertscore_f1'][0]*100:>5.1f}%  "
        f"{s['fuzzy_match'][0]*100:>5.0f}%"
    )

w()
w("  PRODUCTION READINESS VERDICT")
w("  " + "=" * 68)

criteria = [
    ("Retrieval Precision@3 >= 80%",             all_s["ret_prec_at3"][0]*100, 80,  True),
    ("NLI Faithfulness mean >= 40%",             all_s["faithfulness"][0]*100,  40,  True),
    ("BERTScore F1 mean >= 65%",                 all_s["bertscore_f1"][0]*100,  65,  True),
    ("Fuzzy match pass rate >= 70% (>=50% tok)", all_s["fuzz_pass_50"],         70,  True),
    ("Mean latency <= 10s per query",            all_s["latency"][0],           10,  False),
]

passed = 0
for name, val, thresh, want_high in criteria:
    ok  = (val >= thresh) if want_high else (val <= thresh)
    sym = "PASS" if ok else "FAIL"
    if ok: passed += 1
    unit = "s" if "latency" in name.lower() else "%"
    w(f"    [{sym}]  {name:<45}  {val:.1f}{unit}")

w()
w(f"    Overall: {passed}/{len(criteria)} criteria met")
w()

if passed == len(criteria):
    verdict = "PRODUCTION READY"
elif passed >= len(criteria) - 1:
    verdict = "NEAR PRODUCTION READY — minor gaps only"
elif passed >= len(criteria) - 2:
    verdict = "CONDITIONAL — address failing criteria before go-live"
else:
    verdict = "NOT PRODUCTION READY — significant gaps"

w(f"    VERDICT: {verdict}")

w()
w("  RECOMMENDATIONS FOR PRODUCTION GO-LIVE")
w("  " + "-" * 68)
w("  Priority 1 — Index the real Quilter PDFs (highest impact).")
w("  Priority 2 — Switch NLI direction: premise=retrieved_context, hypothesis=answer.")
w("  Priority 3 — Add chunk-level page citations to all final answers (FCA traceability).")
w("  Priority 4 — Tune Ollama prompts to use exact regulatory phrasing from source PDFs.")
w("  Priority 5 — Add a reranker (cross-encoder/ms-marco-MiniLM-L-6-v2) after BM25+dense.")

w()
w("=" * 72)
w(f"  Report saved: {REPORT_OUT}")
w("=" * 72)

report_text = "\n".join(lines)
print(report_text)

with open(REPORT_OUT, "w", encoding="utf-8") as f:
    f.write(report_text)

print(f"\n[Done] Report at {REPORT_OUT}")
print(f"[Done] Full results CSV at {OUTPUT_CSV}")
