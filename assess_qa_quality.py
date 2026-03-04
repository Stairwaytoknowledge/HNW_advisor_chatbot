import sys, re, csv
sys.stdout.reconfigure(encoding='utf-8')
import pdfplumber
from pathlib import Path


#ADD PATH to YOUR CUSTOM MADE HUMAN EVALS CSV FILES AND PDF SOURCES in the format , question, answer, source document (matching the DOC_MAP keys below) and reference
pdf_path = r"path_to_your_pdf_folder"
YES_NO_csv_path = r"path_to_your_yes_no_csv_file.csv"
YES_NO_detailed_csv_path = r"path_to_your_yes_no_detailed_csv_file.csv"

PDF_DIR = Path(pdf_path)
YESNO_CSV  = Path(YES_NO_csv_path)
DETAIL_CSV = Path(YES_NO_detailed_csv_path)

DOC_MAP = {
    "How to use the money in your pension pot (6593)": "6593_pension_pot.pdf",
    "A guide to your tax voucher (6600)":              "6600_tax_voucher.pdf",
    "Discover the freedom of flexible ISAs (qip23731)":"qip23731_flexible_isa.pdf",
    "Collective Investment Account KFD (18179_cia_kfd)":"18179_cia_kfd.pdf",
}

# Extract text from each PDF
pdf_texts = {}
for key, fname in DOC_MAP.items():
    path = PDF_DIR / fname
    if not path.exists():
        pdf_texts[key] = ""
        print(f"MISSING: {fname}")
        continue
    text = ""
    try:
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t: text += t + "\n"
    except Exception as e:
        print(f"ERROR {fname}: {e}")
    pdf_texts[key] = text
    print(f"Extracted {fname}: {len(text):,} chars, {text.count(chr(10))} lines")

# Load QA CSVs
def load_csv(path):
    with open(path, encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))

yesno_qs  = load_csv(YESNO_CSV)
detail_qs = load_csv(DETAIL_CSV)

# Assess each QA pair, check if key answer terms appear in the PDF
def keyword_in_pdf(answer, pdf_text):
    #sample stopwords to ignore - can be expanded as needed. Focus is on content words that indicate coverage of the answer in the PDF.
    STOPWORDS = {"the","a","an","is","are","of","to","in","and","or","for","that","it","its","with","on","at","by","be","as","from","this","was","can","you","your","have","has","if","not","any","but","so","they","their","we","our","will","may","must","when","what","which","who","do","does","did","been","into","than","more","also","only"}
    words = set(re.sub(r'[^a-z0-9£% ]','',answer.lower()).split()) - STOPWORDS
    words = {w for w in words if len(w) > 2}
    if not words: return 1.0, []
    if not pdf_text: return 0.0, list(words)[:5]
    found = [w for w in words if w in pdf_text.lower()]
    missing = [w for w in words if w not in pdf_text.lower()]
    return len(found)/len(words), missing[:5]

print("\n" + "="*70)
print("  QA QUALITY ASSESSMENT AGAINST PDF SOURCES")
print("="*70)

results = []
for qa_type, qs in [("yesno", yesno_qs), ("detailed", detail_qs)]:
    print(f"\n  {qa_type.upper()} questions ({len(qs)} total)")
    print(f"  {'N':>4}  {'Src':>4}  {'Cov':>5}  {'Q (truncated)'}")
    print("  " + "-"*70)
    
    cov_scores = []
    low_cov = []
    hallucinated = []
    
    for i, row in enumerate(qs):
        q   = row["Question"].strip()
        ref = row["Answer"].strip()
        src = row["Source Document"].strip()
        pdf_text = pdf_texts.get(src, "")
        
        cov, missing = keyword_in_pdf(ref, pdf_text)
        cov_scores.append(cov)
        
        if cov < 0.5:
            low_cov.append((i+1, src[:20], cov, q[:60], ref[:60], missing))
        
        if i < 15:
            sym = "OK" if cov >= 0.7 else ("WARN" if cov >= 0.4 else "FAIL")
            print(f"  {i+1:>4}  {sym}  {cov*100:>4.0f}%  {q[:55]}")
    
    mean_cov = sum(cov_scores)/len(cov_scores)*100
    pass_70 = sum(1 for c in cov_scores if c >= 0.70)/len(cov_scores)*100
    pass_50 = sum(1 for c in cov_scores if c >= 0.50)/len(cov_scores)*100
    
    print(f"\n  SUMMARY: mean_coverage={mean_cov:.1f}%  pass@70%={pass_70:.0f}%  pass@50%={pass_50:.0f}%")
    
    if low_cov:
        print(f"\n  LOW COVERAGE QUERIES (cov < 50%):")
        for n, src_s, cov, q, ref, miss in low_cov[:10]:
            print(f"    #{n} [{src_s}] cov={cov*100:.0f}% q={q[:50]}")
            print(f"         ref={ref[:60]}")
            print(f"         missing_words={miss}")

print("\n" + "="*70)
print("  VERDICT SUMMARY")
print("="*70)
