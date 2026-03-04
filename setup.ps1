# Quilter HNW Advisor Assistant v3 — Windows Setup Script
# Run from C:\Code\quilter with: .\setup.ps1
# Requires: Python 3.12, Ollama installed and running

$ErrorActionPreference = "Stop"
$PROJECT_ROOT = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host "=== Quilter HNW Advisor v3 Setup ===" -ForegroundColor Cyan
Write-Host "Project root: $PROJECT_ROOT"

# ── 1. Virtual environment ────────────────────────────────────────────────
Write-Host "`n[1/6] Creating virtual environment..." -ForegroundColor Yellow
if (-not (Test-Path "$PROJECT_ROOT\.venv")) {
    python -m venv "$PROJECT_ROOT\.venv"
    Write-Host "  Virtual environment created." -ForegroundColor Green
} else {
    Write-Host "  Virtual environment already exists, skipping." -ForegroundColor Gray
}

# ── 2. Activate and upgrade pip ───────────────────────────────────────────
Write-Host "`n[2/6] Activating venv and upgrading pip..." -ForegroundColor Yellow
& "$PROJECT_ROOT\.venv\Scripts\Activate.ps1"
python -m pip install --upgrade pip --quiet
Write-Host "  pip upgraded." -ForegroundColor Green

# ── 3. Install requirements ───────────────────────────────────────────────
Write-Host "`n[3/6] Installing requirements (this may take 5-10 minutes)..." -ForegroundColor Yellow
pip install -r "$PROJECT_ROOT\requirements.txt" --quiet
Write-Host "  All packages installed." -ForegroundColor Green

# ── 4. Create directory structure ─────────────────────────────────────────
Write-Host "`n[4/6] Creating project directories..." -ForegroundColor Yellow
$dirs = @("quilter_docs", "index_v3", "logs_v3", "rails", "src", "eval_data")
foreach ($d in $dirs) {
    $path = Join-Path $PROJECT_ROOT $d
    if (-not (Test-Path $path)) {
        New-Item -ItemType Directory -Path $path | Out-Null
        Write-Host "  Created: $d" -ForegroundColor Gray
    }
}
Write-Host "  Directories OK." -ForegroundColor Green

# ── 5. Ollama model verification ──────────────────────────────────────────
Write-Host "`n[5/6] Checking Ollama models..." -ForegroundColor Yellow

$models = @("qwen2.5:14b", "qwen2.5:7b", "llama3.2:3b")
foreach ($model in $models) {
    Write-Host "  Pulling $model (if not already cached)..." -ForegroundColor Gray
    try {
        ollama pull $model
        Write-Host "  OK: $model" -ForegroundColor Green
    } catch {
        Write-Host "  WARNING: Could not pull $model — is Ollama running?" -ForegroundColor Yellow
        Write-Host "  Start Ollama and run: ollama pull $model" -ForegroundColor Yellow
    }
}

# ── 6. Pre-warm model downloads ───────────────────────────────────────────
Write-Host "`n[6/6] Pre-warming ML model downloads..." -ForegroundColor Yellow

# Pre-warm BERTScore (downloads distilbert-base-uncased ~260MB on first use)
Write-Host "  Pre-warming BERTScore (distilbert-base-uncased)..." -ForegroundColor Gray
python -c "
from bert_score import score as bs_score
try:
    P, R, F1 = bs_score(['test answer'], ['reference answer'], lang='en',
                         model_type='distilbert-base-uncased', verbose=False)
    print('  BERTScore OK — F1:', round(F1.mean().item(), 4))
except Exception as e:
    print(f'  BERTScore warning: {e}')
"

# Pre-warm sentence-transformers (BGE-large ~1.3GB, MiniLM ~90MB)
Write-Host "  Pre-warming sentence-transformers..." -ForegroundColor Gray
python -c "
import warnings; warnings.filterwarnings('ignore')
try:
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer('all-MiniLM-L6-v2')
    _ = m.encode(['test'], normalize_embeddings=True)
    print('  SentenceTransformer OK (MiniLM fallback)')
except Exception as e:
    print(f'  SentenceTransformer warning: {e}')
"

# ── Final verification ────────────────────────────────────────────────────
Write-Host "`n=== Import Verification ===" -ForegroundColor Cyan
python -c "
imports = [
    ('faiss', 'faiss'),
    ('fitz (pymupdf)', 'fitz'),
    ('sentence_transformers', 'sentence_transformers'),
    ('rank_bm25', 'rank_bm25'),
    ('ollama', 'ollama'),
    ('pandas', 'pandas'),
    ('numpy', 'numpy'),
    ('httpx', 'httpx'),
    ('tqdm', 'tqdm'),
    ('bert_score', 'bert_score'),
]
all_ok = True
for name, mod in imports:
    try:
        __import__(mod)
        print(f'  OK  {name}')
    except ImportError as e:
        print(f'  FAIL {name}: {e}')
        all_ok = False

# crewai optional
try:
    import crewai
    print(f'  OK  crewai')
except ImportError:
    print(f'  WARN crewai not available (install separately if needed)')

# nemoguardrails optional
try:
    import nemoguardrails
    print(f'  OK  nemoguardrails')
except ImportError:
    print(f'  WARN nemoguardrails not available (Python fallback will be used)')

if all_ok:
    print()
    print('All core imports verified.')
else:
    print()
    print('Some imports failed. Re-run: pip install -r requirements.txt')
"

Write-Host "`n=== Setup Complete ===" -ForegroundColor Cyan
Write-Host "Next steps:" -ForegroundColor White
Write-Host "  1. Activate venv: .\.venv\Scripts\Activate.ps1" -ForegroundColor Gray
Write-Host "  2. Download Quilter docs: python download_quilter_docs.py" -ForegroundColor Gray
Write-Host "  3. Open notebook: jupyter notebook quilter_hnw_advisor_v3.ipynb" -ForegroundColor Gray
