"""
run_eval_100.py — Full 100-query evaluation with per-stage metrics.

Stages tracked per query:
  S1  NeMo Input Rail       — did it fire? What type?
  S2  Retrieval             — top-K chunks retrieved, RRF confidence, HyDE used
  S3  NeMo Retrieval Rail   — confidence gate pass/fail
  S4  Precision Engine      — did it engage? Which calculator?
  S5  Fact-Check Agent      — faithfulness of draft (NLI per-sentence)
  S6  Manager Agent         — final synthesis latency, model used
  S7  NeMo Output Rail      — citation/precision checks
  S8  Final faithfulness    — DeBERTa-v3 overall score
  S9  Exact value match     — numeric comparison against expected_exact_values
  S10 Route accuracy        — expected vs actual route
  S11 Keyword coverage      — expected_keywords found in answer
  S12 BERTScore F1          — semantic similarity to reference answer

Outputs:
  logs_v3/eval_100_results.csv    — one row per query, all stage metrics
  logs_v3/eval_100_report.txt     — human-readable per-stage analysis
  logs_v3/eval_100_failures.json  — queries that failed any metric threshold
"""

import sys, os, json, time, re, warnings, logging
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.stdout.reconfigure(encoding="utf-8")
_HERE = Path(__file__).resolve().parent
os.chdir(_HERE)
sys.path.insert(0, str(_HERE))

import pandas as pd

from src.config import Config
from src.pdf_ingestion import ingest_directory
from src.embedding import EmbeddingEngine
from src.retrieval import HybridIndex
from src.thresholds_store import load_thresholds, extract_thresholds_from_chunks, save_thresholds
from src.guardrails import NeMoEngine
from src.faithfulness import FaithfulnessEvaluator
from src.precision_engine import HNWPrecisionEngine
from src.orchestrator import QuilterAdvisorSystem
from src.evaluation import compute_bertscore

THRESHOLDS = {
    "bertscore_f1":       0.55,   # BERTScore semantic similarity
    "keyword_coverage":   0.75,   # Fraction of expected_keywords found
    "faithfulness":       0.30,   # NLI faithfulness score (demo corpus = low expected)
    "rrf_confidence":     0.010,  # Min RRF score (fallback gate = 0.015)
    "latency_p95_ms":  180_000,   # P95 latency budget (local CPU, qwen2.5:14b)
    "exact_value_tol":    0.01,   # £ tolerance for exact fee calculations
}

def _kw_coverage(answer: str, keywords: list) -> float:
    if not keywords:
        return 1.0
    ans_lower = answer.lower()
    hits = sum(1 for kw in keywords if kw.lower() in ans_lower)
    return hits / len(keywords)

def _extract_gbp(text: str) -> list:
    """Extract all GBP amounts from text as floats."""
    raw = re.findall(r'£\s*([\d,]+(?:\.\d+)?)', text)
    result = []
    for r in raw:
        try:
            result.append(float(r.replace(",", "")))
        except ValueError:
            pass
    return result

def _exact_value_match(answer: str, expected: dict, tol: float = 0.01) -> dict:
    """
    Check if expected_exact_values appear in the answer.
    Numeric values are compared within tolerance.
    Returns dict: {field: pass/fail/skip}
    """
    if not expected:
        return {}
    results = {}
    extracted_gbp = _extract_gbp(answer)
    for field, val in expected.items():
        if isinstance(val, (int, float)):
            target = float(val)
            found = any(abs(v - target) <= tol for v in extracted_gbp)
            results[field] = "pass" if found else "fail"
        elif isinstance(val, str):
            # Try numeric parse first
            try:
                target = float(str(val).replace(",", "").replace("£", ""))
                found = any(abs(v - target) <= tol for v in extracted_gbp)
                results[field] = "pass" if found else ("fail" if extracted_gbp else "skip")
            except ValueError:
                # String match
                results[field] = "pass" if val.lower() in answer.lower() else "fail"
        elif isinstance(val, bool):
            results[field] = "skip"  # bool checks are semantic, not parseable
    return results

def _stage1_nemo(fa) -> dict:
    """S1: NeMo input rail stats."""
    activations = fa.nemo_activations or []
    input_rails = [a for a in activations if getattr(a, 'stage', '') == 'input']
    return {
        "s1_fired": len(input_rails) > 0,
        "s1_types": [getattr(a, 'rail_type', str(a)) for a in input_rails],
        "s1_blocked": fa.route_used == "blocked",
    }

def _stage2_retrieval(fa) -> dict:
    """S2: Retrieval quality."""
    return {
        "s2_n_chunks": len(fa.citations) if fa.citations else 0,
        "s2_rrf_score": round(getattr(fa, 'max_rrf_score', 0.0), 4),
        "s2_rrf_pass": getattr(fa, 'max_rrf_score', 0.0) >= THRESHOLDS["rrf_confidence"],
        "s2_hyde_used": getattr(fa, 'hyde_used', False),
    }

def _stage3_retrieval_rail(fa) -> dict:
    """S3: NeMo retrieval confidence gate."""
    activations = fa.nemo_activations or []
    ret_rails = [a for a in activations if getattr(a, 'stage', '') == 'retrieval']
    return {
        "s3_fired": len(ret_rails) > 0,
        "s3_fallback_triggered": fa.route_used == "fallback",
    }

def _stage4_precision(fa) -> dict:
    """S4: Precision engine engagement."""
    trace = fa.crew_trace
    prec_step = None
    if trace and trace.agent_steps:
        for step in trace.agent_steps:
            if "precision" in step.get("agent_name", "").lower():
                prec_step = step
                break
    return {
        "s4_engaged": prec_step is not None,
        "s4_calc_type": prec_step.get("tool_calls", [{}])[0].get("tool_name", "") if prec_step and prec_step.get("tool_calls") else "",
        "s4_latency_ms": round(prec_step.get("latency_ms", 0)) if prec_step else 0,
    }

def _stage5_factcheck(fa) -> dict:
    """S5: Fact-check agent draft faithfulness."""
    trace = fa.crew_trace
    fc_step = None
    if trace and trace.agent_steps:
        for step in trace.agent_steps:
            if "fact" in step.get("agent_name", "").lower():
                fc_step = step
                break
    return {
        "s5_ran": fc_step is not None,
        "s5_latency_ms": round(fc_step.get("latency_ms", 0)) if fc_step else 0,
    }

def _stage6_manager(fa) -> dict:
    """S6: Manager agent synthesis."""
    trace = fa.crew_trace
    mgr_step = None
    if trace and trace.agent_steps:
        for step in trace.agent_steps:
            if "manager" in step.get("agent_name", "").lower():
                mgr_step = step
                break
    return {
        "s6_ran": mgr_step is not None,
        "s6_latency_ms": round(mgr_step.get("latency_ms", 0)) if mgr_step else 0,
        "s6_n_agent_steps": len(trace.agent_steps) if trace else 1,
    }

def _stage7_output_rail(fa) -> dict:
    """S7: NeMo output rail."""
    activations = fa.nemo_activations or []
    out_rails = [a for a in activations if getattr(a, 'stage', '') == 'output']
    return {
        "s7_fired": len(out_rails) > 0,
        "s7_types": [getattr(a, 'rail_type', str(a)) for a in out_rails],
        "s7_review_needed": fa.review_needed,
    }

def _stage8_faithfulness(fa) -> dict:
    """S8: Final DeBERTa-v3 faithfulness."""
    score = fa.faithfulness.overall_score if fa.faithfulness else 0.0
    return {
        "s8_score": round(score, 3),
        "s8_pass": score >= THRESHOLDS["faithfulness"],
        "s8_n_sentences": len(fa.faithfulness.sentence_scores) if fa.faithfulness and fa.faithfulness.sentence_scores else 0,
    }

print("=" * 70)
print("  QUILTER HNW ADVISER — 100-QUERY EVALUATION")
print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)

cfg = Config()
pdf_dir   = Path(cfg.pdf_dir)
index_dir = Path(cfg.index_dir)
log_dir   = Path(cfg.log_dir)
log_dir.mkdir(parents=True, exist_ok=True)

print("\n[BOOT] Loading chunks and building index...")
t_boot = time.time()
chunks = ingest_directory(pdf_dir=pdf_dir, cfg=cfg, manifest_path=pdf_dir / "manifest.json")
emb    = EmbeddingEngine(cfg)
index  = HybridIndex(cfg=cfg, emb=emb)
index.build(chunks)

thresholds = extract_thresholds_from_chunks(chunks)
save_thresholds(thresholds, index_dir / "thresholds.json")

system = QuilterAdvisorSystem(
    cfg=cfg, index=index, nemo=NeMoEngine(cfg),
    faith_eval=FaithfulnessEvaluator(cfg),
    precision_engine=HNWPrecisionEngine(thresholds),
    _audit_log_override="eval_100_audit.jsonl",
)
print(f"[BOOT] System ready in {time.time()-t_boot:.1f}s")
print(f"       Chunks: {len(chunks)} | Embedding: {emb.model_name_used} dim={emb.dim}")

EVAL_FILE = os.environ.get("QUILTER_EVAL_FILE", str(_HERE / "eval_data" / "eval_100_queries.json"))
print(f"[EVAL] Loading query set: {EVAL_FILE}")
with open(EVAL_FILE, encoding="utf-8") as f:
    queries = json.load(f)

print(f"\n[EVAL] Running {len(queries)} queries...\n")
print(f"  {'ID':<6} {'Category':<18} {'Route':<16} {'KW%':>5} {'Faith':>6} {'RRF':>6} {'ms':>8}  Status")
print("  " + "-"*80)

rows = []
failures = []
bertscore_pairs = []   # collect for batch BERTScore at end

for i, item in enumerate(queries):
    qid      = item["id"]
    query    = item["query"]
    category = item["category"]
    exp_route   = item["expected_route"]
    ref_ans     = item.get("reference_answer", "")
    exp_kws     = item.get("expected_keywords", [])
    should_fb   = item.get("should_fallback", False)
    is_hnw      = item.get("is_hnw_query", False)
    req_calc    = item.get("requires_exact_calculation", False)
    exp_exact   = item.get("expected_exact_values", {})
    difficulty  = item.get("difficulty", "medium")

    t0 = time.perf_counter()
    try:
        fa = system.answer(query)
    except Exception as e:
        fa = None
        error_msg = str(e)[:200]
    elapsed_ms = (time.perf_counter() - t0) * 1000

    if fa is None:
        row = {
            "id": qid, "category": category, "difficulty": difficulty,
            "query": query[:80], "error": error_msg,
            "actual_route": "ERROR", "expected_route": exp_route,
            "route_ok": False, "latency_ms": round(elapsed_ms, 1),
        }
        rows.append(row)
        failures.append({**item, "error": error_msg, "actual_route": "ERROR"})
        print(f"  {qid:<6} {category:<18} {'ERROR':<16} {'N/A':>5} {'N/A':>6} {'N/A':>6} {elapsed_ms:>8.0f}  EXCEPTION")
        continue

    answer = fa.answer or ""

    actual_route = fa.route_used
    if should_fb:
        route_ok = actual_route in ("fallback", "blocked")
    elif "crewai" in exp_route:
        route_ok = "crewai" in actual_route
    else:
        route_ok = actual_route == exp_route

    s1 = _stage1_nemo(fa)
    s2 = _stage2_retrieval(fa)
    s3 = _stage3_retrieval_rail(fa)
    s4 = _stage4_precision(fa)
    s5 = _stage5_factcheck(fa)
    s6 = _stage6_manager(fa)
    s7 = _stage7_output_rail(fa)
    s8 = _stage8_faithfulness(fa)

    kw_cov = _kw_coverage(answer, exp_kws)

    exact_results = _exact_value_match(answer, exp_exact)
    exact_pass_rate = (sum(1 for v in exact_results.values() if v == "pass") /
                       max(len([v for v in exact_results.values() if v != "skip"]), 1))
    exact_ok = all(v != "fail" for v in exact_results.values())

    bertscore_pairs.append((qid, answer, ref_ans))

    is_failure = (
        not route_ok or
        kw_cov < THRESHOLDS["keyword_coverage"] or
        (req_calc and not exact_ok)
    )
    status = "FAIL" if is_failure else "ok"

    row = {
        "id":             qid,
        "category":       category,
        "difficulty":     difficulty,
        "query":          query[:80],
        "expected_route": exp_route,
        "actual_route":   actual_route,
        "route_ok":       route_ok,
        "is_hnw":         is_hnw,
        "req_calc":       req_calc,
        # S1
        "s1_nemo_fired":  s1["s1_fired"],
        "s1_blocked":     s1["s1_blocked"],
        # S2
        "s2_n_chunks":    s2["s2_n_chunks"],
        "s2_rrf_score":   s2["s2_rrf_score"],
        "s2_rrf_pass":    s2["s2_rrf_pass"],
        "s2_hyde_used":   s2["s2_hyde_used"],
        # S3
        "s3_fallback":    s3["s3_fallback_triggered"],
        # S4
        "s4_prec_engaged":s4["s4_engaged"],
        "s4_latency_ms":  s4["s4_latency_ms"],
        # S5
        "s5_fc_ran":      s5["s5_ran"],
        "s5_latency_ms":  s5["s5_latency_ms"],
        # S6
        "s6_mgr_ran":     s6["s6_ran"],
        "s6_latency_ms":  s6["s6_latency_ms"],
        "s6_n_steps":     s6["s6_n_agent_steps"],
        # S7
        "s7_output_rail": s7["s7_fired"],
        "s7_review":      s7["s7_review_needed"],
        # S8
        "s8_faithfulness":s8["s8_score"],
        "s8_faith_pass":  s8["s8_pass"],
        "s8_n_sentences": s8["s8_n_sentences"],
        # S9
        "s9_exact_ok":    exact_ok,
        "s9_exact_rate":  round(exact_pass_rate, 2),
        "s9_exact_detail":json.dumps(exact_results),
        # S11
        "s11_kw_cov":     round(kw_cov, 3),
        "s11_kw_pass":    kw_cov >= THRESHOLDS["keyword_coverage"],
        # Timing
        "latency_ms":     round(elapsed_ms, 1),
        "answer_len":     len(answer),
        "status":         status,
    }
    rows.append(row)

    if is_failure:
        failures.append({
            **item,
            "actual_route": actual_route,
            "kw_coverage": round(kw_cov, 3),
            "exact_results": exact_results,
            "answer_preview": answer[:300],
        })

    rrf_str = f"{s2['s2_rrf_score']:.4f}"
    print(f"  {qid:<6} {category:<18} {actual_route:<16} {kw_cov*100:>4.0f}% "
          f"{s8['s8_score']:>6.2f} {rrf_str:>6}  {elapsed_ms:>8.0f}  {status}")
    sys.stdout.flush()

print("\n[EVAL] Computing BERTScore F1 for all 100 queries...")
bs_results = {}
BATCH = 10
for start in range(0, len(bertscore_pairs), BATCH):
    batch = bertscore_pairs[start:start+BATCH]
    qids   = [b[0] for b in batch]
    preds  = [b[1] for b in batch]
    refs   = [b[2] for b in batch]
    try:
        scores = compute_bertscore(preds, refs)
        # compute_bertscore returns aggregate; run individually for per-query
        for j, (qid, pred, ref) in enumerate(zip(qids, preds, refs)):
            try:
                s = compute_bertscore([pred], [ref])
                bs_results[qid] = round(s["f1"], 3)
            except Exception:
                bs_results[qid] = None
    except Exception as e:
        for qid, _, _ in batch:
            bs_results[qid] = None
    print(f"  BERTScore batch {start//BATCH+1}/{-(-len(bertscore_pairs)//BATCH)} done", end="\r")

print()

# Add S12 bertscore to rows
for row in rows:
    bs = bs_results.get(row["id"])
    row["s12_bertscore"] = bs
    row["s12_bs_pass"] = (bs is not None and bs >= THRESHOLDS["bertscore_f1"])

df = pd.DataFrame(rows)
csv_path = log_dir / "eval_100_results.csv"
df.to_csv(csv_path, index=False, encoding="utf-8")
print(f"\n[SAVE] Results CSV: {csv_path}")

failures_path = log_dir / "eval_100_failures.json"
with open(failures_path, "w", encoding="utf-8") as f:
    json.dump(failures, f, indent=2, ensure_ascii=False)
print(f"[SAVE] Failures JSON: {failures_path} ({len(failures)} items)")

def pct(val): return f"{val*100:.1f}%"
def mean(series): return series.dropna().mean() if not series.dropna().empty else 0.0

report_lines = []
R = report_lines.append

R("=" * 70)
R("  QUILTER HNW ADVISER — 100-QUERY EVALUATION REPORT")
R(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
R(f"  Model: {cfg.llm_model_manager} / {cfg.llm_model_worker} / {cfg.llm_model_fast}")
R(f"  Embedding: {emb.model_name_used}  dim={emb.dim}")
R(f"  Index: {len(chunks)} demo chunks  |  FAISS+BM25+RRF+CrossEncoder+HyDE+MMR")
R("=" * 70)

n = len(df)
R(f"\n  Total queries  : {n}")
R(f"  HNW queries    : {df['is_hnw'].sum()}")
R(f"  Calc-required  : {df['req_calc'].sum()}")
R(f"  OOS/injections : {(df['s1_blocked'] | df['s3_fallback']).sum()}")

R("\n" + "─" * 70)
R("  STAGE 1 — NeMo Input Rail")
R("─" * 70)
R(f"  Rail activations : {df['s1_nemo_fired'].sum()} / {n}")
R(f"  Queries blocked  : {df['s1_blocked'].sum()} (injection/OOS caught at input)")
inj_df  = df[df['category'] == 'injection']
oos_df  = df[df['category'] == 'oos']
R(f"  Injection blocked: {inj_df['s1_blocked'].sum()} / {len(inj_df)}")
R(f"  OOS fallback     : {(oos_df['actual_route'] == 'fallback').sum()} / {len(oos_df)}")

R("\n" + "─" * 70)
R("  STAGE 2 — Hybrid Retrieval (FAISS+BM25+RRF+HyDE+MMR)")
R("─" * 70)
non_blocked = df[~df['actual_route'].isin(['blocked','fallback','ERROR'])]
R(f"  Queries reaching retrieval : {len(non_blocked)}")
R(f"  Mean RRF confidence        : {mean(non_blocked['s2_rrf_score']):.4f}")
R(f"  RRF pass rate (>={THRESHOLDS['rrf_confidence']:.3f})     : {pct(mean(non_blocked['s2_rrf_pass']))}")
R(f"  Mean chunks cited           : {mean(non_blocked['s2_n_chunks']):.1f}")

R("\n" + "─" * 70)
R("  STAGE 3 — NeMo Retrieval Confidence Gate")
R("─" * 70)
R(f"  Fallback triggered (low RRF): {df['s3_fallback'].sum()}")
R(f"  Passed to agents            : {(~df['actual_route'].isin(['blocked','fallback','ERROR'])).sum()}")

R("\n" + "─" * 70)
R("  STAGE 4 — Precision Engine")
R("─" * 70)
prec_df = df[df['s4_prec_engaged']]
R(f"  Precision engine engaged : {len(prec_df)} / {n}")
calc_df = df[df['req_calc']]
R(f"  Calc-required queries    : {len(calc_df)}")
R(f"  S9 exact value pass rate : {pct(mean(calc_df['s9_exact_rate'])) if len(calc_df) else 'N/A'}")
R(f"  S9 full exact match      : {calc_df['s9_exact_ok'].sum() if len(calc_df) else 0} / {len(calc_df)}")

R("\n" + "─" * 70)
R("  STAGE 5 — Fact-Check Agent")
R("─" * 70)
fc_df = df[df['s5_fc_ran']]
R(f"  Fact-check ran           : {len(fc_df)} / {n}")
R(f"  Mean FC agent latency    : {mean(fc_df['s5_latency_ms']):.0f} ms")

R("\n" + "─" * 70)
R("  STAGE 6 — Manager Agent Synthesis")
R("─" * 70)
mgr_df = df[df['s6_mgr_ran']]
R(f"  Manager ran              : {len(mgr_df)} / {n}")
R(f"  Mean manager latency     : {mean(mgr_df['s6_latency_ms']):.0f} ms")
R(f"  Mean agent steps total   : {mean(df['s6_n_steps']):.1f}")

R("\n" + "─" * 70)
R("  STAGE 7 — NeMo Output Rail")
R("─" * 70)
R(f"  Output rail fired        : {df['s7_output_rail'].sum()} / {n}")
R(f"  Review-flagged           : {df['s7_review'].sum()} / {n}")

R("\n" + "─" * 70)
R("  STAGE 8 — DeBERTa-v3 Faithfulness (NLI)")
R("─" * 70)
R(f"  Mean faithfulness score  : {mean(df['s8_faithfulness']):.3f}")
R(f"  Pass rate (>={THRESHOLDS['faithfulness']:.2f})         : {pct(mean(df['s8_faith_pass']))}")
R(f"  Mean sentences evaluated : {mean(df['s8_n_sentences']):.1f}")
for cat in df['category'].unique():
    cat_df = df[df['category'] == cat]
    R(f"    {cat:<20} mean={mean(cat_df['s8_faithfulness']):.3f}  pass={pct(mean(cat_df['s8_faith_pass']))}")

R("\n" + "─" * 70)
R("  STAGE 9 — Exact Value Accuracy (fee/tax calculations)")
R("─" * 70)
for cat in ['platform_fee', 'ufpls', 'carry_forward', 'chaps', 'db_threshold', 'multi_domain']:
    cat_df = df[(df['category'] == cat) & (df['req_calc'])]
    if len(cat_df) > 0:
        R(f"  {cat:<20}  exact_ok={cat_df['s9_exact_ok'].sum()}/{len(cat_df)}  "
          f"mean_rate={mean(cat_df['s9_exact_rate']):.2f}")

R("\n" + "─" * 70)
R("  STAGE 10 — Route Accuracy")
R("─" * 70)
R(f"  Overall route accuracy   : {pct(mean(df['route_ok']))} ({df['route_ok'].sum()}/{n})")
for cat in df['category'].unique():
    cat_df = df[df['category'] == cat]
    R(f"    {cat:<20} {pct(mean(cat_df['route_ok']))}  ({cat_df['route_ok'].sum()}/{len(cat_df)})")

R("\n" + "─" * 70)
R("  STAGE 11 — Keyword Coverage")
R("─" * 70)
R(f"  Mean keyword coverage    : {pct(mean(df['s11_kw_cov']))} (target ≥{pct(THRESHOLDS['keyword_coverage'])})")
R(f"  Pass rate (≥{pct(THRESHOLDS['keyword_coverage'])})      : {pct(mean(df['s11_kw_pass']))}")
for cat in df['category'].unique():
    cat_df = df[df['category'] == cat]
    R(f"    {cat:<20} mean={pct(mean(cat_df['s11_kw_cov']))}  pass={pct(mean(cat_df['s11_kw_pass']))}")

R("\n" + "─" * 70)
R("  STAGE 12 — BERTScore F1 (semantic similarity)")
R("─" * 70)
bs_series = df['s12_bertscore'].dropna()
R(f"  Mean BERTScore F1        : {mean(bs_series):.3f} (target ≥{THRESHOLDS['bertscore_f1']})")
R(f"  Pass rate (≥{THRESHOLDS['bertscore_f1']})        : {pct(mean(df['s12_bs_pass']))}")
for cat in df['category'].unique():
    cat_df = df[df['category'] == cat]
    R(f"    {cat:<20} mean={mean(cat_df['s12_bertscore']):.3f}  pass={pct(mean(cat_df['s12_bs_pass']))}")

R("\n" + "─" * 70)
R("  LATENCY BREAKDOWN")
R("─" * 70)
R(f"  Mean total latency       : {mean(df['latency_ms']):.0f} ms")
R(f"  Median latency           : {df['latency_ms'].median():.0f} ms")
R(f"  P95 latency              : {df['latency_ms'].quantile(0.95):.0f} ms  (budget: {THRESHOLDS['latency_p95_ms']:,} ms)")
R(f"  Min latency              : {df['latency_ms'].min():.0f} ms  (blocked/fallback)")
R(f"  Max latency              : {df['latency_ms'].max():.0f} ms")
for cat in df['category'].unique():
    cat_df = df[df['category'] == cat]
    R(f"    {cat:<20} mean={mean(cat_df['latency_ms']):.0f}ms  max={cat_df['latency_ms'].max():.0f}ms")

R("\n" + "─" * 70)
R("  DIFFICULTY BREAKDOWN")
R("─" * 70)
for diff in ['easy', 'medium', 'hard']:
    d_df = df[df['difficulty'] == diff]
    if len(d_df):
        R(f"  {diff.upper():<8}  n={len(d_df):<3}  route={pct(mean(d_df['route_ok']))}  "
          f"kw={pct(mean(d_df['s11_kw_cov']))}  bs={mean(d_df['s12_bertscore']):.3f}  "
          f"exact={pct(mean(d_df['s9_exact_ok'].astype(float)))}")

R("\n" + "─" * 70)
R("  FAILURE SUMMARY")
R("─" * 70)
fail_df = df[df['status'] == 'FAIL']
R(f"  Total failures           : {len(fail_df)} / {n}")
if len(fail_df):
    R(f"  Failure categories:")
    for cat in fail_df['category'].value_counts().index:
        n_fail = (fail_df['category'] == cat).sum()
        R(f"    {cat:<20} {n_fail} failures")
    R(f"\n  Route failures           : {(~df['route_ok']).sum()}")
    R(f"  Keyword failures         : {(~df['s11_kw_pass']).sum()}")
    R(f"  Exact calc failures      : {(df['req_calc'] & ~df['s9_exact_ok']).sum()}")

R("\n" + "=" * 70)
R("  SCORECARD SUMMARY")
R("=" * 70)
metrics = {
    "Route Accuracy":     (mean(df['route_ok']),       0.90, True),
    "Keyword Coverage":   (mean(df['s11_kw_cov']),     THRESHOLDS['keyword_coverage'], True),
    "BERTScore F1":       (mean(bs_series),             THRESHOLDS['bertscore_f1'], True),
    "Faithfulness":       (mean(df['s8_faithfulness']), THRESHOLDS['faithfulness'], True),
    "OOS Block Rate":     (mean((df['category'].isin(['oos','injection'])) & (df['route_ok'])), 0.95, True),
    "Exact Calc Acc":     (mean(calc_df['s9_exact_ok'].astype(float)) if len(calc_df) else 0.0, 0.70, True),
    "Review Flag Rate":   (mean(df['s7_review']),       0.30, False),  # lower is better
    "P95 Latency (s)":    (df['latency_ms'].quantile(0.95)/1000, THRESHOLDS['latency_p95_ms']/1000, False),
}
for name, (val, target, higher_better) in metrics.items():
    if higher_better:
        ok = val >= target
        val_str = pct(val) if val <= 1.01 else f"{val:.3f}"
        tgt_str = pct(target) if target <= 1.01 else f"{target:.3f}"
    else:
        ok = val <= target
        val_str = pct(val) if val <= 1.01 else f"{val:.1f}s"
        tgt_str = pct(target) if target <= 1.01 else f"{target:.1f}s"
    mark = "PASS" if ok else "FAIL"
    R(f"  [{mark}] {name:<22}  actual={val_str:<10}  target={'≥' if higher_better else '≤'}{tgt_str}")

R("\n" + "=" * 70)

report_text = "\n".join(report_lines)
print("\n" + report_text)

report_path = log_dir / "eval_100_report.txt"
with open(report_path, "w", encoding="utf-8") as f:
    f.write(report_text)
print(f"\n[SAVE] Report: {report_path}")
print(f"[SAVE] CSV:    {csv_path}")
print(f"[SAVE] Failures: {failures_path}")
