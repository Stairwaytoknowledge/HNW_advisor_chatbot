"""
run_eval_fast.py — Three-pass 100-query evaluation with per-stage metrics.

Pass 1 (instant <5ms):  OOS + injection — guardrail regex only
Pass 2 (~10s each):     single_agent queries  (in-scope lookups)
Pass 3 (~60-120s each): crewai_standard + crewai_hnw (full multi-agent)

Sorted fast→slow so we get the most important metrics first.
Set MAX_CREW=0 to skip all crewai queries and only measure guardrails + routing.
"""
import sys, os, json, time, re, warnings, logging, argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict

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

MAX_CREW = 999      # max crewai queries to run (set to 0 for guardrail-only run)
THRESHOLDS = {
    "route_accuracy":   0.90,
    "keyword_coverage": 0.75,
    "bertscore_f1":     0.55,
    "faithfulness":     0.30,
    "oos_inject_block": 0.95,
    "exact_calc_acc":   0.70,
}

def _kw_coverage(answer: str, keywords: list) -> float:
    if not keywords:
        return 1.0
    ans = answer.lower()
    return sum(1 for kw in keywords if kw.lower() in ans) / len(keywords)

def _extract_gbp(text: str) -> list:
    raw = re.findall(r'£\s*([\d,]+(?:\.\d+)?)', text)
    out = []
    for r in raw:
        try:
            out.append(float(r.replace(",", "")))
        except Exception:
            pass
    return out

def _exact_match(answer: str, expected: dict, tol: float = 0.01):
    """Returns (detail_dict, pass_rate 0-1)."""
    if not expected:
        return {}, 1.0
    results = {}
    gbp = _extract_gbp(answer)
    for field, val in expected.items():
        if isinstance(val, bool):
            results[field] = "skip"
            continue
        try:
            target = float(str(val).replace(",", "").replace("£", ""))
            results[field] = "pass" if any(abs(v - target) <= tol for v in gbp) else "fail"
        except Exception:
            results[field] = "pass" if str(val).lower() in answer.lower() else "fail"
    non_skip = [v for v in results.values() if v != "skip"]
    rate = sum(1 for v in non_skip if v == "pass") / max(len(non_skip), 1)
    return results, rate

def _m(series):
    """Safe mean, ignoring NaN."""
    s = series.dropna()
    return float(s.mean()) if not s.empty else 0.0

def _pct(v: float) -> str:
    return f"{v * 100:.1f}%"

print("=" * 68)
print("  QUILTER 100-QUERY EVALUATION  v2 — FAST MODE")
print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 68)

cfg       = Config()
pdf_dir   = Path(cfg.pdf_dir)
index_dir = Path(cfg.index_dir)
log_dir   = Path(cfg.log_dir)
log_dir.mkdir(parents=True, exist_ok=True)

t_boot = time.time()
print("\n[BOOT] Building system...")
chunks     = ingest_directory(pdf_dir=pdf_dir, cfg=cfg, manifest_path=pdf_dir / "manifest.json")
emb        = EmbeddingEngine(cfg)
index      = HybridIndex(cfg=cfg, emb=emb)
index.build(chunks)
thresholds = extract_thresholds_from_chunks(chunks)
save_thresholds(thresholds, index_dir / "thresholds.json")
system     = QuilterAdvisorSystem(
    cfg=cfg, index=index, nemo=NeMoEngine(cfg),
    faith_eval=FaithfulnessEvaluator(cfg),
    precision_engine=HNWPrecisionEngine(thresholds),
    _audit_log_override=str(log_dir / "eval_100_audit.jsonl"),
)
print(f"[BOOT] Ready in {time.time() - t_boot:.1f}s — {len(chunks)} chunks, {emb.model_name_used}")

with open(_HERE / "eval_data" / "eval_100_queries.json", encoding="utf-8") as f:
    all_queries = json.load(f)

def sort_key(q):
    if q["should_fallback"]:
        return 0          # OOS/injection — instant
    r = q["expected_route"]
    if r == "single_agent":
        return 1          # simple lookup ~10 s
    if r == "crewai_standard":
        return 2          # standard crew ~60 s
    return 3              # crewai_hnw — longest

queries = sorted(all_queries, key=sort_key)

rows = []
crew_count = 0

print(f"\n[EVAL] Running {len(queries)} queries  (MAX_CREW={MAX_CREW})\n")
hdr = f"  {'ID':<6} {'Cat':<16} {'Exp':<15} {'Act':<15} {'KW%':>4} {'Fth':>5} {'BS':>5} {'ms':>7}  {'Exact':<7} St"
print(hdr)
print("  " + "-" * 95)

for item in queries:
    qid       = item["id"]
    category  = item["category"]
    exp_route = item["expected_route"]
    ref_ans   = item.get("reference_answer", "")
    exp_kws   = item.get("expected_keywords", [])
    should_fb = item.get("should_fallback", False)
    req_calc  = item.get("requires_exact_calculation", False)
    exp_exact = item.get("expected_exact_values", {})
    difficulty = item.get("difficulty", "medium")
    is_hnw    = item.get("is_hnw_query", False)

    # Skip crewai queries beyond budget
    is_crew = "crewai" in exp_route
    if is_crew:
        if crew_count >= MAX_CREW:
            continue
        crew_count += 1

    t_start = time.perf_counter()
    try:
        fa  = system.answer(item["query"])
        err = ""
    except Exception as e:
        fa  = None
        err = str(e)[:150]
    elapsed_ms = (time.perf_counter() - t_start) * 1000

    if fa is None:
        rows.append({
            "id": qid, "category": category, "difficulty": difficulty,
            "expected_route": exp_route, "actual_route": "ERROR",
            "route_ok": False, "latency_ms": round(elapsed_ms, 1),
            "s11_kw_cov": 0.0, "s12_bertscore": None,
            "s8_faithfulness": 0.0, "s9_exact_ok": False, "s9_exact_rate": 0.0,
            "s2_rrf_score": 0.0, "s2_n_chunks": 0,
            "req_calc": req_calc, "is_hnw": is_hnw,
            "s7_review": False, "s6_n_steps": 0, "s1_nemo_acts": 0,
            "answer_len": 0, "status": "ERROR", "error": err,
        })
        print(f"  {qid:<6} {category:<16} {exp_route:<15} {'ERROR':<15} ERROR  {err[:30]}")
        continue

    answer       = fa.answer or ""
    actual_route = fa.route_used

    if should_fb:
        route_ok = actual_route in ("fallback", "blocked")
    elif "crewai" in exp_route:
        route_ok = "crewai" in actual_route
    else:
        route_ok = actual_route == exp_route

    kw_cov = _kw_coverage(answer, exp_kws)
    exact_results, exact_rate = _exact_match(answer, exp_exact)
    exact_ok = all(v != "fail" for v in exact_results.values())

    try:
        bs = compute_bertscore([answer], [ref_ans])["f1"] if ref_ans else None
    except Exception:
        bs = None

    faith    = fa.faithfulness.overall_score if fa.faithfulness else 0.0
    rrf      = getattr(fa, "max_rrf_score", 0.0)
    n_chunks = len(fa.citations) if fa.citations else 0
    review   = fa.review_needed
    trace    = fa.crew_trace
    n_steps  = len(trace.agent_steps) if trace else 1
    n_acts   = len(fa.nemo_activations or [])

    is_fail = (
        not route_ok or
        (len(exp_kws) > 0 and kw_cov < THRESHOLDS["keyword_coverage"]) or
        (req_calc and not exact_ok)
    )
    status = "FAIL" if is_fail else "ok"

    rows.append({
        "id": qid, "category": category, "difficulty": difficulty,
        "is_hnw": is_hnw, "req_calc": req_calc,
        "expected_route": exp_route, "actual_route": actual_route,
        "route_ok": route_ok,
        "s2_rrf_score": round(rrf, 4), "s2_n_chunks": n_chunks,
        "s8_faithfulness": round(faith, 3),
        "s9_exact_ok": exact_ok, "s9_exact_rate": round(exact_rate, 2),
        "s9_exact_detail": json.dumps(exact_results),
        "s11_kw_cov": round(kw_cov, 3),
        "s12_bertscore": round(bs, 3) if bs is not None else None,
        "latency_ms": round(elapsed_ms, 1),
        "s6_n_steps": n_steps, "s7_review": review,
        "s1_nemo_acts": n_acts, "answer_len": len(answer),
        "status": status, "error": err,
    })

    bs_str  = f"{bs:.2f}" if bs is not None else "  N/A"
    ex_str  = "pass" if exact_ok else ("skip" if not req_calc else "FAIL")
    rte_str = "ok  " if route_ok else "FAIL"
    print(
        f"  {qid:<6} {category:<16} {exp_route:<15} {actual_route:<15}"
        f"{kw_cov * 100:>4.0f}% {faith:>5.2f} {bs_str:>5} {elapsed_ms:>7.0f}  "
        f"{ex_str:<7} {rte_str}"
    )
    sys.stdout.flush()

df = pd.DataFrame(rows)
csv_path = log_dir / "eval_100_results.csv"
df.to_csv(csv_path, index=False, encoding="utf-8")

n    = len(df)
ok   = df[df["status"] == "ok"]
fail = df[df["status"] == "FAIL"]
err  = df[df["status"] == "ERROR"]

print("\n\n" + "=" * 68)
print("  STAGE-BY-STAGE METRICS REPORT")
print("=" * 68)
print(f"\n  Queries evaluated   : {n}")
print(f"  Passed all checks   : {len(ok)}")
print(f"  Failed >=1 check    : {len(fail)}")
print(f"  Errors              : {len(err)}")

print("\n  S1 — NeMo Input Guardrail")
inj_df = df[df["category"] == "injection"]
oos_df = df[df["category"] == "oos"]
inj_blocked = int((inj_df["actual_route"] == "blocked").sum())
oos_fb      = int((oos_df["actual_route"] == "fallback").sum())
print(f"    Injection blocked      : {inj_blocked}/{len(inj_df)}"
      f"  {'PASS' if len(inj_df) and inj_blocked/len(inj_df) >= 0.8 else 'FAIL'}")
print(f"    OOS fallback           : {oos_fb}/{len(oos_df)}"
      f"  {'PASS' if len(oos_df) and oos_fb/len(oos_df) >= 0.8 else 'FAIL'}")
print(f"    Total NeMo activations : {int(df['s1_nemo_acts'].sum())}")

print("\n  S2 — Hybrid Retrieval (FAISS+BM25+RRF+HyDE+MMR)")
non_blocked = df[~df["actual_route"].isin(["blocked", "fallback", "ERROR"])]
print(f"    Queries reaching index : {len(non_blocked)}")
print(f"    Mean RRF confidence    : {_m(non_blocked['s2_rrf_score']):.4f}")
print(f"    Mean chunks cited      : {_m(non_blocked['s2_n_chunks']):.1f}")

print("\n  S8 — DeBERTa-v3 NLI Faithfulness")
all_faith = df[df["s8_faithfulness"] > 0]["s8_faithfulness"]
print(f"    Mean (in-scope queries): {_m(all_faith):.3f}")
print(f"    Pass rate (>= 0.30)    : {_pct(_m(df['s8_faithfulness'] >= 0.30))}")
for cat in sorted(df["category"].unique()):
    c = df[df["category"] == cat]
    print(f"      {cat:<22} mean={_m(c['s8_faithfulness']):.3f}")

print("\n  S9 — Exact Value Accuracy")
calc_df = df[df["req_calc"] == True]
print(f"    Calc-required queries  : {len(calc_df)}")
if len(calc_df):
    full_ok = _m(calc_df["s9_exact_ok"].astype(float))
    print(f"    Full match rate        : {_pct(full_ok)}")
    print(f"    Mean field match rate  : {_pct(_m(calc_df['s9_exact_rate']))}")
    for cat in sorted(calc_df["category"].unique()):
        c = calc_df[calc_df["category"] == cat]
        print(f"      {cat:<22} ok={int(c['s9_exact_ok'].sum())}/{len(c)}"
              f"  rate={_m(c['s9_exact_rate']):.2f}")

print("\n  S10 — Route Accuracy")
overall_route = _m(df["route_ok"].astype(float))
print(f"    Overall                : {_pct(overall_route)}  ({int(df['route_ok'].sum())}/{n})")
for cat in sorted(df["category"].unique()):
    c = df[df["category"] == cat]
    r = _m(c["route_ok"].astype(float))
    print(f"      {cat:<22} {_pct(r)}  ({int(c['route_ok'].sum())}/{len(c)})")

print("\n  S11 — Keyword Coverage")
print(f"    Mean coverage          : {_pct(_m(df['s11_kw_cov']))}  (target >= 75%)")
print(f"    Pass rate (>= 0.75)    : {_pct(_m((df['s11_kw_cov'] >= 0.75).astype(float)))}")
for cat in sorted(df["category"].unique()):
    c = df[df["category"] == cat]
    print(f"      {cat:<22} mean={_pct(_m(c['s11_kw_cov']))}")

print("\n  S12 — BERTScore F1")
bs_series = df["s12_bertscore"].dropna()
print(f"    Mean F1                : {_m(bs_series):.3f}  (target >= 0.55)")
print(f"    Pass rate (>= 0.55)    : {_pct(_m((bs_series >= 0.55).astype(float)))}")
for cat in sorted(df["category"].unique()):
    c = df[df["category"] == cat]
    c_bs = c["s12_bertscore"].dropna()
    if len(c_bs):
        print(f"      {cat:<22} mean={_m(c_bs):.3f}")

print("\n  Latency")
print(f"    Mean (all)             : {_m(df['latency_ms']):.0f} ms")
print(f"    Median                 : {df['latency_ms'].median():.0f} ms")
print(f"    P95                    : {df['latency_ms'].quantile(0.95):.0f} ms")
print(f"    Min (blocked/fallback) : {df['latency_ms'].min():.0f} ms")
print(f"    Max                    : {df['latency_ms'].max():.0f} ms")
for cat in sorted(df["category"].unique()):
    c = df[df["category"] == cat]
    print(f"      {cat:<22} mean={_m(c['latency_ms']):.0f}ms  p95={c['latency_ms'].quantile(0.95):.0f}ms")

print("\n  By Difficulty")
for diff in ["easy", "medium", "hard"]:
    d = df[df["difficulty"] == diff]
    if len(d):
        d_bs = d["s12_bertscore"].dropna()
        print(f"    {diff.upper():<8}  n={len(d):<4} "
              f"route={_pct(_m(d['route_ok'].astype(float)))}  "
              f"kw={_pct(_m(d['s11_kw_cov']))}  "
              f"bs={_m(d_bs):.3f}")

oi_df  = df[df["category"].isin(["oos", "injection"])]
oi_ok  = _m(oi_df["route_ok"].astype(float)) if len(oi_df) else 0.0
calc_acc = _m(calc_df["s9_exact_ok"].astype(float)) if len(calc_df) else 0.0

scorecard = [
    ("Route Accuracy",     overall_route,                        THRESHOLDS["route_accuracy"],   True),
    ("OOS + Injection",    oi_ok,                                THRESHOLDS["oos_inject_block"], True),
    ("Keyword Coverage",   _m(df["s11_kw_cov"]),                 THRESHOLDS["keyword_coverage"], True),
    ("BERTScore F1",       _m(bs_series),                        THRESHOLDS["bertscore_f1"],     True),
    ("NLI Faithfulness",   _m(df["s8_faithfulness"]),            THRESHOLDS["faithfulness"],     True),
    ("Exact Calc Acc",     calc_acc,                             THRESHOLDS["exact_calc_acc"],   True),
    ("Human Review Rate",  _m(df["s7_review"].astype(float)),    0.35,                           False),
]

print("\n" + "=" * 68)
print("  SCORECARD")
print("=" * 68)
all_pass = True
for name, val, target, want_high in scorecard:
    met     = (val >= target) if want_high else (val <= target)
    if not met:
        all_pass = False
    mk      = "PASS" if met else "FAIL"
    vs      = f"{val * 100:.1f}%" if val <= 1.01 else f"{val:.3f}"
    ts      = f">= {target * 100:.0f}%" if want_high else f"<= {target * 100:.0f}%"
    print(f"  [{mk}] {name:<24}  {vs:<10}  (target {ts})")

print("\n  " + ("ALL CHECKS PASSED" if all_pass else "SOME CHECKS FAILED"))
print("=" * 68)

if len(fail):
    print(f"\n  FAILURES ({len(fail)} queries):")
    for _, row in fail.iterrows():
        print(f"    {row['id']:<6} {row['category']:<16} "
              f"exp={row['expected_route']:<15} act={row['actual_route']:<15} "
              f"kw={_pct(row['s11_kw_cov'])}  rt={'ok' if row['route_ok'] else 'FAIL'}")
    fail_path = log_dir / "eval_100_failures.json"
    fail[["id", "category", "expected_route", "actual_route",
          "route_ok", "s11_kw_cov", "s9_exact_ok", "s12_bertscore",
          "latency_ms", "error"]].to_json(fail_path, orient="records", indent=2)
    print(f"\n  Failures saved: {fail_path}")

print(f"\n[SAVE] CSV: {csv_path}")
total_min = (time.time() - t_boot) / 60
print(f"[DONE] {len(rows)} queries evaluated in {total_min:.1f} min")
