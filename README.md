Quilter HNW Adviser Assistant

1. Multi-agent RAG system for Quilter financial advisers 
answers precise operational questions about platform fees, pension rules, KYC, and compliance
Escalates gracefully to the Contact Centre when uncertain. 
No financial advice is given


2. Architecture


Input Query by the user goes through the following flowchart process
-> NeMo Input Rail (OOS/Injection -> Contact Centre fallback) 
-> L1 Token Importance (leave-one-out attribution)
-> Precision Type Detector (platform_fee/db_threshold/mpaa/chaps/ufpls/...)
-> Hybrid Retrieval (FAISS+BM25+RRF+CrossEncoder+HyDE+MMR)
-> NeMo Retrieval Rail (low confidence -> Contact Centre fallback)
-> CrewAI Multi-Agent Pipeline containing
    1. Retrieval Agent (llama3.2:3b —> fast)
    2. Precision Agent (qwen2.5:7b)
    3. Compliance Agent (qwen2.5:7b)
    4. Fact-Check Agent (llama3.2:3b —> on DRAFT)
    5. Manager Agent (qwen2.5:14b —> sees FC output)
-> NeMo Output Rail (citation/precision fail -> Contact Centre)
-> L3 Sentence NLI Faithfulness (DeBERTa-v3)
-> Audit Logging (audit_log·crew_trace·sentence_attribution·nemo_rail_log)
-> FinalAnswer (answer+route+faithfulness+token_importance+doc_versions)

3. Three-layer explainability
L1-> Token importance (retrieval drivers)
L2-> Agent trace (per-agent reasoning + tool calls)
L3-> Sentence NLI faithfulness breakdown

4. Seven-layer defence
a. NeMo input rail (OOS + injection)
b. RRF confidence gate
c. Precision Engine (exact arithmetic)
d. Constitutional prompt (temp=0)
e. NeMo output rail (citation + precision)
f. Fact-Check Agent (on draft before Manager)
g. Human review queue (faithfulness<0.5 or flags)


5. Prerequisites

Requirement 
Python 3.12 
Ollama ≥ 0.3
RAM ≥ 16 GB For qwen2.5:14b 4-bit quantised (10 GB) (localized implementation)
Disk ≥ 30 GB For Models + index + logs



6. Start

a. 1. Clone / unzip the project
```powershell
cd C:\Code\quilter
```

b. Run the setup script (PowerShell, as Administrator)
```powershell
.\setup.ps1
```
This will:
Create .venv and install all Python dependencies
Create required directories (quilter_docs/, index_v3/, logs_v3/, etc.)
Pull three Ollama models: qwen2.5:14b, qwen2.5:7b, llama3.2:3b
Pre-warm distilbert-base-uncased (BERTScore) and sentence-transformers downloads

c. Activate the virtual environment

```powershell
.\.venv\Scripts\Activate.ps1
```

d. (Optional) Sync Quilter PDFs

```powershell
python download_quilter_docs.py
```

If Quilter URLs are not accessible, the system uses a built-in demo corpus(using already downloaded docs) and continues normally.

e. Launch notebook

```powershell
jupyter notebook quilter_hnw_advisor_v3.ipynb

Run cells top to bottom. Cell 1 verifies Ollama health before anything else runs.


7. File Structure


C:\Code\quilter\
├── .venv\                          # Python virtual environment (created by setup.ps1)
├── quilter_docs\                   # Downloaded Quilter PDFs
│   └── manifest.json               # SHA256 manifest for change detection
├── index_v3\                       # FAISS index + BM25 state
│   ├── hybrid_index.pkl
│   └── thresholds.json             # Regulatory values extracted at index time
├── logs_v3\
│   ├── audit_log.jsonl             # Scalar per-query record (7-year retention)
│   ├── crew_trace.jsonl            # Full agent trace per query
│   ├── sentence_attribution.jsonl  # Per-sentence NLI labels
│   ├── nemo_rail_log.jsonl         # Rail activations
│   ├── update_log.jsonl            # PDF version changes
│   └── compare_log.jsonl           # A/B config comparison (isolated)
├── rails\
│   ├── main.co                     # Colang: OOS + injection rails
│   ├── hnw_rails.co                # Colang: HNW routing + precision rails
│   └── config.yml                  # NeMo config: Ollama backend (qwen2.5:7b)
├── eval_data\
│   ├── gold_eval_set.json          # 30 annotated queries with reference answers
│   └── oos_eval_set.json           # 20 OOS + 10 in-scope queries for F1
├── src\
│   ├── config.py                   # Config dataclass — all tuneable parameters
│   ├── models.py                   # Shared dataclasses (Chunk, FinalAnswer, ...)
│   ├── thresholds_store.py         # Regulatory constant extraction (RISK-01 fix)
│   ├── llm_client.py               # Unified Ollama client with model wrappers
│   ├── pdf_ingestion.py            # fitz PDF ingestion + sliding-window chunker
│   ├── embedding.py                # BGE-large embedding + L1 token attribution
│   ├── retrieval.py                # FAISS + BM25 + RRF + CrossEncoder + HyDE + MMR
│   ├── precision_engine.py         # HNW exact arithmetic (fee / DB / MPAA / CHAPS / UFPLS)
│   ├── guardrails.py               # NeMo Engine (real + Python fallback)
│   ├── faithfulness.py             # DeBERTa-v3 NLI sentence faithfulness
│   ├── agents.py                   # CrewAI agent functions (5 agents)
│   ├── orchestrator.py             # QuilterAdvisorSystem — main pipeline
│   ├── evaluation.py               # BERTScore · Recall@K · OOS F1 · Conflicts
│   ├── monitoring.py               # Dashboard + audit lookup
│   └── display.py                  # Rich console renderer for FinalAnswer
├── download_quilter_docs.py        # SHA256-tracked PDF sync from Quilter's site
├── requirements.txt
├── setup.ps1
└── quilter_hnw_advisor_v3.ipynb   # Thin shell — all logic in src/




8.Configuration

All parameters are in src/config.py. Key settings are as follows

Parameter,Default,Description in that order

->llm_model_manager ->qwen2.5:14b-> Manager Agent LLM 
-> llm_model_worker -> qwen2.5:7b-> Precision/Compliance/NeMo LLM 
->llm_model_fast -> llama3.2:3b-> Retrieval/FactCheck LLM 
->use_hyde-> True-> Enable Hypothetical Document Embedding 
->mmr_lambda -> 0.7-> MMR relevance/diversity balance (0=diverse, 1=relevant) 
->rrf_contact_centre_threshold-> 0.015 ->Min RRF score before fallback 
->nli_faithfulness_threshold ->0.50 -> Min faithfulness before review flag 
->chunk_size-> 400-> Tokens per chunk 
->chunk_overlap-> 80-> Token overlap between adjacent chunks 
->top_k_retrieval-> 20-> Candidates before reranking 
->top_k_final->5->Chunks passed to agents 


9.Evaluation->Run full evaluation suite

```python
from src.evaluation import load_gold_set, load_oos_set, run_eval, print_scorecard
from src.evaluation import recall_at_k, oos_detector_f1, detect_cross_document_conflicts

gold_set = load_gold_set(cfg.eval_data_dir)
oos_set  = load_oos_set(cfg.eval_data_dir)
df_eval  = run_eval(system, gold_set, verbose=True)
print_scorecard(df_eval, system)
```

10. Performance check

Metric -> Target -> Notes 

-> BERTScore F1 -> >= 0.65 -> Primary quality metric 
-> Route accuracy -> >= 90% -> crew vs single_agent vs fallback 
-> Recall@5 -> >= 0.70 -> Requires annotated relevant_chunk_ids
-> OOS F1 -> ≥ 0.85 -> On 20 OOS + 10 in-scope queries 
-> Fallback rate -> <= 30% -> Fraction sent to Contact Centre 
-> P95 latency -> <= 8s-> End-to-end including all agents 

11. Annotate gold set for Recall@K

After first index build, run Notebook Cell 15 to populate relevant_chunk_ids

```python
from src.evaluation import annotate_gold_set
annotate_gold_set(system, gold_set, eval_data_dir=cfg.eval_data_dir)
```


12. Compliance -> Look up all artefacts for a query

```python
from src.monitoring import MonitoringDashboard
dash = MonitoringDashboard(cfg)
dash.lookup_query(query_id="QRY-<timestamp>-<hash>")
```

Returns audit record,crew trace,sentence attributions,rail activations

13. Log file schema and examples

a. audit_log.jsonl -> one record per query:
```json
{
  "query_id": "QRY-20240115-ab12cd34",
  "query": "What is the platform fee on £500,000?",
  "route_used": "crewai_hnw",
  "answer_preview": "The platform fee on £500,000...",
  "latency_ms": 3241,
  "faithfulness_score": 0.87,
  "review_needed": false,
  "nemo_activations": [],
  "doc_versions": {"quilter_platform_guide.pdf": "sha256:a1b2c3..."},
  "timestamp": "2024-01-15T14:23:11"
}
```

b. crew_trace.jsonl -> one record per query (full agent trace)
```json
{
  "query_id": "QRY-20240115-ab12cd34",
  "agent_steps": [
    {"agent_name": "RetrievalAgent", "latency_ms": 312, "tool_calls": [...], "output": "..."},
    {"agent_name": "PrecisionAgent", "latency_ms": 891, "reasoning": "...", "output": "..."}
  ]
}
```

c. sentence_attribution.jsonl -> one record per sentence:
```json
{
  "query_id": "QRY-20240115-ab12cd34",
  "sentence_index": 0,
  "sentence_text": "The platform fee on £500,000 is £3,550.00 per annum.",
  "nli_label": "ENTAILMENT",
  "nli_confidence": 0.94,
  "source_chunk_id": "chunk-abc123",
  "source_file": "quilter_platform_guide.pdf",
  "source_page": 7,
  "supported": true
}
```



14. Troubleshooting

a.Ollama model not found
```powershell
ollama pull qwen2.5:14b
ollama pull qwen2.5:7b
ollama pull llama3.2:3b
```

15. FAISS import error on Windows**
```powershell
pip install faiss-cpu --extra-index-url https://download.pytorch.org/whl/cpu
```

16. NeMo Guardrails falls back to Python**

This is expected if Ollama is not running or the rails directory is missing. The system logs using_real_nemo=False and continues with the Python-regex fallback, which covers all OOS and injection patterns.

BERTScore first run is slow**

The distilbert-base-uncased model downloads on first use (~260 MB). Run setup.ps1 once to pre-warm.

17.qwen2.5:14bruns out of RAM

Edit `src/config.py` and change:
```python
llm_model_manager: str = "qwen2.5:7b"
```

18.relevant_chunk_ids` are all empty

Run Notebook Cell 15 (annotate_gold_set) after building the index for the first time. Recall@K returns `0.0` until annotation is complete.
