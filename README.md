# CodeBase Visualizer

> High-Performance Graph RAG & Code Intelligence Engine for Large-Scale Codebases.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg)](https://fastapi.tiangolo.com/)
[![Tree-sitter](https://img.shields.io/badge/AST-Tree--sitter-green.svg)](https://tree-sitter.github.io/)
[![NetworkX](https://img.shields.io/badge/Graph-NetworkX-orange.svg)](https://networkx.org/)
[![Qdrant](https://img.shields.io/badge/Vector%20DB-Qdrant-red.svg)](https://qdrant.tech/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Architecture Overview

**CodeBase Visualizer** is a dual-engine code understanding platform combining **AST-derived structural call graphs** with **dense vector embeddings** (Graph RAG) to eliminate hallucinations when explaining complex repository architectures.

```
                    Developer Query (Natural Language)
                                   │
                                   ▼
        ┌─────────────────────────────────────────────────────┐
        │        Stage 1: Dense Vector Retrieval (Qdrant)     │
        │  • all-MiniLM-L6-v2 (384-d Cosine Similarity)       │
        │  • Retrieves top-k semantically closest Anchors     │
        └──────────────────────────┬──────────────────────────┘
                                   │
                                   ▼
        ┌─────────────────────────────────────────────────────┐
        │      Stage 2: Graph Context Expansion (NetworkX)    │
        │  • 1-hop Caller (predecessor) & Callee (successor)  │
        │  • PageRank Centrality & Git Churn Metadata         │
        └──────────────────────────┬──────────────────────────┘
                                   │
                                   ▼
        ┌─────────────────────────────────────────────────────┐
        │       Structured Prompt Assembly (XML Budgeted)     │
        │  • <code_snippets>: Full source code for Anchors    │
        │  • <graph_neighbors>: Structural caller/callee tags │
        │  • <graph_topology>: ASCII call hierarchy summary   │
        └──────────────────────────┬──────────────────────────┘
                                   │
                                   ▼
        ┌─────────────────────────────────────────────────────┐
        │         Local LLM Inference (Ollama Orchestrator)   │
        │  • Streaming SSE (/stream) & Non-streaming (/query) │
        │  • Strict citation rule: [file_path::func::<name>]  │
        └──────────────────────────┬──────────────────────────┘
                                   │
                                   ▼
        ┌─────────────────────────────────────────────────────┐
        │    Anti-Hallucination Grounding Evaluation Engine   │
        │  • Grounded: Vector Anchor | Graph Neighbor         │
        │  • Hallucinated: Unverified citation                │
        │  • Telemetry: Adherence Score (0.0 – 1.0)           │
        └─────────────────────────────────────────────────────┘
```

---

## Hallucination Comparison Test: Graph Augmentation Ablation

To evaluate the causal efficacy of structural graph augmentation, we perform controlled **Ablation Testing** on our hand-labeled benchmark question set (`LABELED_ABLATION_DATASET`), comparing:
- **Condition A (Graph Augmentation ON)**: Dense vector retrieval expanded with 1-hop caller/callee neighbors and topology.
- **Condition B (Graph Augmentation OFF)**: Pure dense vector search anchors only (ablation baseline).

Both pipelines generate answers evaluated by `rag.llm_engine.evaluate_grounding()`.

### Benchmark Results Table

| Query ID | Architectural Focus | Primary Anchor | Graph Neighbors | Adherence (OFF) | Adherence (ON) | Grounding Delta | Hallucinations Prevented |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| `ABL-01` | Error Handling & Cleanup | `clone_repository` | `1` | 100.0% | 100.0% | 0.0% | 0 |
| `ABL-02` | Reverse Caller Tracing | `cleanup_repo_directory` | `1` | 100.0% | 100.0% | 0.0% | 0 |
| `ABL-03` | Pipeline Orchestration | `scan_repository` | `3` | 60.0% | 100.0% | +40.0% | **-2** |
| `ABL-04` | Dead Code Detection | `detect_dead_code` | `5` | 100.0% | 100.0% | 0.0% | 0 |
| `ABL-05` | Vector Ingestion Pipeline | `batch_upsert_chunks` | `1` | 100.0% | 100.0% | 0.0% | 0 |
| `ABL-06` | AST Structural Parsing | `extract_structure` | `2` | 100.0% | 100.0% | 0.0% | 0 |
| `ABL-07` | Graph Construction | `build_graph` | `3` | 50.0% | 100.0% | +50.0% | **-1** |
| `ABL-08` | Hybrid Search Retrieval | `hybrid_search` | `2` | 100.0% | 100.0% | 0.0% | 0 |
| `ABL-09` | Prompt Engineering & Grounding | `build_rag_prompt` | `6` | 100.0% | 100.0% | 0.0% | 0 |
| `ABL-10` | Storage Maintenance | `garbage_collect_temp_repos` | `6` | 100.0% | 100.0% | 0.0% | 0 |
| **MACRO** | **Overall Benchmark** | **10 Queries** | **3 Recovered** | **91.0%** | **100.0%** | **+9.0%** | **-3 Total** |

### Quantitative Ablation Summary

```
=========================================================================================================
  * Macro Grounding Adherence (Graph OFF) : 91.00%
  * Macro Grounding Adherence (Graph ON)  : 100.00%
  * Grounding Adherence Gain (Delta)      : +9.00%
  * Hallucination Rate (Graph OFF)        : 11.54%
  * Hallucination Rate (Graph ON)         : 0.00%
  * Relative Hallucination Reduction      : -100.00%
  * Total Hallucinations Detected (OFF)   : 3
  * Total Hallucinations Detected (ON)    : 0
  * Structural Graph Citations Grounded   : 3
=========================================================================================================
```

---

## Learnings: Ablation Testing for RAG Pipelines

Ablation testing is the foundational experimental discipline for evaluating and optimizing Retrieval-Augmented Generation (RAG) architectures.

### 1. Causality vs. Correlation in RAG
Standard end-to-end evaluation often muddles causes: when an LLM gives a poor architectural response, was the prompt poorly formatted, the temperature too high, or the retriever omitting vital dependencies?
- **Ablation Principle**: Hold every variable constant (identical question, identical vector top-$k$, identical system prompt and citation rules) and toggle exactly one architectural component (structural graph augmentation).
- **Finding**: Grounding failures in multi-component queries (`ABL-03`, `ABL-07`) dropped to zero solely through the addition of 1-hop graph neighbors.

### 2. The "Vector Anchor Trap" (Semantic Similarity $\neq$ Structural Relationship)
Dense semantic search operates on textual cosine similarity. While effective at finding the entry point for a concept, it is blind to caller/callee topology:
- In `scan_repository`, the function body coordinates `parse_with_timeout`, `detect_dead_code`, `compute_pagerank`, and `graph_to_json`.
- A developer asking *"How does scan_repository orchestrate AST parsing, dead code detection, and graph metrics?"* produces a vector query that retrieves `scan_repository` and `compute_pagerank`, but leaves `detect_dead_code` and `parse_with_timeout` unretrieved.
- **The Failure Mode**: When the LLM attempts to answer the developer's question accurately, it cites the unretrieved components. In a pure vector pipeline (Graph OFF), these are flagged as **hallucinated citations** because they do not exist in the context window.
- **The Graph RAG Solution**: 1-hop graph expansion automatically injects these callees into `<graph_neighbors>`, elevating Grounding Adherence from **60.0% to 100.0%**.

### 3. Context Purity vs. Context Omission
Increasing vector top-$k$ to compensate for missing dependencies creates **Context Poisoning**:
- Expanding $k$ from 3 to 10 lowers `Precision@k`, adds irrelevant distractor functions, inflates prompt token costs, and increases latency.
- In contrast, **Targeted 1-Hop Graph Expansion** adds only direct topological callers and callees in compact XML blocks, maintaining high context density without prompt bloat.

### 4. Deterministic Grounding Evaluation as a CI Guardrail
Rather than relying on noisy, non-deterministic LLM-as-a-judge scorers:
- Enforce canonical citation tags: `[file_path::func::<func_name>]`.
- Automatically parse citations via regex and benchmark against `included_anchors` and `included_neighbors`.
- Compute deterministic metrics:
  $$\text{Adherence Score} = \frac{|\text{Valid Citations}|}{|\text{Total Citations}|}$$
- This enables reproducible regression testing on every commit.

---

## Running the Evaluation Suites

### 1. Hallucination Comparison Benchmark (Ablation Test)
Runs the 10-query ablation suite comparing Graph Augmentation ON vs. OFF:
```bash
python eval/hallucination_eval.py
```
Exports detailed artifacts:
- Markdown report: `eval/hallucination_report.md`
- JSON metrics: `eval/hallucination_report.json`

To run with live local Ollama inference:
```bash
python eval/hallucination_eval.py --live --model codellama
```

### 2. Quantitative Retrieval Evaluation Benchmark
Benchmarks dense vector retrieval across 20 natural-language queries:
```bash
python eval/retrieval_eval.py --k 3
```

### 3. Unit & Integration Test Suite
```bash
pytest tests/ -v
```

---

## Project Structure

```
├── app/
│   ├── main.py                     # FastAPI application factory
│   ├── routers/
│   │   ├── graph.py                # Graph visualization endpoints
│   │   └── query.py                # Graph RAG query & SSE streaming endpoints
│   └── services/
│       ├── ast_engine.py           # Tree-sitter AST parser
│       ├── git_service.py          # Git repository cloning & commit churn
│       └── graph_service.py        # NetworkX graph construction
├── parser/
│   └── repo_walker.py              # Recursive repository scanner & dead code detection
├── rag/
│   ├── embeddings.py               # SentenceTransformer all-MiniLM-L6-v2
│   ├── hybrid_retriever.py         # Two-stage vector + graph hybrid retriever
│   ├── ingestion.py                # Qdrant vector ingestion pipeline
│   ├── llm_engine.py               # Ollama client orchestration & grounding evaluation
│   ├── prompt_builder.py           # Structured XML prompt builder with token budgeting
│   ├── qdrant_client.py            # Qdrant client connection & collection management
│   └── test_search.py              # Semantic search query execution
├── eval/
│   ├── hallucination_eval.py       # Hallucination comparison & ablation test harness
│   ├── hallucination_report.md     # Markdown ablation benchmark report
│   ├── hallucination_report.json   # Machine-readable ablation benchmark data
│   ├── retrieval_eval.py           # Quantitative retrieval evaluation harness
│   ├── eval_report.md              # Markdown retrieval evaluation report
│   └── eval_report.json            # Machine-readable retrieval evaluation data
└── tests/                          # 175+ unit, integration, and ablation test suite
    ├── check_prompt.py             # Diagnostic prompt verification script
    ├── test_improved_prompt.py     # Prompt rule enhancement integration test
    ├── test_refined.py             # Step-by-step code analysis integration test
    ├── test_rule_fix.py            # Async streaming prompt rule test
    ├── test_stream_vs_sync.py      # Streaming vs sync response comparison test
    ├── test_synthesis.py           # Architecture explanation synthesis test
    └── test_hallucination_eval.py  # Ablation harness unit test suite
```

---

## License

This project is licensed under the MIT License.
