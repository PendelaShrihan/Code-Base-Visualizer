# Quantitative Retrieval Evaluation Report

## Executive Summary
- **Benchmark Size**: 20 Hand-Labeled Natural Language Questions
- **Retrieval Cutoff (k)**: 3
- **Embedding Model**: `all-MiniLM-L6-v2` (384 dimensions, Cosine Metric)
- **Macro Precision@3**: **33.3%** (Single-target theoretical upper bound: 33.3%)
- **Macro Recall@3**: **100.0%**
- **Macro F1@3**: **0.5000**
- **Hit Rate@3**: **100.0%**
- **Mean Reciprocal Rank (MRR)**: **1.0000**
- **Top-1 Accuracy**: **100.0%**

---

## Quantitative RAG Evaluation Methodology

### 1. The Precision@k vs Recall@k Dilemma
In dense retrieval for RAG, **Recall@k** measures *completeness* (whether the gold function reaches the LLM context), while **Precision@k** measures *purity* (how much noise/distraction is injected into the context window).

$$Precision@k = \frac{|Retrieved_k \cap Relevant|}{k}$$
$$Recall@k = \frac{|Retrieved_k \cap Relevant|}{|Relevant|}$$

> [!IMPORTANT]
> When evaluating questions with a single known-correct function ($|Relevant| = 1$) at $k=3$, the maximum possible Precision@3 is $\frac{1}{3} \approx 33.3\%$. The remaining two retrieved positions are non-gold distractors. Recall@3, however, reaches 100% whenever the gold function is placed in ranks 1, 2, or 3.

### 2. Metric Interpretation for Code RAG
| Metric | Formula | Practical Meaning in Codebases | Target Goal |
| :--- | :--- | :--- | :--- |
| **Recall@3** | Hits in top-3 / Total Ground Truth | Does the prompt contain the function the developer asked about? | $\ge 85\%$ |
| **Precision@3** | Hits in top-3 / 3 | How concentrated is the context window? | $\ge 30\%$ (for single-target) |
| **MRR** | $\frac{1}{N} \sum \frac{1}{Rank_1}$ | Is the best match at the very top of the list? | $\ge 0.75$ |
| **Top-1 Accuracy** | Proportion of queries with gold at Rank 1 | Can the pipeline answer directly using the primary anchor? | $\ge 70\%$ |

---

## Detailed Per-Query Results

| ID | Natural Language Query | Ground Truth | Rank 1 Retrieved | Score | P@3 | R@3 | MRR | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Q01** | How do I clone a remote git repository into a local temporary folder with guardrails? | `clone_repository` | `clone_repository` | 0.7048 | 0.33 | 1.00 | 1.00 | ✅ PASS |
| **Q02** | Where is the cleanup logic to delete cloned repository files and clear read-only permissions? | `cleanup_repo_directory` | `cleanup_repo_directory` | 0.6130 | 0.33 | 1.00 | 1.00 | ✅ PASS |
| **Q03** | How can I purge stale temporary repository directories older than thirty minutes? | `garbage_collect_temp_repos` | `garbage_collect_temp_repos` | 0.5587 | 0.33 | 1.00 | 1.00 | ✅ PASS |
| **Q04** | Which function calculates the total byte size of a directory tree while ignoring symlinks? | `get_directory_size_bytes` | `get_directory_size_bytes` | 0.6792 | 0.33 | 1.00 | 1.00 | ✅ PASS |
| **Q05** | Where is commit churn or revision count calculated for an individual source file? | `count_file_commits` | `count_file_commits` | 0.7171 | 0.33 | 1.00 | 1.00 | ✅ PASS |
| **Q06** | How does the system parse Python code with Tree-sitter while enforcing a strict timeout? | `parse_with_timeout` | `parse_with_timeout` | 0.7631 | 0.33 | 1.00 | 1.00 | ✅ PASS |
| **Q07** | How does the parser retrieve all function and method names declared in a Python file? | `extract_function_names` | `extract_function_names` | 0.6667 | 0.33 | 1.00 | 1.00 | ✅ PASS |
| **Q08** | Which helper extracts classes, method calls, and imported modules using Tree-sitter queries? | `extract_structure` | `extract_structure` | 0.6453 | 0.33 | 1.00 | 1.00 | ✅ PASS |
| **Q09** | How are intra-file caller and callee function relationships extracted from the syntax tree? | `extract_call_edges` | `extract_call_edges` | 0.7173 | 0.33 | 1.00 | 1.00 | ✅ PASS |
| **Q10** | How is a single file's extracted structure converted into a NetworkX directed graph? | `build_graph` | `build_graph` | 0.6975 | 0.33 | 1.00 | 1.00 | ✅ PASS |
| **Q11** | Where does the repo walker recursively traverse directory trees and aggregate AST graphs? | `scan_repository` | `scan_repository` | 0.6045 | 0.33 | 1.00 | 1.00 | ✅ PASS |
| **Q12** | How are uncalled functions with zero incoming call-graph edges detected as candidate dead code? | `detect_dead_code` | `detect_dead_code` | 0.6428 | 0.33 | 1.00 | 1.00 | ✅ PASS |
| **Q13** | Where is PageRank centrality calculated across all nodes in the codebase dependency graph? | `compute_pagerank` | `compute_pagerank` | 0.8216 | 0.33 | 1.00 | 1.00 | ✅ PASS |
| **Q14** | Which function serializes the NetworkX graph with node attributes and edges into a JSON dictionary? | `graph_to_json` | `graph_to_json` | 0.7787 | 0.33 | 1.00 | 1.00 | ✅ PASS |
| **Q15** | How are code chunk texts batch encoded into dense vector representations using SentenceTransformer? | `embed_chunks` | `embed_chunks` | 0.7579 | 0.33 | 1.00 | 1.00 | ✅ PASS |
| **Q16** | Where does the ingestion pipeline generate deterministic UUID-v5 chunk records from graph nodes? | `extract_function_chunks` | `extract_function_chunks` | 0.6353 | 0.33 | 1.00 | 1.00 | ✅ PASS |
| **Q17** | How are embedded code chunk vectors and metadata payloads upserted into Qdrant in batches? | `batch_upsert_chunks` | `batch_upsert_chunks` | 0.7686 | 0.33 | 1.00 | 1.00 | ✅ PASS |
| **Q18** | Where is the Qdrant code_chunks collection created with 384 dimensions and cosine metric? | `create_code_chunks_collection` | `create_code_chunks_collection` | 0.7781 | 0.33 | 1.00 | 1.00 | ✅ PASS |
| **Q19** | How does vector search find the top-k most semantically similar code functions for a question? | `search_functions` | `search_functions` | 0.6851 | 0.33 | 1.00 | 1.00 | ✅ PASS |
| **Q20** | How does hybrid retrieval expand vector search anchors with 1-hop caller and callee graph neighbors? | `hybrid_search` | `hybrid_search` | 0.7460 | 0.33 | 1.00 | 1.00 | ✅ PASS |

---

## Performance by Category

| Category | Queries | Precision@3 | Recall@3 | Hit Rate | MRR |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Repository Operations** | 5 | 0.333 | 1.000 | 100.0% | 1.000 |
| **AST Parsing** | 4 | 0.333 | 1.000 | 100.0% | 1.000 |
| **Graph Construction** | 1 | 0.333 | 1.000 | 100.0% | 1.000 |
| **Repository Walking** | 1 | 0.333 | 1.000 | 100.0% | 1.000 |
| **Dead Code Detection** | 1 | 0.333 | 1.000 | 100.0% | 1.000 |
| **Graph Centrality** | 1 | 0.333 | 1.000 | 100.0% | 1.000 |
| **Graph Serialization** | 1 | 0.333 | 1.000 | 100.0% | 1.000 |
| **Vector Embeddings** | 1 | 0.333 | 1.000 | 100.0% | 1.000 |
| **Vector Ingestion** | 2 | 0.333 | 1.000 | 100.0% | 1.000 |
| **Vector Database** | 1 | 0.333 | 1.000 | 100.0% | 1.000 |
| **Semantic Retrieval** | 1 | 0.333 | 1.000 | 100.0% | 1.000 |
| **Hybrid Retrieval** | 1 | 0.333 | 1.000 | 100.0% | 1.000 |

---

## Performance by Query Difficulty

| Difficulty | Queries | Precision@3 | Recall@3 | Hit Rate | MRR |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **direct** | 13 | 0.333 | 1.000 | 100.0% | 1.000 |
| **synonym** | 2 | 0.333 | 1.000 | 100.0% | 1.000 |
| **conceptual** | 5 | 0.333 | 1.000 | 100.0% | 1.000 |

---

## Diagnostic Insights & Next Steps
1. **Dense Semantic Matching Strength**: Strong semantic correlation on natural language queries with phrasing differences (e.g. 'purge stale clones' successfully retrieves `garbage_collect_temp_repos`).
2. **Context Window Optimization**: For prompts generated with retrieved code, a high Recall@3 ensures the necessary code body is sent to the LLM, while keeping $k=3$ limits total token usage.
3. **Graph-Augmentation Synergy**: By feeding the vector search output to `rag.hybrid_retriever.hybrid_search`, 1-hop graph neighbors can be pulled in structurally even if their textual description has lower direct semantic similarity.