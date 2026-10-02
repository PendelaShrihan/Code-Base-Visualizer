"""
tests/test_synthesis.py
-----------------------
Integration test for full end-to-end prompt building and architecture synthesis.
Requires cached graph and vector chunks for psf-requests in Redis/Qdrant.
"""

from __future__ import annotations

import sys
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from pathlib import Path
from app.routers.query import _build_code_cache, _load_cached_graph
from rag.hybrid_retriever import hybrid_search
from rag.llm_engine import LLMEngine
from rag.prompt_builder import build_rag_prompt


def run_synthesis_test() -> None:
    repo_id = "psf-requests"
    query = "What does the send function do and which functions does it call?"

    graph = _load_cached_graph(repo_id)
    code_cache = _build_code_cache(graph) if graph else {}
    results = hybrid_search(query=query, graph=graph, top_k=5, repo_id=repo_id)

    print(f"Hybrid search returned {len(results)} results")

    prompt = build_rag_prompt(query=query, hybrid_results=results, code_cache=code_cache, repo_root=Path("/app"))

    engine = LLMEngine()
    print("\n--- SYNTHESIZING WITH CURRENT PROMPT ---")
    res = engine.synthesize(prompt=prompt, query=query)
    print("CONTENT:\n", res.content)
    print("\nCITATIONS:\n", [c.raw_text for c in res.citations])


if __name__ == "__main__":
    run_synthesis_test()
