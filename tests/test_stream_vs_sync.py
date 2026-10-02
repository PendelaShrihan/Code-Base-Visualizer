"""
tests/test_stream_vs_sync.py
----------------------------
Integration test comparing streaming vs synchronous LLM synthesis.
Requires cached graph and vector chunks for psf-requests in Redis/Qdrant.
"""

from __future__ import annotations

import asyncio
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


async def main() -> None:
    repo_id = "psf-requests"
    query = "What does the send function do and which functions does it call?"

    graph = _load_cached_graph(repo_id)
    print("Graph:", graph is not None)
    code_cache = _build_code_cache(graph) if graph else {}
    print("Code cache:", len(code_cache))
    results = hybrid_search(query=query, graph=graph, top_k=5, repo_id=repo_id)
    print("Results:", len(results))

    prompt = build_rag_prompt(query=query, hybrid_results=results, code_cache=code_cache, repo_root=Path("/app"))
    print("\n--- PROMPT SYSTEM (first 200 chars) ---")
    print(prompt.system_prompt[:200])
    print("\n--- PROMPT USER (snippets check) ---")
    print("code_snippet rank in user_prompt:", "code_snippet rank" in prompt.user_prompt)

    engine = LLMEngine()

    # Test stream_synthesize_async
    print("\n--- TESTING stream_synthesize_async ---")
    stream_chunks = []
    final_res = None
    async for chunk, maybe_res in engine.stream_synthesize_async(prompt=prompt, query=query):
        if chunk:
            stream_chunks.append(chunk)
        if maybe_res:
            final_res = maybe_res

    print("Stream chunks count:", len(stream_chunks))
    print("Stream content:\n", "".join(stream_chunks))


if __name__ == "__main__":
    asyncio.run(main())
