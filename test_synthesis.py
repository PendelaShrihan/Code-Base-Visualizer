import json
from app.routers.query import _load_cached_graph, _build_code_cache
from rag.hybrid_retriever import hybrid_search
from rag.prompt_builder import build_rag_prompt, build_system_prompt, StructuredPrompt
from rag.llm_engine import LLMEngine
from pathlib import Path

repo_id = "psf-requests"
query = "What does the send function do and which functions does it call?"

graph = _load_cached_graph(repo_id)
code_cache = _build_code_cache(graph)
results = hybrid_search(query=query, graph=graph, top_k=5, repo_id=repo_id)

print(f"Hybrid search returned {len(results)} results")

# Build standard prompt
prompt = build_rag_prompt(query=query, hybrid_results=results, code_cache=code_cache, repo_root=Path("/app"))

engine = LLMEngine()
print("\n--- SYNTHESIZING WITH CURRENT PROMPT ---")
res = engine.synthesize(prompt=prompt, query=query)
print("CONTENT:\n", res.content)
print("\nCITATIONS:\n", [c.raw_text for c in res.citations])
