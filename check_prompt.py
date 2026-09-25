import json
from app.routers.query import _load_cached_graph, _build_code_cache
from rag.hybrid_retriever import hybrid_search
from rag.prompt_builder import build_rag_prompt, resolve_function_code
from pathlib import Path

repo_id = "psf-requests"
query = "What does the send function do and which functions does it call?"

graph = _load_cached_graph(repo_id)
print(f"Graph loaded: {graph is not None}")
if graph:
    print(f"Graph node count: {graph.number_of_nodes()}")
    code_cache = _build_code_cache(graph)
    print(f"Code cache length: {len(code_cache)}")
    
    # Run hybrid_search
    results = hybrid_search(query=query, graph=graph, top_k=5, repo_id=repo_id)
    print(f"Hybrid search returned {len(results)} results:")
    for r in results:
        fn = r.get("func_name")
        fp = r.get("file_path")
        src = r.get("source")
        code = resolve_function_code(func_name=fn, file_path=fp, repo_root=Path("/app"), code_cache=code_cache)
        print(f" - [{src}] {fp} :: {fn} -> Code resolved: {code is not None} (len={len(code) if code else 0})")

    # Build prompt
    prompt = build_rag_prompt(query=query, hybrid_results=results, code_cache=code_cache, repo_root=Path("/app"))
    print("\n--- USER PROMPT FIRST 1500 CHARS ---")
    print(prompt.user_prompt[:1500])
