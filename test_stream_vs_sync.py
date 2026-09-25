import asyncio
from app.routers.query import _load_cached_graph, _build_code_cache
from rag.hybrid_retriever import hybrid_search
from rag.prompt_builder import build_rag_prompt
from rag.llm_engine import LLMEngine
from pathlib import Path

async def main():
    repo_id = "psf-requests"
    query = "What does the send function do and which functions does it call?"
    
    graph = _load_cached_graph(repo_id)
    print("Graph:", graph is not None)
    code_cache = _build_code_cache(graph)
    print("Code cache:", len(code_cache))
    results = hybrid_search(query=query, graph=graph, top_k=5, repo_id=repo_id)
    print("Results:", len(results))
    
    prompt = build_rag_prompt(query=query, hybrid_results=results, code_cache=code_cache, repo_root=Path("/app"))
    print("\n--- PROMPT SYSTEM (first 200 chars) ---")
    print(prompt.system_prompt[:200])
    print("\n--- PROMPT USER (snippets check) ---")
    print("code_snippet rank in user_prompt:", "code_snippet rank" in prompt.user_prompt)
    
    engine = LLMEngine()
    
    # 1. Test stream_synthesize_async
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

asyncio.run(main())
