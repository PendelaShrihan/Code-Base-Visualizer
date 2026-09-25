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
    code_cache = _build_code_cache(graph)
    results = hybrid_search(query=query, graph=graph, top_k=5, repo_id=repo_id)
    
    prompt = build_rag_prompt(query=query, hybrid_results=results, code_cache=code_cache, repo_root=Path("/app"))
    
    prompt.system_prompt = (
        "You are an expert Software Architecture and Code Intelligence Specialist.\n"
        "Your task is to answer developer questions about a repository using the provided <repository_context>.\n\n"
        "CRITICAL GROUNDING & ACCURACY RULES:\n"
        "1. STRICT GROUNDING: Rely on the code snippets and graph metadata in <repository_context>. Thoroughly inspect the source code implementations in <code_snippets>. Trace the logic, control flow, arguments, return values, and every function/method call made inside the function body.\n"
        "2. FUNCTION & METHOD CALLS: Look directly inside the function definitions in <code_snippets>. Trace all method calls, function invocations, and helper calls made in the code (e.g. self.get_adapter, adapter.send, resolve_proxies, dispatch_hook, extract_cookies_to_jar, etc.). Describe what each call does in the context of the function.\n"
        "3. CALL RELATIONSHIPS: Information about callers and callees comes from BOTH the source code implementations in <code_snippets> and from structural links in <graph_neighbors>. If <graph_neighbors> is empty, rely on the calls visible in <code_snippets>.\n"
        "4. MANDATORY CITATIONS: Cite the file and function name in format [file_path::func::<func_name>].\n"
        "5. IGNORANCE PROTOCOL: Only if <repository_context> contains no functions or snippets matching the query topic, state clearly: 'I cannot answer this question based on the provided codebase context.' and specify what is missing.\n"
        "6. EVIDENCE FIRST: Begin your response with a concise <evidence_summary> noting which functions and relationships you consulted, followed by your structured explanation."
    )
    
    engine = LLMEngine()
    
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
