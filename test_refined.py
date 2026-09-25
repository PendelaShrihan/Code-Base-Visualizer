from app.routers.query import _load_cached_graph, _build_code_cache
from rag.hybrid_retriever import hybrid_search
from rag.prompt_builder import build_rag_prompt
from rag.llm_engine import LLMEngine
from pathlib import Path

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
    "1. STRICT GROUNDING: Rely on the code snippets and graph metadata provided in <repository_context>. Thoroughly inspect the source code implementations in <code_snippets>. Trace the logic, control flow, arguments, and any function/method calls made inside the function body.\n"
    "2. CODE ANALYSIS: When asked what a function does or what functions it calls, read its implementation in <code_snippets> step-by-step and list all functions, methods, or hooks it invokes (e.g. self.get_adapter, adapter.send, resolve_proxies, dispatch_hook, etc.).\n"
    "3. ANCHORS VS NEIGHBORS: Functions in <code_snippets> provide full source code. Graph Neighbors in <graph_neighbors> provide structural 1-hop context.\n"
    "4. MANDATORY CITATIONS: Cite the file and function name in format [file_path::func::<func_name>].\n"
    "5. IGNORANCE PROTOCOL: Only if <repository_context> is empty or has zero relevant code for the requested topic, state: 'I cannot answer this question based on the provided codebase context.'\n"
    "6. EVIDENCE FIRST: Begin your response with a concise <evidence_summary> noting which functions and relationships you consulted, followed by your structured explanation."
)

prompt.full_prompt = f"{prompt.system_prompt}\n\n{prompt.user_prompt}"

engine = LLMEngine()
res = engine.synthesize(prompt=prompt, query=query)
print("--- ANSWER ---")
print(res.content)
