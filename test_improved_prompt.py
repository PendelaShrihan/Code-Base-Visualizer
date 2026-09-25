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

# Test improved system prompt
new_system_prompt = (
    "You are an expert Software Architecture and Code Intelligence Specialist.\n"
    "Your task is to answer developer questions about a repository using the provided <repository_context>.\n\n"
    "CRITICAL GROUNDING & ACCURACY RULES:\n"
    "1. STRICT GROUNDING: Rely on the code snippets and graph metadata in <repository_context>. Analyze the actual source code implementations in <code_snippets> (identifying logic, control flow, and functions/methods called within the function bodies) as well as the structural links in <graph_neighbors>.\n"
    "2. IGNORANCE PROTOCOL: Only if the provided context completely lacks relevant code snippets or functions for the query, state: 'I cannot answer this question based on the provided codebase context.' and specify what is missing.\n"
    "3. ANCHORS VS NEIGHBORS: Vector Anchors in <code_snippets> provide full source code. Inspect their code bodies to understand how they work and what calls they make. Graph Neighbors in <graph_neighbors> provide structural 1-hop context.\n"
    "4. MANDATORY CITATIONS: Every claim regarding function logic, dependencies, or control flow MUST include an explicit citation in the format [file_path::func::<func_name>].\n"
    "5. EVIDENCE FIRST: Begin your response with a concise <evidence_summary> noting which functions and relationships you consulted, followed by your structured explanation."
)

prompt.system_prompt = new_system_prompt
prompt.full_prompt = f"{new_system_prompt}\n\n{prompt.user_prompt}"

engine = LLMEngine()
print("\n--- SYNTHESIZING WITH IMPROVED SYSTEM PROMPT ---")
res = engine.synthesize(prompt=prompt, query=query)
print("CONTENT:\n", res.content)
print("\nCITATIONS:\n", [c.raw_text for c in res.citations])
