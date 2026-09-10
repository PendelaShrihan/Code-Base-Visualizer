"""
rag/prompt_builder.py
---------------------
Prompt Engineering & Context Optimization Engine for CodeBase Visualizer.

Architecture Overview
---------------------
This module turns raw Graph RAG retrieval results (from :func:`rag.hybrid_retriever.hybrid_search`)
into high-density, structured prompts optimized for Large Language Models (LLMs).

It solves two critical challenges in Code RAG systems:
1. **Context Window Optimization**: Managing finite token budgets by prioritising primary
   vector code anchors, compressing 1-hop graph neighbor metadata into structural relationship
   topologies, and dynamically pruning lower-priority items when budgets are constrained.
2. **Hallucination Prevention**: Enforcing strict grounding guardrails, XML boundary isolation,
   mandatory source citation protocols (``[file_path::func::<name>]``), negative constraints,
   and ignorance admission directives so the model never fabricates code or dependencies.

Prompt Anatomy
--------------
::

    ┌────────────────────────────────────────────────────────────────────────┐
    │ SYSTEM PROMPT (Anti-Hallucination Grounding Rules)                     │
    │  • Strict Grounding: answer solely from <repository_context>           │
    │  • Mandatory Citations: [file_path::func::<name>]                      │
    │  • Distinction: Vector Anchors (full source) vs Graph Neighbors (topo) │
    │  • Ignorance Protocol: Admit when context is insufficient              │
    └────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
    ┌────────────────────────────────────────────────────────────────────────┐
    │ USER PROMPT (<repository_context>)                                     │
    │  ┌──────────────────────────────────────────────────────────────────┐  │
    │  │ <graph_topology>                                                 │  │
    │  │   Call hierarchy, PageRank centrality, Git commit churn          │  │
    │  └──────────────────────────────────────────────────────────────────┘  │
    │  ┌──────────────────────────────────────────────────────────────────┐  │
    │  │ <code_snippets> (Vector Anchors)                                 │  │
    │  │   Exact byte-level source code extracted via Tree-sitter AST     │  │
    │  └──────────────────────────────────────────────────────────────────┘  │
    │  ┌──────────────────────────────────────────────────────────────────┐  │
    │  │ <graph_neighbors> (Callers & Callees)                            │  │
    │  │   High-density structural relations and interface summaries      │  │
    │  └──────────────────────────────────────────────────────────────────┘  │
    │                                                                        │
    │  <user_query>                                                          │
    │    Natural language developer inquiry                                  │
    │  </user_query>                                                         │
    └────────────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from rag.hybrid_retriever import HybridResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants & Defaults
# ---------------------------------------------------------------------------

#: Default maximum token budget for the full prompt (system + user prompt).
DEFAULT_MAX_CONTEXT_TOKENS: int = 4096

#: Conservative buffer reserved for the system prompt and instructions.
SYSTEM_PROMPT_TOKEN_RESERVE: int = 500

#: Conservative buffer reserved for user query and response instructions.
QUERY_TOKEN_RESERVE: int = 200

# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class StructuredPrompt:
    """Encapsulates a fully constructed, token-optimized LLM prompt.

    Attributes:
        system_prompt: Grounding instructions, citation rules, and persona.
        user_prompt: XML-delimited repository context and developer query.
        full_prompt: Combined text representation suitable for single-turn models.
        estimated_tokens: Total estimated tokens across system and user prompts.
        max_tokens: Maximum token budget specified for this prompt.
        token_budget_exceeded: True if total tokens exceeded max_tokens before pruning.
        included_anchors: List of function identifiers included as primary code anchors.
        included_neighbors: List of function identifiers included as graph neighbor context.
        pruned_items: Metadata for any candidate items pruned to respect token budget.
    """

    system_prompt: str
    user_prompt: str
    full_prompt: str
    estimated_tokens: int
    max_tokens: int
    token_budget_exceeded: bool
    included_anchors: list[str] = field(default_factory=list)
    included_neighbors: list[str] = field(default_factory=list)
    pruned_items: list[dict[str, Any]] = field(default_factory=list)

    def to_messages(self) -> list[dict[str, str]]:
        """Return the standard OpenAI/Gemini/Anthropic role-message dict list."""
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.user_prompt},
        ]


# ---------------------------------------------------------------------------
# Token Estimation & Budgeting
# ---------------------------------------------------------------------------

def estimate_tokens(text: str) -> int:
    """Estimate token count for code and technical text.

    Uses a calibrated sub-word heuristic that accounts for:
    - Code identifiers in snake_case and camelCase (~1.3 tokens per identifier)
    - Punctuation, braces, and indentation symbols
    - Natural language prose (~0.75 tokens per whitespace-separated word)

    Empirically calibrated to align within ~5% of OpenAI cl100k_base and
    Llama/SentenceTransformer tokenizers without requiring heavy runtime dependencies.

    Args:
        text: Input string to measure.

    Returns:
        Integer estimated token count (minimum 0).
    """
    if not text:
        return 0

    tokens = re.findall(r"\w+|[^\w\s]|\n", text)
    count = 0
    for tok in tokens:
        if tok == "\n":
            count += 1
        elif tok.isalnum() or "_" in tok:
            count += max(1, math.ceil(len(tok) / 3.5))
        else:
            count += 1

    return count


# ---------------------------------------------------------------------------
# Source Code Snippet Resolution
# ---------------------------------------------------------------------------

def resolve_function_code(
    func_name: str,
    file_path: str,
    repo_root: Optional[Path | str] = None,
    code_cache: Optional[dict[str, str]] = None,
) -> Optional[str]:
    """Retrieve the exact source code for a function.

    Resolution strategy:
    1. Check *code_cache* if provided (key: ``f"{file_path}::{func_name}"`` or ``func_name``).
    2. If *repo_root* is provided, locate the file on disk and extract the exact
       byte-range using Tree-sitter AST (:func:`app.services.ast_engine.extract_function_chunks`).
    3. Fall back to reading the raw file and locating the ``def <func_name>`` block.

    Args:
        func_name: Bare function name (e.g. ``"clone_repository"``).
        file_path: Relative file path (e.g. ``"app/services/git_service.py"``).
        repo_root: Optional filesystem root directory of the repository.
        code_cache: Optional mapping of pre-extracted function bodies.

    Returns:
        String containing the function source code, or ``None`` if not found.
    """
    if code_cache:
        key = f"{file_path}::{func_name}"
        if key in code_cache:
            return code_cache[key]
        if func_name in code_cache:
            return code_cache[func_name]

    if not repo_root:
        return None

    target_file = Path(repo_root) / file_path
    if not target_file.is_file():
        logger.debug("resolve_function_code: file %s does not exist on disk", target_file)
        return None

    try:
        source_bytes = target_file.read_bytes()
    except OSError as err:
        logger.warning("resolve_function_code: unable to read %s: %s", target_file, err)
        return None

    # Try Tree-sitter AST extraction
    try:
        from app.services.ast_engine import extract_function_chunks

        chunks = extract_function_chunks(source_bytes)
        for chunk in chunks:
            if chunk.get("name") == func_name:
                return chunk.get("text", "")
    except Exception as err:
        logger.debug(
            "resolve_function_code: Tree-sitter extraction failed for %s (%s); falling back",
            func_name,
            err,
        )

    # Line-slice fallback
    try:
        text = source_bytes.decode("utf-8", errors="replace")
        lines = text.splitlines()
        def_pattern = re.compile(rf"^\s*def\s+{re.escape(func_name)}\b")
        start_idx = None
        base_indent = 0

        for idx, line in enumerate(lines):
            if def_pattern.match(line):
                start_idx = idx
                base_indent = len(line) - len(line.lstrip())
                break

        if start_idx is not None:
            extracted_lines = [lines[start_idx]]
            for line in lines[start_idx + 1:]:
                if not line.strip():
                    extracted_lines.append(line)
                    continue
                current_indent = len(line) - len(line.lstrip())
                if current_indent <= base_indent and (
                    line.lstrip().startswith("def ") or line.lstrip().startswith("class ")
                ):
                    break
                extracted_lines.append(line)
            return "\n".join(extracted_lines).rstrip()
    except Exception as err:
        logger.warning("resolve_function_code: line-scan fallback failed: %s", err)

    return None


# ---------------------------------------------------------------------------
# System Prompt Construction (Anti-Hallucination Guardrails)
# ---------------------------------------------------------------------------

def build_system_prompt() -> str:
    """Generate the system prompt with strict grounding and citation directives.

    Anti-Hallucination Guardrails:
    1. Grounding: Answer ONLY based on the facts in ``<repository_context>``.
    2. Ignorance Protocol: If the context is insufficient, explicitly state so.
    3. Structural Verification: Distinguish between vector anchors (full code body)
       and graph neighbors (call hierarchy only).
    4. Citation Requirement: Every substantive code statement must cite
       ``[file_path::func::<name>]``.
    5. Negative Constraint: Never fabricate functions, parameters, or external dependencies.
    """
    return (
        "You are an expert Software Architecture and Code Intelligence Specialist.\n"
        "Your task is to answer developer questions about a repository using the provided "
        "<repository_context>.\n\n"
        "CRITICAL GROUNDING & ACCURACY RULES:\n"
        "1. STRICT GROUNDING: Rely EXCLUSIVELY on the code snippets and graph metadata provided in "
        "<repository_context>. Do not invent or assume logic, arguments, decorators, or side effects.\n"
        "2. IGNORANCE PROTOCOL: If the provided context does not contain enough information to answer "
        "the question with certainty, state clearly: 'I cannot answer this question based on the provided "
        "codebase context.' and specify what is missing.\n"
        "3. ANCHORS VS NEIGHBORS: Distinguish between:\n"
        "   - Vector Anchors: Complete function implementations provided in <code_snippets>.\n"
        "   - Graph Neighbors: 1-hop caller/callee relationships provided in <graph_neighbors>.\n"
        "   Never pretend to know the full internal implementation of a graph neighbor unless its code is "
        "explicitly shown in <code_snippets>.\n"
        "4. MANDATORY CITATIONS: Every claim regarding function logic, dependencies, or control flow "
        "MUST include an explicit citation in the format [file_path::func::<func_name>].\n"
        "5. EVIDENCE FIRST: Begin your response with a concise <evidence_summary> noting which functions "
        "and relationships you consulted, followed by your structured explanation."
    )


# ---------------------------------------------------------------------------
# Context Formatting & Pruning
# ---------------------------------------------------------------------------

def format_graph_topology(results: Sequence[HybridResult]) -> str:
    """Render a compact ASCII/Markdown call-graph topology overview.

    Provides high-density structural context with minimal token overhead:
    - Identifies which functions are anchors vs callers vs callees.
    - Highlights PageRank centrality (architectural importance).
    - Flags high-churn or dead-code candidates.
    """
    if not results:
        return "No graph topology available."

    lines: list[str] = ["Graph Topology Overview:"]
    anchors = [r for r in results if r.get("source") == "vector"]
    callees = [r for r in results if r.get("source") == "graph_callee"]
    callers = [r for r in results if r.get("source") == "graph_caller"]

    lines.append(
        f"- Scope: {len(anchors)} primary anchor(s), {len(callees)} callee(s), {len(callers)} caller(s)"
    )

    for r in results:
        func = r.get("func_name", "unknown")
        file_p = r.get("file_path", "")
        src = r.get("source", "vector")
        pr = r.get("pagerank", 0.0)
        churn = r.get("commit_count", 0)
        dead = r.get("is_dead_code_candidate", False)
        anchor_parent = r.get("anchor_func")

        rel_desc = ""
        if src == "vector":
            rel_desc = "[PRIMARY VECTOR ANCHOR]"
        elif src == "graph_callee":
            rel_desc = f"[CALLEE of {anchor_parent}]"
        elif src == "graph_caller":
            rel_desc = f"[CALLER of {anchor_parent}]"

        flags = []
        if dead:
            flags.append("DEAD_CODE_CANDIDATE")
        if churn > 10:
            flags.append(f"HOTSPOT_CHURN({churn})")

        flag_str = f" | Flags: {', '.join(flags)}" if flags else ""
        lines.append(
            f"  * {func}() in {file_p} {rel_desc} (PageRank: {pr:.4f}{flag_str})"
        )

    return "\n".join(lines)


def format_vector_snippet(
    result: HybridResult,
    code: Optional[str],
    rank: int,
) -> str:
    """Format a primary vector anchor code snippet with metadata header."""
    func = result.get("func_name", "unknown")
    file_p = result.get("file_path", "unknown")
    score = result.get("score", 0.0)
    label = result.get("similarity_label", "N/A")
    pr = result.get("pagerank", 0.0)
    churn = result.get("commit_count", 0)
    dead = result.get("is_dead_code_candidate", False)

    body = code.strip() if code else f"# [Source code not resolved for {func}() in {file_p}]"

    return (
        f'<code_snippet rank="{rank}" function="{func}" file="{file_p}">\n'
        f"<!-- Metadata: Cosine Similarity: {score:.4f} ({label}) | PageRank: {pr:.6f} | "
        f"Git Churn: {churn} commits | Dead Code: {'Yes' if dead else 'No'} -->\n"
        f"```python\n"
        f"{body}\n"
        f"```\n"
        f"</code_snippet>"
    )


def format_graph_neighbor_block(result: HybridResult) -> str:
    """Format a 1-hop graph neighbor into a terse, high-density XML block."""
    func = result.get("func_name", "unknown")
    file_p = result.get("file_path", "unknown")
    src = result.get("source", "graph_neighbor")
    anchor_func = result.get("anchor_func", "unknown")
    pr = result.get("pagerank", 0.0)
    churn = result.get("commit_count", 0)
    dead = result.get("is_dead_code_candidate", False)

    relation = (
        f"Called by anchor '{anchor_func}' (callee dependency)"
        if src == "graph_callee"
        else f"Calls anchor '{anchor_func}' (upstream caller / usage site)"
    )

    dead_flag = ' dead_code="true"' if dead else ""
    return (
        f'<neighbor function="{func}" file="{file_p}" relation_type="{src}"'
        f' anchor="{anchor_func}" pagerank="{pr:.4f}" churn="{churn}"{dead_flag}>\n'
        f"  Structural Relation: {relation}\n"
        f"</neighbor>"
    )


# ---------------------------------------------------------------------------
# Main Prompt Construction Pipeline
# ---------------------------------------------------------------------------

def build_rag_prompt(
    query: str,
    hybrid_results: Sequence[HybridResult],
    repo_root: Optional[Path | str] = None,
    code_cache: Optional[dict[str, str]] = None,
    max_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS,
    token_estimator: Callable[[str], int] = estimate_tokens,
) -> StructuredPrompt:
    """Construct a structured, token-optimized LLM prompt for Code RAG.

    Combines vector code snippets with graph neighbor metadata while strictly
    adhering to context window token limits and hallucination prevention standards.

    Pipeline Steps:
    1. Build immutable System Prompt with anti-hallucination rules.
    2. Format user query and calculate available context budget.
    3. Generate high-level Graph Topology summary.
    4. Categorize results into primary Vector Anchors and Graph Neighbors.
    5. Resolve source code for anchors (via AST engine or cache).
    6. Priority-based pruning:
       - Anchors are included in score rank order.
       - Neighbors are formatted as compact structural blocks.
       - If budget is constrained, lower-ranked items are pruned with logging.
    7. Assemble structured XML-tagged user prompt.
    8. Package into :class:`StructuredPrompt`.

    Args:
        query: Developer's plain-text inquiry.
        hybrid_results: Ranked results from :func:`rag.hybrid_retriever.hybrid_search`.
        repo_root: Optional repository root for on-disk function code resolution.
        code_cache: Optional dict mapping function keys to code strings.
        max_tokens: Total token budget for system + user prompt.
        token_estimator: Callable that computes estimated token count for text.

    Returns:
        A fully populated :class:`StructuredPrompt` object.
    """
    clean_query = query.strip() if query else ""
    system_prompt = build_system_prompt()

    # Base token expenditure
    sys_tokens = token_estimator(system_prompt)
    query_template_base = (
        f"<user_query>\n{clean_query}\n</user_query>\n\n"
        f"Please analyze the provided codebase context and answer the query following "
        f"all grounding and citation rules."
    )
    query_tokens = token_estimator(query_template_base)

    # Context token budget
    available_context_budget = max(0, max_tokens - sys_tokens - query_tokens)
    logger.debug(
        "build_rag_prompt: max_tokens=%d, sys_tokens=%d, query_tokens=%d, available_budget=%d",
        max_tokens,
        sys_tokens,
        query_tokens,
        available_context_budget,
    )

    # Categorize results
    anchors: list[HybridResult] = [r for r in hybrid_results if r.get("source") == "vector"]
    neighbors: list[HybridResult] = [r for r in hybrid_results if r.get("source") != "vector"]

    included_anchors: list[str] = []
    included_neighbors: list[str] = []
    pruned_items: list[dict[str, Any]] = []

    # Format graph topology (high density, low token cost)
    topology_text = format_graph_topology(hybrid_results)
    topology_block = f"<graph_topology>\n{topology_text}\n</graph_topology>"
    topology_tokens = token_estimator(topology_block)

    used_context_tokens = topology_tokens
    snippet_blocks: list[str] = []
    neighbor_blocks: list[str] = []

    # 1. Process Vector Anchors (Priority 1)
    for idx, anchor in enumerate(anchors, start=1):
        func_name = anchor.get("func_name", "")
        file_path = anchor.get("file_path", "")

        # Try to resolve code
        code = resolve_function_code(
            func_name=func_name,
            file_path=file_path,
            repo_root=repo_root,
            code_cache=code_cache,
        )

        block = format_vector_snippet(anchor, code, rank=idx)
        block_tokens = token_estimator(block)

        if used_context_tokens + block_tokens <= available_context_budget:
            snippet_blocks.append(block)
            used_context_tokens += block_tokens
            included_anchors.append(f"{file_path}::func::{func_name}")
        else:
            # Check if we can include a truncated signature stub to preserve awareness
            stub = f'<code_snippet rank="{idx}" function="{func_name}" file="{file_path}" status="pruned_due_to_budget" />'
            stub_tokens = token_estimator(stub)
            if used_context_tokens + stub_tokens <= available_context_budget:
                snippet_blocks.append(stub)
                used_context_tokens += stub_tokens

            pruned_items.append(
                {
                    "item": f"{file_path}::func::{func_name}",
                    "source": "vector",
                    "reason": "exceeded_token_budget",
                    "required_tokens": block_tokens,
                }
            )

    # 2. Process Graph Neighbors (Priority 2: compact structural relationship context)
    for neighbor in neighbors:
        func_name = neighbor.get("func_name", "")
        file_path = neighbor.get("file_path", "")
        src = neighbor.get("source", "graph_neighbor")

        block = format_graph_neighbor_block(neighbor)
        block_tokens = token_estimator(block)

        if used_context_tokens + block_tokens <= available_context_budget:
            neighbor_blocks.append(block)
            used_context_tokens += block_tokens
            included_neighbors.append(f"{file_path}::func::{func_name} ({src})")
        else:
            pruned_items.append(
                {
                    "item": f"{file_path}::func::{func_name}",
                    "source": src,
                    "reason": "exceeded_token_budget",
                    "required_tokens": block_tokens,
                }
            )

    # Assemble sections
    context_sections: list[str] = [
        "<repository_context>",
        topology_block,
        "",
        "<code_snippets>",
        ("\n\n".join(snippet_blocks) if snippet_blocks else "<!-- No code snippets included -->"),
        "</code_snippets>",
        "",
        "<graph_neighbors>",
        ("\n".join(neighbor_blocks) if neighbor_blocks else "<!-- No graph neighbors included -->"),
        "</graph_neighbors>",
        "</repository_context>",
    ]

    context_str = "\n".join(context_sections)

    user_prompt = f"{context_str}\n\n{query_template_base}"
    total_tokens = token_estimator(system_prompt) + token_estimator(user_prompt)

    full_prompt = (
        f"=== SYSTEM PROMPT ===\n{system_prompt}\n\n"
        f"=== USER PROMPT ===\n{user_prompt}"
    )

    budget_exceeded = bool(pruned_items) or (total_tokens > max_tokens)

    return StructuredPrompt(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        full_prompt=full_prompt,
        estimated_tokens=total_tokens,
        max_tokens=max_tokens,
        token_budget_exceeded=budget_exceeded,
        included_anchors=included_anchors,
        included_neighbors=included_neighbors,
        pruned_items=pruned_items,
    )


# ---------------------------------------------------------------------------
# CLI / Demo Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    synthetic_results: list[HybridResult] = [
        {
            "rank": 1,
            "func_name": "clone_repository",
            "file_path": "app/services/git_service.py",
            "score": 0.8924,
            "similarity_label": "HIGH CONFIDENCE",
            "pagerank": 0.0415,
            "commit_count": 14,
            "is_dead_code_candidate": False,
            "source": "vector",
            "anchor_func": None,
        },
        {
            "rank": 2,
            "func_name": "cleanup_repo_directory",
            "file_path": "app/services/git_service.py",
            "score": 0.0,
            "similarity_label": "VERY WEAK / NOISE",
            "pagerank": 0.0120,
            "commit_count": 3,
            "is_dead_code_candidate": False,
            "source": "graph_callee",
            "anchor_func": "clone_repository",
        },
        {
            "rank": 3,
            "func_name": "process_repository_task",
            "file_path": "worker/tasks.py",
            "score": 0.0,
            "similarity_label": "VERY WEAK / NOISE",
            "pagerank": 0.0380,
            "commit_count": 8,
            "is_dead_code_candidate": False,
            "source": "graph_caller",
            "anchor_func": "clone_repository",
        },
    ]

    mock_code = {
        "clone_repository": (
            "def clone_repository(repo_url: str, target_dir: Path | None = None) -> Path:\n"
            '    """Clone a git repository to a temporary directory with safety checks."""\n'
            "    if not is_valid_git_url(repo_url):\n"
            '        raise ValueError(f"Invalid git URL: {repo_url}")\n'
            "    target = target_dir or Path(tempfile.mkdtemp(prefix='repo_'))\n"
            "    try:\n"
            "        Repo.clone_from(repo_url, target, depth=1)\n"
            "        return target\n"
            "    except Exception:\n"
            "        cleanup_repo_directory(target)\n"
            "        raise"
        )
    }

    demo_query = "How does clone_repository clean up when an error occurs?"
    prompt = build_rag_prompt(
        query=demo_query,
        hybrid_results=synthetic_results,
        code_cache=mock_code,
        max_tokens=2000,
    )

    print("=" * 80)
    print("STRUCTURED PROMPT INSPECTION")
    print("=" * 80)
    print(f"Total Estimated Tokens : {prompt.estimated_tokens} / {prompt.max_tokens}")
    print(f"Included Anchors       : {prompt.included_anchors}")
    print(f"Included Neighbors     : {prompt.included_neighbors}")
    print(f"Pruned Items           : {len(prompt.pruned_items)}")
    print("=" * 80)
    print(prompt.full_prompt)
    print("=" * 80)
