"""
repo_walker.py
==============
Walk an entire repository tree, analyse every Python source file with the
Week-2 AST engine, and accumulate the results into a single
:class:
etworkx.DiGraph.

Public API
----------
scan_repository(repo_root)  ->  nx.DiGraph
    Recursively discover every `.py` file under *repo_root*, run
    `extract_structure()` and `extract_call_edges()` on each one, and
    merge all per-file sub-graphs into one accumulator graph.

attach_churn(graph, clone_path)  ->  None
    Compute git commit counts using GitPython and attach `commit_count`
    attributes to all relevant file/child nodes in the graph in-place.

graph_to_json(g)  ->  dict
    Serialise the accumulator graph to a JSON-friendly dict with two lists:
    `nodes` (id + all attributes) and `edges` (source, target, rel, edge_type).

Node-ID scoping convention
--------------------------
Because the same function name (e.g. `helper`) can appear in many files,
every node ID is *scoped* by the relative file path so IDs never collide::

    `<rel_path>::file`
    `<rel_path>::class::<ClassName>`
    `<rel_path>::import::<module_or_name>`
    `<rel_path>::call::<obj.method>`
    `<rel_path>::func::<function_name>`

Directories skipped by the walker
----------------------------------
`.git`, `venv`, `.venv`, `__pycache__`, `.tox`, `.eggs`,
`dist`, `build`, `.mypy_cache`, `.pytest_cache`, `node_modules`
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Union

import networkx as nx

from app.services.ast_engine import (
    extract_call_edges,
    extract_function_names,
    extract_structure,
)
from app.services.git_service import count_file_commits

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Directories to skip during the recursive walk
# ---------------------------------------------------------------------------

_SKIP_DIRS: frozenset[str] = frozenset({
    ".git",
    "venv",
    ".venv",
    "__pycache__",
    ".tox",
    ".eggs",
    "dist",
    "build",
    ".mypy_cache",
    ".pytest_cache",
    "node_modules",
    ".ruff_cache",
    ".hypothesis",
    "htmlcov",
    ".hg",
    ".svn",
})


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _iter_python_files(repo_root: Path) -> list[Path]:
    """Return all `.py` files under *repo_root*, skipping noise dirs.

    Uses an explicit DFS stack rather than `rglob` so we can prune entire
    subtrees (e.g. `venv/`) before ever descending into them.

    Args:
        repo_root: Absolute path to the repository root.

    Returns:
        Sorted list of absolute :class:Path objects for every `.py` file
        found.
    """
    found: list[Path] = []
    stack: list[Path] = [repo_root]

    while stack:
        current = stack.pop()
        try:
            entries = list(current.iterdir())
        except PermissionError:
            logger.warning("Permission denied: %s - skipping", current)
            continue

        for entry in entries:
            if entry.is_dir():
                if entry.name not in _SKIP_DIRS:
                    stack.append(entry)
            elif entry.is_file() and entry.suffix == ".py":
                found.append(entry)

    return sorted(found)


def _build_file_subgraph(
    file_path: Path,
    repo_root: Path,
    source_code: bytes,
) -> nx.DiGraph:
    """Analyse one Python file and return a scoped sub-graph.

    All node IDs are prefixed with `<rel_path>::` so they remain unique
    when merged into the accumulator graph.

    Args:
        file_path:   Absolute path to the `.py` file.
        repo_root:   Repository root used to compute the relative path that
                     forms the node-ID prefix.
        source_code: Raw UTF-8 bytes of the file (already read by caller).

    Returns:
        A :class:
x.DiGraph representing the single file.
    """
    rel = file_path.relative_to(repo_root).as_posix()
    prefix = rel  # e.g. "app/services/git_service.py"

    structure = extract_structure(source_code)
    call_edges = extract_call_edges(source_code)

    g: nx.DiGraph = nx.DiGraph()

    # -- root (file) node -----------------------------------------------------
    file_node_id = f"{prefix}::file"
    g.add_node(file_node_id, kind="file", path=rel, label=rel)

    # -- classes --------------------------------------------------------------
    for cls_name in structure.get("classes", []):
        node_id = f"{prefix}::class::{cls_name}"
        g.add_node(node_id, kind="class", name=cls_name, file=rel,
                   label=cls_name)
        g.add_edge(file_node_id, node_id, rel="contains", edge_type="EXTRACTED")

    # -- imports --------------------------------------------------------------
    for imp in structure.get("imports", []):
        node_id = f"{prefix}::import::{imp}"
        g.add_node(node_id, kind="import", name=imp, file=rel, label=imp)
        g.add_edge(file_node_id, node_id, rel="imports", edge_type="EXTRACTED")

    # -- method / attribute calls  (obj.method style) -------------------------
    seen_calls: set[str] = set()
    for call in structure.get("method_calls", []):
        node_id = f"{prefix}::call::{call}"
        if node_id not in seen_calls:
            seen_calls.add(node_id)
            g.add_node(node_id, kind="call_target", name=call, file=rel,
                       label=call)
        g.add_edge(file_node_id, node_id, rel="calls", edge_type="EXTRACTED")

    # -- function definitions -------------------------------------------------
    for func_name in extract_function_names(source_code):
        func_id = f"{prefix}::func::{func_name}"
        if func_id not in g:
            g.add_node(func_id, kind="function", name=func_name, file=rel,
                       label=func_name)

    # -- intra-file function->function call edges -----------------------------
    for caller, callee in call_edges:
        caller_id = f"{prefix}::func::{caller}"
        callee_id = f"{prefix}::func::{callee}"
        if caller_id not in g:
            g.add_node(caller_id, kind="function", name=caller, file=rel,
                       label=caller)
        if callee_id not in g:
            g.add_node(callee_id, kind="function", name=callee, file=rel,
                       label=callee)
        g.add_edge(caller_id, callee_id, rel="func_call", edge_type="EXTRACTED")

    return g


# ---------------------------------------------------------------------------
# Dead Code Detection & Entry Point Filtering
# ---------------------------------------------------------------------------

_ENTRY_POINT_EXACT_NAMES: frozenset[str] = frozenset({
    "main",
    "__main__",
    "cli",
    "run",
    "app",
    "handler",
    "lambda_handler",
    "execute",
    "start",
    "setup",
    "teardown",
    "setUp",
    "tearDown",
    "setUpClass",
    "tearDownClass",
    "lifespan",
    "startup",
    "shutdown",
    "init_app",
    "health",
    "health_check",
    "index",
    "root",
})

_ROUTER_DIR_NAMES: frozenset[str] = frozenset({
    "routers",
    "routes",
    "views",
    "endpoints",
    "controllers",
    "api",
})

_ENTRY_POINT_FILENAMES: frozenset[str] = frozenset({
    "main.py",
    "app.py",
    "wsgi.py",
    "asgi.py",
    "manage.py",
    "cli.py",
    "__main__.py",
    "tasks.py",
    "conftest.py",
})


def is_known_entry_point(func_name: str, file_path: str = "") -> bool:
    """
    Determine whether a function is a known framework or application entry point.

    Entry points are functions invoked externally by frameworks, runners, or CLI tools
    and should not be flagged as dead code even if they have zero intra-repo call edges.

    Exclusions:
      - Dunder / magic methods (__init__, __call__, etc.)
      - Standard entry points (main, cli, run, etc.)
      - Test functions (test_* or *_test) or functions inside test directories
      - HTTP route / endpoint handlers (get_*, post_*, health_check, or files in routers/api)
      - Celery / background tasks (*_task, task_*, process_*)
      - Framework lifecycle hooks (setup, teardown, lifespan, startup, shutdown)

    Args:
        func_name: Simple name of the function.
        file_path: Relative or absolute file path containing the function.

    Returns:
        True if the function matches known entry point heuristics; False otherwise.
    """
    # 1. Dunder / special methods
    if func_name.startswith("__") and func_name.endswith("__"):
        return True

    # 2. Exact match on known entry point names
    if func_name in _ENTRY_POINT_EXACT_NAMES:
        return True

    # 3. Test functions
    if func_name.startswith("test_") or func_name.endswith("_test"):
        return True

    # 4. HTTP method / route handler naming patterns
    if func_name.startswith(("get_", "post_", "put_", "delete_", "patch_", "head_", "options_")):
        return True

    # 5. Background / task handlers
    if func_name.startswith(("task_", "process_")) or func_name.endswith("_task"):
        return True

    # 6. File path heuristics
    if file_path:
        path_obj = Path(file_path)
        # Test directories or files
        if any(part in ("test", "tests") for part in path_obj.parts):
            return True
        if path_obj.name.startswith("test_") or path_obj.name.endswith("_test.py") or path_obj.name == "conftest.py":
            return True

        # Router / API / endpoint modules
        if any(part in _ROUTER_DIR_NAMES for part in path_obj.parts):
            return True

        # Top-level executable scripts
        if path_obj.name in _ENTRY_POINT_FILENAMES:
            if func_name in ("main", "run", "cli", "handler") or func_name.startswith(("process_", "task_")):
                return True

    return False


def detect_dead_code(graph: nx.DiGraph) -> list[dict[str, Any]]:
    """
    Identify candidate dead code from the existing call graph without re-parsing.

    Analyzes all `kind="function"` nodes in *graph* and flags those having
    zero incoming call-graph edges (`rel="func_call"`), while explicitly
    excluding known entry points (such as `main`, route handlers, dunder methods,
    and test functions).

    Node attributes are updated in-place:
      - `is_dead_code_candidate` (bool): True if flagged as dead code.
      - `dead_code_reason` (str | None): Diagnostic explanation.

    Args:
        graph: NetworkX DiGraph representing the codebase.

    Returns:
        List of dicts representing all candidate dead code functions with details.
    """
    candidates: list[dict[str, Any]] = []

    for node_id, attrs in graph.nodes(data=True):
        if attrs.get("kind") != "function":
            attrs["is_dead_code_candidate"] = False
            continue

        func_name = attrs.get("name", "")
        file_path = attrs.get("file", "") or attrs.get("path", "")

        # Count incoming call-graph edges (rel="func_call")
        incoming_calls = sum(
            1
            for _, _, edge_data in graph.in_edges(node_id, data=True)
            if edge_data.get("rel") == "func_call"
        )

        if incoming_calls == 0 and not is_known_entry_point(func_name, file_path):
            attrs["is_dead_code_candidate"] = True
            attrs["dead_code_reason"] = "Zero incoming call-graph edges (uncalled function)"
            candidate_info = {
                "node_id": node_id,
                "name": func_name,
                "file": file_path,
                "pagerank": attrs.get("pagerank", 0.0),
                "commit_count": attrs.get("commit_count", 0),
                "reason": attrs["dead_code_reason"],
            }
            candidates.append(candidate_info)
        else:
            attrs["is_dead_code_candidate"] = False

    logger.info("Dead code detection complete: found %d candidate(s)", len(candidates))
    return candidates


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def scan_repository(repo_root: Union[str, Path]) -> nx.DiGraph:
    """Walk *repo_root* and build one merged graph for the whole repository.

    For each `.py` file found (skipping noise directories), the function:

    1. Reads the file bytes.
    2. Calls :func:~app.services.ast_engine.extract_structure to get
       classes, imports, and method calls.
    3. Calls :func:~app.services.ast_engine.extract_call_edges to get
       intra-file function->function edges.
    4. Adds all nodes and edges scoped by relative file path to a single
       accumulator :class:`nx.DiGraph`.
    5. Computes PageRank centrality for all nodes in the accumulator graph
       and stores the scores as a `pagerank` node attribute.
    6. Runs :func:`detect_dead_code` to flag candidate dead code.

    Args:
        repo_root: Path to the root directory to scan.  Resolved to an
                   absolute path before walking.

    Returns:
        A single :class:`nx.DiGraph` whose nodes and edges represent the
        entire repository.  The graph carries a `repo_root` graph-level
        attribute, `pagerank` node attributes, and dead-code annotations.

    Raises:
        NotADirectoryError: If *repo_root* does not point at a directory.
    """
    root = Path(repo_root).resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"repo_root is not a directory: {root}")

    accumulator: nx.DiGraph = nx.DiGraph()
    accumulator.graph["repo_root"] = str(root)

    py_files = _iter_python_files(root)
    logger.info("scan_repository: found %d Python files under %s",
                len(py_files), root)

    for py_file in py_files:
        try:
            source = py_file.read_bytes()
        except OSError as exc:
            logger.warning("Could not read %s: %s - skipping", py_file, exc)
            continue

        try:
            sub = _build_file_subgraph(py_file, root, source)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to analyse %s: %s - skipping", py_file, exc)
            continue

        accumulator.update(sub)
        logger.debug("Added %d nodes, %d edges from %s",
                     sub.number_of_nodes(), sub.number_of_edges(),
                     py_file.relative_to(root))

    logger.info(
        "scan_repository: total %d nodes, %d edges",
        accumulator.number_of_nodes(),
        accumulator.number_of_edges(),
    )

    # -----------------------------------------------------------------------
    # PageRank Centrality
    # -----------------------------------------------------------------------
    if accumulator.number_of_nodes() > 0:
        try:
            try:
                scores = nx.pagerank(accumulator)
            except (ModuleNotFoundError, ImportError):
                from networkx.algorithms.link_analysis.pagerank_alg import (
                    _pagerank_python,
                )
                scores = _pagerank_python(accumulator)
            nx.set_node_attributes(accumulator, scores, "pagerank")
            logger.info("Computed PageRank centrality for %d nodes", len(scores))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to calculate PageRank: %s", exc)

    # -----------------------------------------------------------------------
    # Dead Code Candidate Detection
    # -----------------------------------------------------------------------
    detect_dead_code(accumulator)

    return accumulator


def attach_churn(graph: nx.DiGraph, clone_path: Union[str, Path]) -> None:
    """Compute git commit counts and attach `commit_count` to nodes in-place.

    Args:
        graph: NetworkX DiGraph whose nodes will be annotated.
        clone_path: Path to the root of the cloned git repository.
    """
    churn = count_file_commits(clone_path)
    for node_id, attrs in graph.nodes(data=True):
        file_path = attrs.get("path") or attrs.get("file")
        if file_path and file_path in churn:
            attrs["commit_count"] = churn[file_path]
        elif attrs.get("kind") == "file":
            attrs["commit_count"] = churn.get(attrs.get("path", ""), 0)


def graph_to_json(g: nx.DiGraph) -> dict:
    """Serialise *g* to a JSON-friendly dict with dead code summary.

    Output format::

        {
            "meta": {
                "node_count": <int>,
                "edge_count": <int>,
                "repo_root":  "<str>",
                "dead_code_count": <int>,
                "dead_code_candidates": [
                    {
                        "node_id": "<str>",
                        "name": "<str>",
                        "file": "<str>",
                        "pagerank": <float>,
                        "commit_count": <int>,
                        "reason": "<str>"
                    }
                ]
            },
            "nodes": [
                {
                    "id": "<node_id>",
                    "kind": "<kind>",
                    "pagerank": <float>,
                    "commit_count": <int | optional>,
                    "is_dead_code_candidate": <bool>,
                    ...
                },
                ...
            ],
            "edges": [
                {
                    "source": "<src>",
                    "target": "<dst>",
                    "rel": "<rel>",
                    "edge_type": "EXTRACTED" | "INFERRED"
                },
                ...
            ]
        }

    Args:
        g: A :class:`nx.DiGraph` produced by :func:`scan_repository` or any
           compatible graph.

    Returns:
        A plain Python dict that is directly `json.dumps`-able.
    """
    nodes = [
        {"id": node_id, **attrs}
        for node_id, attrs in g.nodes(data=True)
    ]
    edges = [
        {"source": src, "target": dst, **edge_attrs}
        for src, dst, edge_attrs in g.edges(data=True)
    ]
    meta: dict = {
        "node_count": g.number_of_nodes(),
        "edge_count": g.number_of_edges(),
    }
    if "repo_root" in g.graph:
        meta["repo_root"] = g.graph["repo_root"]

    dead_candidates = [
        {
            "node_id": n["id"],
            "name": n.get("name", ""),
            "file": n.get("file", "") or n.get("path", ""),
            "pagerank": n.get("pagerank", 0.0),
            "commit_count": n.get("commit_count", 0),
            "reason": n.get("dead_code_reason", "Zero incoming call-graph edges"),
        }
        for n in nodes
        if n.get("is_dead_code_candidate")
    ]
    meta["dead_code_count"] = len(dead_candidates)
    meta["dead_code_candidates"] = dead_candidates

    return {"meta": meta, "nodes": nodes, "edges": edges}


# ---------------------------------------------------------------------------
# Integration smoke-test  --  run with:
#   python -m parser.repo_walker          (scans app/ by default)
#   python -m parser.repo_walker <path>   (scans <path>)
# ---------------------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s  %(message)s",
    )

    _repo_root = Path(__file__).resolve().parent.parent / "app"
    if len(sys.argv) > 1:
        _repo_root = Path(sys.argv[1]).resolve()

    print(f"\n{'='*60}")
    print(f"  Scanning: {_repo_root}")
    print(f"{'='*60}\n")

    g = scan_repository(_repo_root)

    # ---- Summary -------------------------------------------------------------
    file_nodes   = [(n, d) for n, d in g.nodes(data=True) if d["kind"] == "file"]
    class_nodes  = [(n, d) for n, d in g.nodes(data=True) if d["kind"] == "class"]
    import_nodes = [(n, d) for n, d in g.nodes(data=True) if d["kind"] == "import"]
    call_nodes   = [(n, d) for n, d in g.nodes(data=True) if d["kind"] == "call_target"]
    func_nodes   = [(n, d) for n, d in g.nodes(data=True) if d["kind"] == "function"]

    print(f"Total nodes : {g.number_of_nodes()}")
    print(f"Total edges : {g.number_of_edges()}")
    print(f"  files     : {len(file_nodes)}")
    print(f"  classes   : {len(class_nodes)}")
    print(f"  imports   : {len(import_nodes)}")
    print(f"  calls     : {len(call_nodes)}")
    print(f"  functions : {len(func_nodes)}")

    # ---- Top PageRank Centrality Nodes ---------------------------------------
    print("\n-- Top 5 PageRank Centrality Nodes --")
    ranked = sorted(
        g.nodes(data=True),
        key=lambda item: item[1].get("pagerank", 0.0),
        reverse=True,
    )[:5]
    for node_id, attrs in ranked:
        print(f"  {attrs.get('pagerank', 0.0):.6f}  [{attrs.get('kind', 'unknown'):<11}]  {node_id}")

    # ---- Per-file breakdown --------------------------------------------------
    print("\n-- Per-file node inventory --")
    for file_id, _ in sorted(file_nodes, key=lambda x: x[0]):
        successors = list(g.successors(file_id))
        by_kind: dict[str, int] = {}
        for s in successors:
            k = g.nodes[s]["kind"]
            by_kind[k] = by_kind.get(k, 0) + 1
        summary = "  ".join(f"{k}={v}" for k, v in sorted(by_kind.items()))
        rel = file_id.replace("::file", "")
        print(f"  {rel:<45}  {summary}")

    # ---- Function call edges -------------------------------------------------
    func_edges = [
        (s, t) for s, t, d in g.edges(data=True) if d.get("rel") == "func_call"
    ]
    if func_edges:
        print("\n-- Intra-file function call edges --")
        for src, dst in func_edges:
            print(f"  {src}  ->  {dst}")
    else:
        print("\n-- No intra-file function call edges found --")

    # ---- graph_to_json round-trip --------------------------------------------
    print("\n-- graph_to_json round-trip --")
    data = graph_to_json(g)
    raw = json.dumps(data)
    print(f"  JSON length : {len(raw):,} bytes")
    print(f"  meta        : {data['meta']}")
    print("  first 3 nodes:")
    for node in data["nodes"][:3]:
        print(f"    {node}")

    # ---- Assertions ----------------------------------------------------------
    print("\n-- Assertions --")
    assert g.number_of_nodes() > 0, "Graph must not be empty"
    assert len(file_nodes) >= 1, "Expected at least one file node"
    assert {"nodes", "edges", "meta"} <= data.keys(), "graph_to_json missing keys"
    assert all("id" in n for n in data["nodes"]), "Every node must have an 'id'"
    assert all(
        "pagerank" in n and isinstance(n["pagerank"], (float, int))
        for n in data["nodes"]
    ), "Every node must have a numeric 'pagerank' score"
    assert all(
        "source" in e and "target" in e for e in data["edges"]
    ), "Every edge must have 'source' and 'target'"
    assert all(
        e.get("edge_type") == "EXTRACTED" for e in data["edges"]
    ), "Every edge must have edge_type='EXTRACTED'"
    file_ids = [n for n, d in g.nodes(data=True) if d["kind"] == "file"]
    assert len(file_ids) == len(set(file_ids)), "Duplicate file node IDs detected"
    print("[OK] All assertions passed.")
