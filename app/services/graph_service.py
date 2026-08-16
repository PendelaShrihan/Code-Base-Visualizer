"""
graph_service.py
Converts a file path + its extract_structure() result into a populated
NetworkX directed graph (nx.DiGraph).

Node types (stored as node attribute ``kind``):
    "file"         – the file being analysed (always the root node)
    "class"        – a class defined in the file
    "import"       – an imported name / module
    "call_target"  – the callee side of a method / attribute call (obj.method)
    "function"     – a function defined in this file (caller or callee side
                     of an intra-file call)

Edge types (stored as edge attribute ``rel``):
    "contains"   – file → class  (the file defines this class)
    "imports"    – file → import (the file imports this name)
    "calls"      – file → call_target (the file makes this obj.method call)
    "func_call"  – function → function (intra-file call graph edge)
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import networkx as nx

from app.services.ast_engine import extract_call_edges, extract_structure


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_graph(
    file_path: Union[str, Path],
    structure: dict[str, list[str]],
    source_code: bytes | None = None,
) -> nx.DiGraph:
    """
    Build a directed graph from a file's extracted structure.

    The graph has a single root node whose ID is the string representation of
    *file_path*.  Every class, import, and call target found in *structure*
    becomes a child node connected to that root with a typed edge.

    Intra-file function-to-function call edges (``rel="func_call"``) are also
    added when *source_code* is supplied.  Each locally-defined function that
    participates in at least one such edge is added as a ``kind="function"``
    node.

    Args:
        file_path:    Path to the analysed source file (used as the root node
                      ID and stored in the ``path`` node attribute).
        structure:    Dict returned by
                      :func:`~app.services.ast_engine.extract_structure` with
                      keys ``"classes"``, ``"method_calls"``, and
                      ``"imports"``.
        source_code:  Optional raw source bytes.  When provided,
                      :func:`~app.services.ast_engine.extract_call_edges` is
                      called and the resulting function→function edges are
                      added to the graph.

    Returns:
        A populated :class:`nx.DiGraph`.  Every node carries a ``kind``
        attribute; every edge carries a ``rel`` attribute.

    Example::

        >>> from pathlib import Path
        >>> from app.services.ast_engine import extract_structure_from_file
        >>> from app.services.graph_service import build_graph
        >>> path = Path("app/services/git_service.py")
        >>> src = path.read_bytes()
        >>> g = build_graph(path, extract_structure_from_file(path), src)
        >>> "app/services/git_service.py" in g.nodes
        True
    """
    g: nx.DiGraph = nx.DiGraph()

    file_id = str(file_path)

    # -- root node ------------------------------------------------------------
    g.add_node(file_id, kind="file", path=file_id)

    # -- classes --------------------------------------------------------------
    for cls_name in structure.get("classes", []):
        node_id = f"class::{cls_name}"
        g.add_node(node_id, kind="class", name=cls_name)
        g.add_edge(file_id, node_id, rel="contains")

    # -- imports --------------------------------------------------------------
    for imp in structure.get("imports", []):
        node_id = f"import::{imp}"
        g.add_node(node_id, kind="import", name=imp)
        g.add_edge(file_id, node_id, rel="imports")

    # -- method / attribute calls ---------------------------------------------
    seen_calls: set[str] = set()
    for call in structure.get("method_calls", []):
        node_id = f"call::{call}"
        if node_id not in seen_calls:
            seen_calls.add(node_id)
            g.add_node(node_id, kind="call_target", name=call)
        g.add_edge(file_id, node_id, rel="calls")

    # -- intra-file function→function call edges ------------------------------
    if source_code is not None:
        for caller, callee in extract_call_edges(source_code):
            caller_id = f"func::{caller}"
            callee_id = f"func::{callee}"
            if caller_id not in g:
                g.add_node(caller_id, kind="function", name=caller)
            if callee_id not in g:
                g.add_node(callee_id, kind="function", name=callee)
            g.add_edge(caller_id, callee_id, rel="func_call")

    return g


def build_graph_from_file(file_path: Union[str, Path]) -> nx.DiGraph:
    """
    Convenience wrapper: parse *file_path* and build its graph in one call.

    Args:
        file_path: Filesystem path to a ``.py`` file.

    Returns:
        Same as :func:`build_graph`.
    """
    path = Path(file_path)
    source_code = path.read_bytes()
    structure = extract_structure(source_code)
    return build_graph(path, structure, source_code)


# ---------------------------------------------------------------------------
# Quick smoke-test -- run with:  python -m app.services.graph_service
# ---------------------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover
    import sys

    _here = Path(__file__).parent
    _target = _here / "git_service.py"

    if not _target.exists():
        print(f"[ERROR] Test file not found: {_target}", file=sys.stderr)
        sys.exit(1)

    print(f"Building graph for: {_target}")
    print("=" * 60)

    g = build_graph_from_file(_target)
    file_node = str(_target)

    # -- Nodes ----------------------------------------------------------------
    print(f"\n-- Nodes ({g.number_of_nodes()}) --")
    for node, attrs in g.nodes(data=True):
        print(f"  [{attrs['kind']:12}]  {node}")

    # -- Edges ----------------------------------------------------------------
    print(f"\n-- Edges ({g.number_of_edges()}) --")
    for src, dst, attrs in g.edges(data=True):
        print(f"  {Path(src).name}  --[{attrs['rel']}]-->  {dst}")

    # -- Successors of the file node ------------------------------------------
    print(f"\n-- Successors of  {Path(file_node).name}  (direct children) --")
    for s in g.successors(file_node):
        print(f"  -> {s}  (rel={g[file_node][s]['rel']})")

    # -- Predecessors of a specific import ------------------------------------
    sample_import = next(
        (n for n, d in g.nodes(data=True) if d["kind"] == "import"), None
    )
    if sample_import:
        print(f"\n-- Predecessors of  {sample_import}  (who imports it?) --")
        for p in g.predecessors(sample_import):
            print(f"  <- {p}")

    # -- Predecessors of a specific call_target -------------------------------
    clone_node = next(
        (n for n, d in g.nodes(data=True)
         if d["kind"] == "call_target" and "clone_from" in n),
        None,
    )
    if clone_node:
        print(f"\n-- Predecessors of  {clone_node}  (who calls it?) --")
        for p in g.predecessors(clone_node):
            print(f"  <- {p}")

    # -- Structural assertions ------------------------------------------------
    print("\n-- Assertions --")
    assert file_node in g.nodes, "Root file node missing"
    assert not any(
        d["kind"] == "class" for _, d in g.nodes(data=True)
    ), "git_service.py has no classes -- graph should have none"
    assert any(
        "clone_from" in n for n in g.nodes
    ), "Expected call::git.Repo.clone_from in graph"
    assert any(
        d["kind"] == "import" and "shutil" in n
        for n, d in g.nodes(data=True)
    ), "Expected import::shutil in graph"
    print("[OK] All assertions passed.")
