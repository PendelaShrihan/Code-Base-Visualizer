"""
tests/test_graph_filter.py
--------------------------
Unit tests for filter_graph in parser/repo_walker.py.
Verifies removal of standard library imports, common built-ins, and isolates.
"""

from __future__ import annotations

import networkx as nx
import pytest

from parser.repo_walker import filter_graph


def test_filter_removes_all_import_nodes() -> None:
    """filter_graph must remove ALL import-kind nodes, not just stdlib ones.

    3rd-party imports (e.g. custom_pkg, fastapi, pydantic) clutter the
    semantic graph just as much as stdlib imports, so the blanket rule
    drops every node whose kind == 'import'.
    """
    g = nx.DiGraph()
    g.add_node("file.py::file", kind="file")
    g.add_node("file.py::import::os", kind="import", name="os")
    g.add_node("file.py::import::sys", kind="import", name="sys")
    g.add_node("file.py::import::typing.Optional", kind="import", name="typing.Optional")
    # 3rd-party import — should ALSO be removed under the new blanket rule
    g.add_node("file.py::import::custom_pkg", kind="import", name="custom_pkg")

    g.add_edge("file.py::file", "file.py::import::os", rel="imports")
    g.add_edge("file.py::file", "file.py::import::sys", rel="imports")
    g.add_edge("file.py::file", "file.py::import::typing.Optional", rel="imports")
    g.add_edge("file.py::file", "file.py::import::custom_pkg", rel="imports")

    filter_graph(g, remove_isolates=False)

    node_ids = set(g.nodes())
    # Stdlib imports gone
    assert "file.py::import::os" not in node_ids
    assert "file.py::import::sys" not in node_ids
    assert "file.py::import::typing.Optional" not in node_ids
    # 3rd-party import also gone (new blanket rule)
    assert "file.py::import::custom_pkg" not in node_ids
    # File node itself is preserved
    assert "file.py::file" in node_ids



def test_filter_removes_call_targets_and_builtins() -> None:
    g = nx.DiGraph()
    g.add_node("file.py::file", kind="file")
    g.add_node("file.py::call::print", kind="call_target", name="print")
    g.add_node("file.py::call::my_custom_func", kind="call_target", name="my_custom_func")
    g.add_node("file.py::func::print", kind="function", name="print")
    g.add_node("file.py::func::my_real_func", kind="function", name="my_real_func")

    g.add_edge("file.py::file", "file.py::call::print", rel="calls")
    g.add_edge("file.py::file", "file.py::call::my_custom_func", rel="calls")
    g.add_edge("file.py::file", "file.py::func::print", rel="contains")
    g.add_edge("file.py::file", "file.py::func::my_real_func", rel="contains")

    filter_graph(g, remove_isolates=False)

    node_ids = set(g.nodes())
    # All call_target nodes are stripped
    assert "file.py::call::print" not in node_ids
    assert "file.py::call::my_custom_func" not in node_ids
    # Built-in function is stripped
    assert "file.py::func::print" not in node_ids
    # Custom real function is preserved
    assert "file.py::func::my_real_func" in node_ids


def test_filter_removes_isolates() -> None:
    g = nx.DiGraph()
    g.add_node("connected_1", kind="file")
    g.add_node("connected_2", kind="function")
    g.add_edge("connected_1", "connected_2")

    g.add_node("lonely_orphan", kind="function")

    filter_graph(g, remove_isolates=True)

    assert "connected_1" in g
    assert "connected_2" in g
    assert "lonely_orphan" not in g


def test_filter_prunes_level_3_leaves_when_large() -> None:
    g = nx.DiGraph()
    root = "main.py::file"
    g.add_node(root, kind="file", level=2)

    # Add 250 nodes so total > 200
    for i in range(250):
        node_id = f"node_{i}"
        # Make node a level 3 leaf (degree 1)
        g.add_node(node_id, kind="function", level=3)
        g.add_edge(root, node_id)

    assert g.number_of_nodes() > 200
    filter_graph(g, remove_isolates=True)

    # Level 3 leaves should be pruned down
    assert g.number_of_nodes() <= 200
