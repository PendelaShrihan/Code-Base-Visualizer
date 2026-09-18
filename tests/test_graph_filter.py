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


def test_filter_removes_stdlib_imports() -> None:
    g = nx.DiGraph()
    g.add_node("file.py::file", kind="file")
    g.add_node("file.py::import::os", kind="import", name="os")
    g.add_node("file.py::import::sys", kind="import", name="sys")
    g.add_node("file.py::import::typing.Optional", kind="import", name="typing.Optional")
    g.add_node("file.py::import::custom_pkg", kind="import", name="custom_pkg")

    g.add_edge("file.py::file", "file.py::import::os", rel="imports")
    g.add_edge("file.py::file", "file.py::import::sys", rel="imports")
    g.add_edge("file.py::file", "file.py::import::typing.Optional", rel="imports")
    g.add_edge("file.py::file", "file.py::import::custom_pkg", rel="imports")

    filter_graph(g, remove_isolates=False)

    node_ids = set(g.nodes())
    assert "file.py::import::os" not in node_ids
    assert "file.py::import::sys" not in node_ids
    assert "file.py::import::typing.Optional" not in node_ids
    assert "file.py::import::custom_pkg" in node_ids
    assert "file.py::file" in node_ids


def test_filter_removes_builtins_and_stdlib_calls() -> None:
    g = nx.DiGraph()
    g.add_node("file.py::file", kind="file")
    g.add_node("file.py::call::print", kind="call_target", name="print")
    g.add_node("file.py::call::len", kind="call_target", name="len")
    g.add_node("file.py::call::dict", kind="call_target", name="dict")
    g.add_node("file.py::call::my_custom_func", kind="call_target", name="my_custom_func")

    g.add_edge("file.py::file", "file.py::call::print", rel="calls")
    g.add_edge("file.py::file", "file.py::call::len", rel="calls")
    g.add_edge("file.py::file", "file.py::call::dict", rel="calls")
    g.add_edge("file.py::file", "file.py::call::my_custom_func", rel="calls")

    filter_graph(g, remove_isolates=False)

    node_ids = set(g.nodes())
    assert "file.py::call::print" not in node_ids
    assert "file.py::call::len" not in node_ids
    assert "file.py::call::dict" not in node_ids
    assert "file.py::call::my_custom_func" in node_ids


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
