"""
tests/test_hierarchical_graph.py
--------------------------------
Unit tests for node filtering and Cytoscape compound graph serialization:
1. Exclude/flag standalone tests, migration scripts, and utility scripts.
2. Generate parent compound containers (folder:<module>).
3. Assign parent references on leaf nodes.
4. Ensure separated nodes and edges in Cytoscape JSON format.
"""

from __future__ import annotations

import networkx as nx
import pytest

from parser.repo_walker import (
    filter_graph,
    is_test_migration_or_script,
    serialize_to_cytoscape,
    graph_to_cytoscape_json,
    graph_to_json,
)


def test_is_test_migration_or_script() -> None:
    # Test paths
    assert is_test_migration_or_script("tests/test_auth.py::file", {"path": "tests/test_auth.py"})
    assert is_test_migration_or_script("app/tests/unit.py::file", {"path": "app/tests/unit.py"})
    assert is_test_migration_or_script("test_auth.py::file", {"path": "test_auth.py"})
    assert is_test_migration_or_script("auth_test.py::file", {"path": "auth_test.py"})

    # Migration paths
    assert is_test_migration_or_script("migrations/versions/001.py::file", {"path": "migrations/versions/001.py"})
    assert is_test_migration_or_script("alembic/env.py::file", {"path": "alembic/env.py"})

    # Script paths
    assert is_test_migration_or_script("scripts/seed_db.py::file", {"path": "scripts/seed_db.py"})
    assert is_test_migration_or_script("script/deploy.py::file", {"path": "script/deploy.py"})

    # Primary code (should NOT be flagged)
    assert not is_test_migration_or_script("agent/action_engine.py::file", {"path": "agent/action_engine.py"})
    assert not is_test_migration_or_script("app/services/git_service.py::file", {"path": "app/services/git_service.py"})


def test_filter_excludes_tests_migrations_scripts() -> None:
    g = nx.DiGraph()
    # Legitimate services
    g.add_node("agent/main.py::file", kind="file", path="agent/main.py", level=2)
    g.add_node("agent/action_engine.py::file", kind="file", path="agent/action_engine.py", level=2)
    g.add_edge("agent/main.py::file", "agent/action_engine.py::file", rel="depends_on")

    # Tests, migrations, and utility scripts
    g.add_node("tests/test_agent.py::file", kind="file", path="tests/test_agent.py", level=2)
    g.add_node("scripts/migrate_db.py::file", kind="file", path="scripts/migrate_db.py", level=2)
    g.add_node("migrations/0001_init.py::file", kind="file", path="migrations/0001_init.py", level=2)

    g.add_edge("tests/test_agent.py::file", "agent/action_engine.py::file", rel="depends_on")
    g.add_edge("scripts/migrate_db.py::file", "agent/main.py::file", rel="depends_on")

    # Filter with default remove_tests_and_scripts=True
    filter_graph(g, remove_isolates=False, remove_tests_and_scripts=True)

    nodes = set(g.nodes())
    assert "agent/main.py::file" in nodes
    assert "agent/action_engine.py::file" in nodes

    assert "tests/test_agent.py::file" not in nodes
    assert "scripts/migrate_db.py::file" not in nodes
    assert "migrations/0001_init.py::file" not in nodes


def test_filter_flags_tests_when_removal_disabled() -> None:
    g = nx.DiGraph()
    g.add_node("tests/test_unit.py::file", kind="file", path="tests/test_unit.py", level=2)
    g.add_node("app/main.py::file", kind="file", path="app/main.py", level=2)
    g.add_edge("tests/test_unit.py::file", "app/main.py::file", rel="depends_on")

    filter_graph(g, remove_isolates=False, remove_tests_and_scripts=False)

    assert "tests/test_unit.py::file" in g
    assert g.nodes["tests/test_unit.py::file"].get("is_test_or_script") is True
    assert g.nodes["app/main.py::file"].get("is_test_or_script") is not True


def test_serialize_to_cytoscape_compound_format() -> None:
    g = nx.DiGraph()
    # Add nodes from two modules: agent and core
    g.add_node("agent/main.py::file", kind="file", path="agent/main.py", label="main.py")
    g.add_node("agent/action_engine.py::file", kind="file", path="agent/action_engine.py", label="action_engine.py")
    g.add_node("core/config.py::file", kind="file", path="core/config.py", label="config.py")

    # Add edges
    g.add_edge("agent/main.py::file", "agent/action_engine.py::file", rel="calls")
    g.add_edge("agent/action_engine.py::file", "core/config.py::file", rel="depends_on")

    result = serialize_to_cytoscape(g)

    # 1. Separated nodes and edges
    assert "nodes" in result
    assert "edges" in result
    assert isinstance(result["nodes"], list)
    assert isinstance(result["edges"], list)

    # 2. Parent compound containers
    parent_nodes = [n for n in result["nodes"] if n["data"].get("isParent")]
    parent_ids = {n["data"]["id"] for n in parent_nodes}
    assert "folder:agent" in parent_ids
    assert "folder:core" in parent_ids

    agent_folder = next(n["data"] for n in parent_nodes if n["data"]["id"] == "folder:agent")
    assert agent_folder["label"] == "Agent Module"
    assert agent_folder["isParent"] is True

    # 3. Leaf node parent references
    leaf_nodes = [n for n in result["nodes"] if not n["data"].get("isParent")]
    leaf_by_id = {n["data"]["id"]: n["data"] for n in leaf_nodes}

    assert "agent/action_engine.py" in leaf_by_id
    action_engine = leaf_by_id["agent/action_engine.py"]
    assert action_engine["parent"] == "folder:agent"
    assert action_engine["label"] == "action_engine.py"
    assert action_engine["type"] == "service"  # 'service' inferred from name

    assert "core/config.py" in leaf_by_id
    config_node = leaf_by_id["core/config.py"]
    assert config_node["parent"] == "folder:core"

    # 4. Edges correctly connected to leaf nodes with label
    assert len(result["edges"]) == 2
    edge_pairs = {(e["data"]["source"], e["data"]["target"], e["data"]["label"]) for e in result["edges"]}
    assert ("agent/main.py", "agent/action_engine.py", "CALLS") in edge_pairs
    assert ("agent/action_engine.py", "core/config.py", "DEPENDS_ON") in edge_pairs


def test_graph_to_json_cytoscape_flag() -> None:
    g = nx.DiGraph()
    g.add_node("agent/worker.py::file", kind="file", path="agent/worker.py")
    res = graph_to_json(g, cytoscape_format=True)
    assert "nodes" in res
    assert "edges" in res
    assert any(n["data"].get("id") == "folder:agent" for n in res["nodes"])
