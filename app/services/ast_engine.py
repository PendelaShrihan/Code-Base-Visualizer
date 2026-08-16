"""
ast_engine.py
Parses Python source files into ASTs using Tree-sitter and provides
utilities to extract structural information (e.g. function names,
class names, method calls, and imports).
"""

from __future__ import annotations

from pathlib import Path
from typing import Generator

import tree_sitter_python as tspython
from tree_sitter import Language, Node, Parser, Query, QueryCursor


# ---------------------------------------------------------------------------
# Language + Parser setup (module-level singletons — cheap to reuse)
# ---------------------------------------------------------------------------

PY_LANGUAGE: Language = Language(tspython.language())

_parser: Parser = Parser(PY_LANGUAGE)


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _walk(node: Node) -> Generator[Node, None, None]:
    """Depth-first traversal that yields every node in the tree."""
    yield node
    for child in node.children:
        yield from _walk(child)


def _get_child_by_field(node: Node, field: str) -> Node | None:
    """Return the first child of *node* that sits in *field*, or None."""
    return node.child_by_field_name(field)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse(source_code: bytes) -> Node:
    """
    Parse *source_code* and return the root node of the syntax tree.

    Args:
        source_code: UTF-8-encoded Python source.

    Returns:
        The root ``Node`` of the Tree-sitter parse tree.
    """
    tree = _parser.parse(source_code)
    return tree.root_node


def extract_function_names(source_code: bytes) -> list[str]:
    """
    Return every top-level *and* nested function name declared in
    *source_code*.

    The extractor walks the full AST and collects every
    ``function_definition`` node.  For each such node it reads the
    ``name`` field — a Tree-sitter named field that always points at the
    ``identifier`` child carrying the actual function name text.

    Args:
        source_code: UTF-8-encoded Python source.

    Returns:
        A list of function name strings in the order they appear in the
        source, including nested/inner functions.

    Example::

        >>> src = b"def hello(): pass\\ndef world(): pass"
        >>> extract_function_names(src)
        ['hello', 'world']
    """
    root = parse(source_code)
    names: list[str] = []

    for node in _walk(root):
        if node.type == "function_definition":
            name_node = _get_child_by_field(node, "name")
            if name_node is not None:
                names.append(name_node.text.decode("utf-8"))

    return names


def extract_function_names_from_file(path: str | Path) -> list[str]:
    """
    Convenience wrapper: read a file from disk and extract function names.

    Args:
        path: Filesystem path to a ``.py`` file.

    Returns:
        Same as :func:`extract_function_names`.
    """
    source = Path(path).read_bytes()
    return extract_function_names(source)


# ---------------------------------------------------------------------------
# Combined structural query
# ---------------------------------------------------------------------------

# A single Tree-sitter query that captures three kinds of structural nodes:
#
#   @class.name   – identifier inside a class_definition's "name" field
#   @call.method  – attribute expression that is the *function* of a call node,
#                   matching the two-level  obj.method(...)  pattern (e.g.
#                   git.Repo.clone_from, dest.mkdir, Path(...))
#   @import.name  – dotted name / alias in a plain  import X  statement
#   @import_from.module / @import_from.name – module + imported names from
#                   "from X import Y" statements
#
_STRUCTURE_QUERY_SRC: str = """
; ── class definitions ────────────────────────────────────────────────────────
(class_definition
  name: (identifier) @class.name)

; ── method / attribute calls  (obj.attr(...) pattern) ───────────────────────
; Matches any call whose callee is an attribute access, e.g.:
;   dest.mkdir(parents=True)   →  dest.mkdir
;   git.Repo.clone_from(...)   →  git.Repo.clone_from
;   Path(tempfile.gettempdir()) is a plain identifier call, NOT matched here
(call
  function: (attribute) @call.method)

; ── plain  import X [as Y], import a.b.c ────────────────────────────────────
(import_statement
  name: [(dotted_name) (aliased_import)] @import.name)

; ── from X import Y [as Z] ──────────────────────────────────────────────────
(import_from_statement
  module_name: (dotted_name) @import_from.module
  name: [(dotted_name) (aliased_import)] @import_from.name)
"""

_STRUCTURE_QUERY: Query = Query(PY_LANGUAGE, _STRUCTURE_QUERY_SRC)


# ---------------------------------------------------------------------------
# extract_structure
# ---------------------------------------------------------------------------

def extract_structure(source_code: bytes) -> dict[str, list[str]]:
    """
    Run the combined structural query and return a summary dict.

    The returned dictionary has three keys:

    * ``"classes"``     – names of every ``class`` definition in the file.
    * ``"method_calls"``– text of every ``obj.method`` attribute expression
                          that is used as the callee of a call node, in
                          source order.  Nested attribute chains (e.g.
                          ``git.Repo.clone_from``) are returned as-is.
    * ``"imports"``     – deduplicated list of module / name strings from
                          *both* ``import X`` and ``from X import Y`` forms.

    Args:
        source_code: UTF-8-encoded Python source.

    Returns:
        A dict with keys ``"classes"``, ``"method_calls"``, and
        ``"imports"``.  Each value is a list of strings.

    Example::

        >>> src = Path("git_service.py").read_bytes()
        >>> s = extract_structure(src)
        >>> "clone_repository" not in s["classes"]  # it's a function, not a class
        True
        >>> any("clone_from" in mc for mc in s["method_calls"])
        True
    """
    tree = _parser.parse(source_code)
    root = tree.root_node

    classes: list[str] = []
    method_calls: list[str] = []
    # Use a dict to preserve first-seen order while deduplicating imports.
    imports_seen: dict[str, None] = {}

    for capture_name, nodes in QueryCursor(_STRUCTURE_QUERY).captures(root).items():
        for node in nodes:
            text = node.text.decode("utf-8")

            if capture_name == "class.name":
                classes.append(text)

            elif capture_name == "call.method":
                method_calls.append(text)

            elif capture_name in ("import.name", "import_from.module", "import_from.name"):
                imports_seen.setdefault(text, None)

    return {
        "classes": classes,
        "method_calls": method_calls,
        "imports": list(imports_seen),
    }


def extract_structure_from_file(path: str | Path) -> dict[str, list[str]]:
    """
    Convenience wrapper: read a file from disk and return its structure.

    Args:
        path: Filesystem path to a ``.py`` file.

    Returns:
        Same as :func:`extract_structure`.
    """
    source = Path(path).read_bytes()
    return extract_structure(source)


# ---------------------------------------------------------------------------
# Call-edge extraction
# ---------------------------------------------------------------------------

def extract_call_edges(source_code: bytes) -> list[tuple[str, str]]:
    """
    Return a list of ``(caller, callee)`` edges for every *intra-file*
    function call found in *source_code*.

    The algorithm uses a single DFS over the Tree-sitter AST with a
    push/pop stack to keep track of which ``function_definition`` the
    current node belongs to.  A call is emitted only when:

    * the call's ``function`` field is a plain ``identifier`` (i.e. *not*
      an attribute chain like ``obj.method``), **and**
    * that identifier names a function defined in the same source file.

    This deliberately excludes built-ins (``print``, ``len``, …) and any
    third-party / stdlib functions that are not locally defined.

    Args:
        source_code: UTF-8-encoded Python source.

    Returns:
        Ordered list of ``(caller_name, callee_name)`` tuples.  Duplicate
        edges (same caller calling the same callee multiple times) are
        **not** deduplicated — the caller can deduplicate if needed.

    Example::

        >>> src = (
        ...     b"def helper():\\n    pass\\n\\n"
        ...     b"def main():\\n    helper()\\n    print('done')\\n"
        ... )
        >>> extract_call_edges(src)
        [('main', 'helper')]
    """
    root = parse(source_code)

    # Build the set of locally-defined function names so we can filter out
    # built-ins and external calls in O(1).
    known: set[str] = set(extract_function_names(source_code))

    edges: list[tuple[str, str]] = []
    # Stack of function names — top is the innermost enclosing function.
    scope_stack: list[str] = []

    def _dfs(node: Node) -> None:
        # Track entry / exit of function definitions.
        entered_scope = False
        if node.type == "function_definition":
            name_node = _get_child_by_field(node, "name")
            if name_node is not None:
                scope_stack.append(name_node.text.decode("utf-8"))
                entered_scope = True

        # Detect a plain-identifier call while we are inside a function.
        if node.type == "call" and scope_stack:
            func_node = _get_child_by_field(node, "function")
            if func_node is not None and func_node.type == "identifier":
                callee = func_node.text.decode("utf-8")
                if callee in known:
                    edges.append((scope_stack[-1], callee))

        for child in node.children:
            _dfs(child)

        if entered_scope:
            scope_stack.pop()

    _dfs(root)
    return edges


def extract_call_edges_from_file(path: str | Path) -> list[tuple[str, str]]:
    """
    Convenience wrapper: read a file from disk and extract call edges.

    Args:
        path: Filesystem path to a ``.py`` file.

    Returns:
        Same as :func:`extract_call_edges`.
    """
    source = Path(path).read_bytes()
    return extract_call_edges(source)


# ---------------------------------------------------------------------------
# Quick smoke-test — run with:  python -m app.services.ast_engine
# ---------------------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover
    import json
    import sys

    # Resolve git_service.py relative to *this* file so the test works
    # regardless of the working directory.
    _here = Path(__file__).parent
    _target = _here / "git_service.py"

    if not _target.exists():
        print(f"[ERROR] Test file not found: {_target}", file=sys.stderr)
        sys.exit(1)

    print(f"Analysing: {_target}\n")
    result = extract_structure_from_file(_target)
    print(json.dumps(result, indent=2))

    # Minimal assertions so the script exits non-zero on regression.
    assert result["classes"] == [], (
        f"Expected no classes in git_service.py, got: {result['classes']}"
    )
    assert any("clone_from" in mc for mc in result["method_calls"]), (
        "Expected git.Repo.clone_from in method_calls"
    )
    assert any("mkdir" in mc for mc in result["method_calls"]), (
        "Expected dest.mkdir in method_calls"
    )
    assert any("shutil" in imp for imp in result["imports"]), (
        "Expected 'shutil' in imports"
    )
    assert any("pathlib" in imp for imp in result["imports"]), (
        "Expected 'pathlib' in imports"
    )
    print("\n[OK] All assertions passed.")

    # -----------------------------------------------------------------------
    # Smoke-test: extract_call_edges on a synthetic snippet
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Smoke-test: extract_call_edges")
    print("=" * 60)

    _SNIPPET = (
        b"def helper():\n    pass\n\n"
        b"def main():\n    helper()\n    print('done')  # built-in -- NOT an edge\n"
    )

    edges = extract_call_edges(_SNIPPET)
    print(f"Edges found: {edges}")

    assert edges == [("main", "helper")], (
        f"Expected [(\"main\", \"helper\")], got: {edges}"
    )
    print("[OK] extract_call_edges smoke-test passed.")

    # Also exercise the real file.
    real_edges = extract_call_edges_from_file(_target)
    print(f"\nCall edges in {_target.name}: {real_edges}")
