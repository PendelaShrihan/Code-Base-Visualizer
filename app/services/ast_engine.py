"""
ast_engine.py
Parses Python source files into ASTs using Tree-sitter and provides
utilities to extract structural information (e.g. function names).
"""

from __future__ import annotations

from pathlib import Path
from typing import Generator

import tree_sitter_python as tspython
from tree_sitter import Language, Node, Parser


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
