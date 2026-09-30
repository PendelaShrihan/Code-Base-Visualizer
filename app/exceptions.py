"""
app/exceptions.py
-----------------
Custom domain exceptions for CodeBase Visualizer ingestion guardrails.
"""

from __future__ import annotations


class IngestionGuardrailError(Exception):
    """Base exception for all ingestion guardrail violations on untrusted external data."""


class CloneTimeoutError(IngestionGuardrailError):
    """Raised when repository cloning exceeds the allowed execution time limit."""


class RepoSizeLimitExceededError(IngestionGuardrailError):
    """Raised when a cloned repository exceeds the maximum storage threshold."""


class MaxFileCountExceededError(IngestionGuardrailError):
    """Raised when the repository contains more files than the allowable limit before parsing."""


class FileParseTimeoutError(IngestionGuardrailError):
    """Raised when Tree-sitter parsing or AST analysis of an individual file exceeds the allowed time limit."""
