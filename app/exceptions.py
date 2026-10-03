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


class RateLimitExceededError(Exception):
    """Raised when a client exceeds the allowed request frequency for an endpoint."""

    def __init__(self, message: str, retry_after: int = 60, max_requests: int = 5, window_seconds: int = 3600):
        super().__init__(message)
        self.message = message
        self.retry_after = retry_after
        self.max_requests = max_requests
        self.window_seconds = window_seconds

