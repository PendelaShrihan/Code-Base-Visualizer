"""
rag/llm_engine.py
-----------------
Local LLM Service Integration & Client Orchestration for CodeBase Visualizer.

Architecture Overview
---------------------
This module bridges high-density Graph RAG prompts (produced by :mod:`rag.prompt_builder`)
with local open-weights language models served via **Ollama** (e.g., CodeLlama, Llama 3,
Qwen 2.5 Coder, Mistral, DeepSeek-Coder).

It implements enterprise-grade **LLM Client Orchestration**:
1. **Connection Lifecycle & Health Checking**: Probing Ollama daemon state, validating model
   availability, and offering actionable diagnostics.
2. **Streaming & Non-Streaming Inference**: Full support for real-time token streaming
   via Server-Sent Events/JSON streams, alongside atomic non-streaming chat requests.
3. **Resilience & Fault Tolerance**: Decorator-driven exponential backoff with jitter for
   transient transport errors, granular exception taxonomy, and fail-safe recovery.
4. **Citation Extraction & Grounding Verification**: Enforcing anti-hallucination compliance
   by parsing ``[file_path::func::<name>]`` citations from the model output and benchmarking
   them against vector code anchors and graph neighbor context.

Prompt-to-Explanation Pipeline
------------------------------
::

    ┌────────────────────────────────────────────────────────────────────────┐
    │ StructuredPrompt (from rag.prompt_builder)                             │
    │   • System Prompt: Persona, Grounding Directives, Citation Rules       │
    │   • User Prompt  : <repository_context> + <user_query>                 │
    │   • Metadata     : included_anchors, included_neighbors                │
    └───────────────────────────────────┬────────────────────────────────────┘
                                        │
                                        ▼
    ┌────────────────────────────────────────────────────────────────────────┐
    │ OllamaClient Orchestrator (rag.llm_engine)                             │
    │   • Endpoint: POST /api/chat                                           │
    │   • Payloads: {model, messages: [{role, content}], options, stream}    │
    │   • Retry Policy: Exponential backoff with jitter                      │
    └───────────────────────────────────┬────────────────────────────────────┘
                                        │
                                        ▼
    ┌────────────────────────────────────────────────────────────────────────┐
    │ Citation & Grounding Adherence Evaluator                               │
    │   • Extracts: [path::func::<name>] tags                                │
    │   • Matches against: prompt.included_anchors & neighbors               │
    │   • Calculates Grounding Adherence Score (0.0 – 1.0)                   │
    └───────────────────────────────────┬────────────────────────────────────┘
                                        │
                                        ▼
    ┌────────────────────────────────────────────────────────────────────────┐
    │ SynthesisResult                                                        │
    │   • content: Markdown architectural narrative                          │
    │   • citations: Grounded citations vs Hallucinations                    │
    │   • token_usage: prompt_eval_count, eval_count, tokens/sec             │
    └────────────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import httpx

if TYPE_CHECKING:
    from rag.hybrid_retriever import HybridResult
    from rag.prompt_builder import StructuredPrompt
else:
    HybridResult = dict[str, Any]
    StructuredPrompt = Any

# Ensure project root is on sys.path for CLI execution
_current_dir = str(Path(__file__).resolve().parent)
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants & Defaults
# ---------------------------------------------------------------------------

DEFAULT_OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "codellama")
DEFAULT_OLLAMA_TIMEOUT: float = float(os.getenv("OLLAMA_TIMEOUT", "120.0"))
DEFAULT_TEMPERATURE: float = 0.2
DEFAULT_TOP_P: float = 0.9
DEFAULT_MAX_TOKENS: int = 2048

# Citation Regex: matches [path/to/file.ext::func::func_name] with optional angle brackets
CITATION_REGEX: re.Pattern = re.compile(
    r"\[([a-zA-Z0-9_\-./\\]+)::func::<?([a-zA-Z0-9_]+)>?\]"
)

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class LLMEngineError(Exception):
    """Base exception for all LLM engine and orchestration errors."""


class OllamaConnectionError(LLMEngineError):
    """Raised when Ollama daemon is unreachable or offline."""


class OllamaModelNotFoundError(LLMEngineError):
    """Raised when the requested model is not found in the local Ollama instance."""


class OllamaTimeoutError(LLMEngineError):
    """Raised when an inference request times out."""


class OllamaResponseError(LLMEngineError):
    """Raised when Ollama returns an HTTP error or malformed payload."""


# ---------------------------------------------------------------------------
# Configuration & Metrics Data Structures
# ---------------------------------------------------------------------------

@dataclass
class OllamaConfig:
    """Configuration parameters for the Ollama client."""

    base_url: str = DEFAULT_OLLAMA_BASE_URL
    model: str = DEFAULT_OLLAMA_MODEL
    temperature: float = DEFAULT_TEMPERATURE
    top_p: float = DEFAULT_TOP_P
    num_predict: int = DEFAULT_MAX_TOKENS
    timeout: float = DEFAULT_OLLAMA_TIMEOUT
    max_retries: int = 3
    backoff_factor: float = 0.5
    keep_alive: str = "5m"

    @property
    def chat_endpoint(self) -> str:
        return f"{self.base_url.rstrip('/')}/api/chat"

    @property
    def generate_endpoint(self) -> str:
        return f"{self.base_url.rstrip('/')}/api/generate"

    @property
    def tags_endpoint(self) -> str:
        return f"{self.base_url.rstrip('/')}/api/tags"

    def to_options_dict(self) -> dict[str, Any]:
        """Convert hyperparameter configuration into Ollama API options payload."""
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "num_predict": self.num_predict,
        }


@dataclass
class TokenUsage:
    """Inference timing and token generation telemetry."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    eval_duration_ms: float = 0.0
    prompt_eval_duration_ms: float = 0.0
    total_duration_ms: float = 0.0
    tokens_per_second: float = 0.0

    @classmethod
    def from_ollama_response(cls, data: dict[str, Any], fallback_wall_time_ms: float = 0.0) -> TokenUsage:
        """Parse token telemetry from Ollama response metadata."""
        prompt_tokens = int(data.get("prompt_eval_count", 0))
        completion_tokens = int(data.get("eval_count", 0))
        total_tokens = prompt_tokens + completion_tokens

        # Ollama returns durations in nanoseconds
        eval_dur_ms = data.get("eval_duration", 0) / 1_000_000.0
        prompt_eval_dur_ms = data.get("prompt_eval_duration", 0) / 1_000_000.0
        total_dur_ms = data.get("total_duration", 0) / 1_000_000.0

        if total_dur_ms <= 0:
            total_dur_ms = fallback_wall_time_ms

        tps = 0.0
        if eval_dur_ms > 0 and completion_tokens > 0:
            tps = (completion_tokens / eval_dur_ms) * 1000.0
        elif total_dur_ms > 0 and completion_tokens > 0:
            tps = (completion_tokens / total_dur_ms) * 1000.0

        return cls(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            eval_duration_ms=round(eval_dur_ms, 2),
            prompt_eval_duration_ms=round(prompt_eval_dur_ms, 2),
            total_duration_ms=round(total_dur_ms, 2),
            tokens_per_second=round(tps, 2),
        )


@dataclass
class Citation:
    """Represents a code reference cited within an LLM response."""

    raw_text: str
    file_path: str
    func_name: str
    is_grounded: bool
    source: str  # "vector_anchor" | "graph_neighbor" | "hallucinated"

    @property
    def identifier(self) -> str:
        return f"{self.file_path}::func::{self.func_name}"


@dataclass
class GroundingReport:
    """Detailed evaluation of how strictly the model adhered to retrieved context."""

    total_citations: int
    valid_citations: list[Citation] = field(default_factory=list)
    hallucinated_citations: list[Citation] = field(default_factory=list)
    adherence_score: float = 1.0  # 1.0 = 100% grounded
    is_fully_grounded: bool = True
    unmentioned_anchors: list[str] = field(default_factory=list)


@dataclass
class SynthesisResult:
    """The synthesized architectural explanation with metadata and grounding telemetry."""

    content: str
    query: str
    model: str
    token_usage: TokenUsage
    grounding_report: GroundingReport
    raw_response: dict[str, Any] = field(default_factory=dict)

    @property
    def citations(self) -> list[Citation]:
        """Return all citations found in content."""
        return self.grounding_report.valid_citations + self.grounding_report.hallucinated_citations

    @property
    def is_grounded(self) -> bool:
        """True if all citations were grounded in provided context."""
        return self.grounding_report.is_fully_grounded

    def to_dict(self) -> dict[str, Any]:
        """Export synthesis summary as a JSON-serializable dictionary."""
        return {
            "query": self.query,
            "model": self.model,
            "content": self.content,
            "token_usage": asdict(self.token_usage),
            "grounding": {
                "adherence_score": self.grounding_report.adherence_score,
                "is_fully_grounded": self.grounding_report.is_fully_grounded,
                "total_citations": self.grounding_report.total_citations,
                "valid_count": len(self.grounding_report.valid_citations),
                "hallucinated_count": len(self.grounding_report.hallucinated_citations),
                "valid_citations": [c.raw_text for c in self.grounding_report.valid_citations],
                "hallucinated_citations": [c.raw_text for c in self.grounding_report.hallucinated_citations],
            },
        }


# ---------------------------------------------------------------------------
# Citation Parsing & Grounding Evaluation
# ---------------------------------------------------------------------------

def extract_citations(text: str) -> list[tuple[str, str, str]]:
    """Extract all [file_path::func::<name>] citations from generated text.

    Returns:
        List of tuples: (raw_match, normalized_file_path, func_name).
    """
    matches = CITATION_REGEX.findall(text)
    results: list[tuple[str, str, str]] = []
    for file_path, func_name in matches:
        raw = f"[{file_path}::func::{func_name}]"
        norm_path = file_path.replace("\\", "/").strip()
        results.append((raw, norm_path, func_name.strip()))
    return results


def evaluate_grounding(
    text: str,
    included_anchors: Sequence[str],
    included_neighbors: Sequence[str],
) -> GroundingReport:
    """Evaluate whether citations in `text` strictly adhere to the retrieved context.

    Args:
        text: Synthesized explanation text from the LLM.
        included_anchors: Functions included as primary vector code snippets.
                          Format: ``"file_path::func::func_name"``.
        included_neighbors: Functions included as graph neighbors.
                          Format: ``"file_path::func::func_name"`` or ``"... (source)"``.

    Returns:
        GroundingReport detailing valid vs hallucinated citations and adherence score.
    """
    parsed = extract_citations(text)
    if not parsed:
        # No citations emitted; consider neutral/grounded if admission of ignorance
        return GroundingReport(total_citations=0, adherence_score=1.0, is_fully_grounded=True)

    # Normalize anchors and neighbors into lookup sets
    def normalize_key(item: str) -> tuple[str, str]:
        # Strip trailing parenthetical e.g. "(graph_callee)"
        clean = item.split(" (")[0].strip()
        if "::func::" in clean:
            parts = clean.split("::func::")
            return (parts[0].replace("\\", "/").strip(), parts[1].strip())
        return ("", clean)

    anchor_set = {normalize_key(a) for a in included_anchors}
    neighbor_set = {normalize_key(n) for n in included_neighbors}

    valid: list[Citation] = []
    hallucinated: list[Citation] = []

    for raw, path, func in parsed:
        key = (path, func)
        # Check exact path + func match or bare func match if path was partial
        matched_anchor = (key in anchor_set) or any(k[1] == func for k in anchor_set)
        matched_neighbor = (key in neighbor_set) or any(k[1] == func for k in neighbor_set)

        if matched_anchor:
            valid.append(
                Citation(
                    raw_text=raw,
                    file_path=path,
                    func_name=func,
                    is_grounded=True,
                    source="vector_anchor",
                )
            )
        elif matched_neighbor:
            valid.append(
                Citation(
                    raw_text=raw,
                    file_path=path,
                    func_name=func,
                    is_grounded=True,
                    source="graph_neighbor",
                )
            )
        else:
            hallucinated.append(
                Citation(
                    raw_text=raw,
                    file_path=path,
                    func_name=func,
                    is_grounded=False,
                    source="hallucinated",
                )
            )

    total = len(parsed)
    score = len(valid) / total if total > 0 else 1.0
    fully_grounded = len(hallucinated) == 0

    # Identify anchors that were not mentioned
    cited_funcs = {f for _, _, f in parsed}
    unmentioned = [a for a in included_anchors if normalize_key(a)[1] not in cited_funcs]

    return GroundingReport(
        total_citations=total,
        valid_citations=valid,
        hallucinated_citations=hallucinated,
        adherence_score=round(score, 4),
        is_fully_grounded=fully_grounded,
        unmentioned_anchors=unmentioned,
    )


# ---------------------------------------------------------------------------
# Ollama Client Orchestrator
# ---------------------------------------------------------------------------

class OllamaClient:
    """Robust, production-ready client for the local Ollama REST API.

    Features:
    - Connection pooling and keep-alive configuration.
    - Automatic exponential backoff retry on transient transport errors.
    - Synchronous and asynchronous chat and generate methods.
    - Real-time token streaming generators.
    """

    def __init__(
        self,
        config: Optional[OllamaConfig] = None,
        sync_client: Optional[httpx.Client] = None,
        async_client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        self.config = config or OllamaConfig()
        self._sync_client = sync_client
        self._async_client = async_client
        self._owns_sync_client = sync_client is None
        self._owns_async_client = async_client is None

    def _get_sync_client(self) -> httpx.Client:
        if self._sync_client is None or self._sync_client.is_closed:
            self._sync_client = httpx.Client(timeout=self.config.timeout)
            self._owns_sync_client = True
        return self._sync_client

    def _get_async_client(self) -> httpx.AsyncClient:
        if self._async_client is None or self._async_client.is_closed:
            self._async_client = httpx.AsyncClient(timeout=self.config.timeout)
            self._owns_async_client = True
        return self._async_client

    def close(self) -> None:
        """Close synchronous client connections."""
        if self._owns_sync_client and self._sync_client and not self._sync_client.is_closed:
            self._sync_client.close()

    async def aclose(self) -> None:
        """Close asynchronous client connections."""
        if self._owns_async_client and self._async_client and not self._async_client.is_closed:
            await self._async_client.aclose()

    def __enter__(self) -> OllamaClient:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    async def __aenter__(self) -> OllamaClient:
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.aclose()

    # -----------------------------------------------------------------------
    # Health & Discovery
    # -----------------------------------------------------------------------

    def health_check(self, timeout: float = 3.0) -> bool:
        """Check if Ollama service is reachable and responsive."""
        try:
            client = self._get_sync_client()
            resp = client.get(self.config.tags_endpoint, timeout=timeout)
            return resp.status_code == 200
        except Exception:
            return False

    async def health_check_async(self, timeout: float = 3.0) -> bool:
        """Asynchronously check if Ollama service is reachable."""
        try:
            client = self._get_async_client()
            resp = await client.get(self.config.tags_endpoint, timeout=timeout)
            return resp.status_code == 200
        except Exception:
            return False

    def list_models(self) -> list[str]:
        """Fetch list of model names currently available in Ollama."""
        try:
            client = self._get_sync_client()
            resp = client.get(self.config.tags_endpoint)
            if resp.status_code != 200:
                raise OllamaResponseError(f"Failed to list models: HTTP {resp.status_code}")
            data = resp.json()
            return [m.get("name", "") for m in data.get("models", [])]
        except httpx.ConnectError as err:
            raise OllamaConnectionError(
                f"Cannot connect to Ollama at {self.config.base_url}. Ensure Ollama is running (`ollama serve`)."
            ) from err
        except Exception as err:
            if isinstance(err, LLMEngineError):
                raise
            raise OllamaResponseError(f"Error querying Ollama models: {err}") from err

    def is_model_available(self, model_name: Optional[str] = None) -> bool:
        """Check if specified model (or configured model) is installed locally."""
        target = (model_name or self.config.model).strip().lower()
        try:
            available = self.list_models()
            for m in available:
                m_clean = m.strip().lower()
                if m_clean == target or m_clean.startswith(f"{target}:"):
                    return True
            return False
        except Exception:
            return False

    def resolve_model(self, requested_model: Optional[str] = None) -> str:
        """Resolve target model name, falling back to installed local models if needed."""
        target = (requested_model or self.config.model).strip()
        try:
            available = self.list_models()
            if not available:
                return target
            for m in available:
                if m.lower() == target.lower() or m.lower().startswith(f"{target.lower()}:"):
                    return m
            # If target not found, look for popular installed code or general models
            for pref in ["qwen2.5-coder", "codellama", "llama3", "mistral", "deepseek-coder"]:
                for m in available:
                    if pref in m.lower():
                        logger.info("Target model '%s' not present. Auto-selected available model: %s", target, m)
                        return m
            return available[0]
        except Exception:
            return target

    # -----------------------------------------------------------------------
    # Synchronous Chat & Stream
    # -----------------------------------------------------------------------

    def chat(
        self,
        messages: list[dict[str, str]],
        model: Optional[str] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Execute a non-streaming chat request against /api/chat with retries.

        Args:
            messages: Role-message dicts: [{"role": "system"|"user"|"assistant", "content": "..."}]
            model: Model name override (defaults to config.model).
            options: Hyperparameter options dict override.

        Returns:
            Ollama response payload dictionary.
        """
        target_model = model or self.config.model
        req_options = options or self.config.to_options_dict()

        payload = {
            "model": target_model,
            "messages": messages,
            "stream": False,
            "options": req_options,
            "keep_alive": self.config.keep_alive,
        }

        retries = 0
        last_err: Optional[Exception] = None
        client = self._get_sync_client()

        while retries <= self.config.max_retries:
            try:
                t0 = time.time()
                resp = client.post(
                    self.config.chat_endpoint,
                    json=payload,
                    timeout=self.config.timeout,
                )
                wall_time_ms = (time.time() - t0) * 1000.0

                if resp.status_code == 404:
                    raise OllamaModelNotFoundError(
                        f"Model '{target_model}' not found in Ollama. Pull it via: `ollama pull {target_model}`"
                    )
                if resp.status_code != 200:
                    raise OllamaResponseError(
                        f"Ollama returned HTTP {resp.status_code}: {resp.text}"
                    )

                data = resp.json()
                data["_wall_time_ms"] = wall_time_ms
                return data

            except httpx.ConnectError as err:
                last_err = OllamaConnectionError(
                    f"Failed to connect to Ollama at {self.config.base_url}. Ensure daemon is running."
                )
            except httpx.TimeoutException as err:
                last_err = OllamaTimeoutError(
                    f"Inference request timed out after {self.config.timeout}s."
                )
            except (OllamaModelNotFoundError, OllamaResponseError):
                raise
            except Exception as err:
                last_err = OllamaResponseError(f"Unexpected error communicating with Ollama: {err}")

            retries += 1
            if retries <= self.config.max_retries:
                sleep_sec = (self.config.backoff_factor * (2 ** (retries - 1))) + random.uniform(0, 0.1)
                logger.warning(
                    "Ollama request failed (%s). Retrying %d/%d in %.2fs...",
                    last_err,
                    retries,
                    self.config.max_retries,
                    sleep_sec,
                )
                time.sleep(sleep_sec)

        if last_err:
            raise last_err
        raise OllamaResponseError("Chat request failed after max retries.")

    def chat_stream(
        self,
        messages: list[dict[str, str]],
        model: Optional[str] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> Iterator[dict[str, Any]]:
        """Stream chat tokens chunk-by-chunk from /api/chat.

        Yields:
            Dict chunks as parsed JSON from Ollama stream.
        """
        target_model = model or self.config.model
        req_options = options or self.config.to_options_dict()

        payload = {
            "model": target_model,
            "messages": messages,
            "stream": True,
            "options": req_options,
            "keep_alive": self.config.keep_alive,
        }

        client = self._get_sync_client()
        try:
            with client.stream(
                "POST",
                self.config.chat_endpoint,
                json=payload,
                timeout=self.config.timeout,
            ) as response:
                if response.status_code == 404:
                    raise OllamaModelNotFoundError(
                        f"Model '{target_model}' not found in Ollama. Pull it via: `ollama pull {target_model}`"
                    )
                if response.status_code != 200:
                    raise OllamaResponseError(f"Ollama stream returned HTTP {response.status_code}")

                for line in response.iter_lines():
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                        yield chunk
                    except json.JSONDecodeError:
                        continue

        except httpx.ConnectError as err:
            raise OllamaConnectionError(f"Cannot connect to Ollama at {self.config.base_url}") from err
        except httpx.TimeoutException as err:
            raise OllamaTimeoutError(f"Streaming inference timed out after {self.config.timeout}s") from err

    # -----------------------------------------------------------------------
    # Asynchronous Chat & Stream
    # -----------------------------------------------------------------------

    async def chat_async(
        self,
        messages: list[dict[str, str]],
        model: Optional[str] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Asynchronously execute non-streaming chat request with retries."""
        target_model = model or self.config.model
        req_options = options or self.config.to_options_dict()

        payload = {
            "model": target_model,
            "messages": messages,
            "stream": False,
            "options": req_options,
            "keep_alive": self.config.keep_alive,
        }

        client = self._get_async_client()
        retries = 0
        last_err: Optional[Exception] = None

        while retries <= self.config.max_retries:
            try:
                t0 = time.time()
                resp = await client.post(
                    self.config.chat_endpoint,
                    json=payload,
                    timeout=self.config.timeout,
                )
                wall_time_ms = (time.time() - t0) * 1000.0

                if resp.status_code == 404:
                    raise OllamaModelNotFoundError(
                        f"Model '{target_model}' not found in Ollama. Pull via: `ollama pull {target_model}`"
                    )
                if resp.status_code != 200:
                    raise OllamaResponseError(f"Ollama returned HTTP {resp.status_code}: {resp.text}")

                data = resp.json()
                data["_wall_time_ms"] = wall_time_ms
                return data

            except httpx.ConnectError as err:
                last_err = OllamaConnectionError(f"Failed to connect to Ollama at {self.config.base_url}")
            except httpx.TimeoutException as err:
                last_err = OllamaTimeoutError(f"Inference timed out after {self.config.timeout}s")
            except (OllamaModelNotFoundError, OllamaResponseError):
                raise
            except Exception as err:
                last_err = OllamaResponseError(f"Async error with Ollama: {err}")

            retries += 1
            if retries <= self.config.max_retries:
                sleep_sec = (self.config.backoff_factor * (2 ** (retries - 1))) + random.uniform(0, 0.1)
                await asyncio.sleep(sleep_sec)

        if last_err:
            raise last_err
        raise OllamaResponseError("Async chat request failed after max retries.")

    async def chat_stream_async(
        self,
        messages: list[dict[str, str]],
        model: Optional[str] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Asynchronously stream chat tokens chunk-by-chunk."""
        target_model = model or self.config.model
        req_options = options or self.config.to_options_dict()

        payload = {
            "model": target_model,
            "messages": messages,
            "stream": True,
            "options": req_options,
            "keep_alive": self.config.keep_alive,
        }

        client = self._get_async_client()
        try:
            async with client.stream(
                "POST",
                self.config.chat_endpoint,
                json=payload,
                timeout=self.config.timeout,
            ) as response:
                if response.status_code == 404:
                    raise OllamaModelNotFoundError(f"Model '{target_model}' not found in Ollama.")
                if response.status_code != 200:
                    raise OllamaResponseError(f"Ollama stream HTTP {response.status_code}")

                async for line in response.aiter_lines():
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                        yield chunk
                    except json.JSONDecodeError:
                        continue
        except httpx.ConnectError as err:
            raise OllamaConnectionError(f"Cannot connect to Ollama at {self.config.base_url}") from err
        except httpx.TimeoutException as err:
            raise OllamaTimeoutError(f"Streaming inference timed out after {self.config.timeout}s") from err


# ---------------------------------------------------------------------------
# High-Level Architecture Explanation Synthesis Engine
# ---------------------------------------------------------------------------

class LLMEngine:
    """High-level facade for synthesizing code architecture explanations.

    Orchestrates:
    - Structured prompt consumption
    - Ollama model inference
    - Token usage tracking
    - Real-time streaming
    - Grounding adherence evaluation
    """

    def __init__(
        self,
        config: Optional[OllamaConfig] = None,
        client: Optional[OllamaClient] = None,
    ) -> None:
        self.config = config or OllamaConfig()
        self.client = client or OllamaClient(config=self.config)

    def synthesize(
        self,
        prompt: Union[StructuredPrompt, str, list[dict[str, str]]],
        query: str = "",
        model: Optional[str] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> SynthesisResult:
        """Synthesize an architectural explanation from a prompt.

        Args:
            prompt: Either a StructuredPrompt instance (from rag.prompt_builder),
                    a raw full prompt string, or a pre-formatted messages list.
            query: User developer inquiry (extracted automatically if StructuredPrompt).
            model: Optional model override.
            options: Optional inference options.

        Returns:
            SynthesisResult containing narrative, citations, metrics, and adherence.
        """
        messages, extracted_query, anchors, neighbors = self._normalize_prompt_input(prompt, query)
        target_model = self.client.resolve_model(model or self.config.model)

        t0 = time.time()
        response_data = self.client.chat(messages=messages, model=target_model, options=options)
        wall_time_ms = (time.time() - t0) * 1000.0

        content = response_data.get("message", {}).get("content", "")
        token_usage = TokenUsage.from_ollama_response(response_data, fallback_wall_time_ms=wall_time_ms)
        grounding_report = evaluate_grounding(content, anchors, neighbors)

        return SynthesisResult(
            content=content,
            query=extracted_query,
            model=target_model,
            token_usage=token_usage,
            grounding_report=grounding_report,
            raw_response=response_data,
        )

    def stream_synthesize(
        self,
        prompt: Union[StructuredPrompt, str, list[dict[str, str]]],
        query: str = "",
        model: Optional[str] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> Iterator[Tuple[str, Optional[SynthesisResult]]]:
        """Stream explanation token chunks, yielding the final SynthesisResult on completion.

        Yields:
            Tuples of (token_chunk: str, final_result: Optional[SynthesisResult]).
            For intermediate chunks, final_result is None.
            The final chunk yields ("", SynthesisResult).
        """
        messages, extracted_query, anchors, neighbors = self._normalize_prompt_input(prompt, query)
        target_model = self.client.resolve_model(model or self.config.model)

        collected_content: list[str] = []
        final_meta: dict[str, Any] = {}
        t0 = time.time()

        for chunk in self.client.chat_stream(messages=messages, model=target_model, options=options):
            msg_chunk = chunk.get("message", {}).get("content", "")
            if msg_chunk:
                collected_content.append(msg_chunk)
                yield (msg_chunk, None)

            if chunk.get("done", False):
                final_meta = chunk

        wall_time_ms = (time.time() - t0) * 1000.0
        full_text = "".join(collected_content)
        token_usage = TokenUsage.from_ollama_response(final_meta, fallback_wall_time_ms=wall_time_ms)
        grounding_report = evaluate_grounding(full_text, anchors, neighbors)

        final_result = SynthesisResult(
            content=full_text,
            query=extracted_query,
            model=target_model,
            token_usage=token_usage,
            grounding_report=grounding_report,
            raw_response=final_meta,
        )

        yield ("", final_result)

    async def stream_synthesize_async(
        self,
        prompt: Union[StructuredPrompt, str, list[dict[str, str]]],
        query: str = "",
        model: Optional[str] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> AsyncIterator[Tuple[str, Optional[SynthesisResult]]]:
        """Asynchronously stream tokens and yield final SynthesisResult."""
        messages, extracted_query, anchors, neighbors = self._normalize_prompt_input(prompt, query)
        target_model = self.client.resolve_model(model or self.config.model)

        collected_content: list[str] = []
        final_meta: dict[str, Any] = {}
        t0 = time.time()

        async for chunk in self.client.chat_stream_async(messages=messages, model=target_model, options=options):
            msg_chunk = chunk.get("message", {}).get("content", "")
            if msg_chunk:
                collected_content.append(msg_chunk)
                yield (msg_chunk, None)

            if chunk.get("done", False):
                final_meta = chunk

        wall_time_ms = (time.time() - t0) * 1000.0
        full_text = "".join(collected_content)
        token_usage = TokenUsage.from_ollama_response(final_meta, fallback_wall_time_ms=wall_time_ms)
        grounding_report = evaluate_grounding(full_text, anchors, neighbors)

        final_result = SynthesisResult(
            content=full_text,
            query=extracted_query,
            model=target_model,
            token_usage=token_usage,
            grounding_report=grounding_report,
            raw_response=final_meta,
        )

        yield ("", final_result)

    def _normalize_prompt_input(
        self,
        prompt: Union[StructuredPrompt, str, list[dict[str, str]]],
        query: str,
    ) -> Tuple[list[dict[str, str]], str, list[str], list[str]]:
        """Normalize various input prompt formats into uniform chat messages and context lists."""
        anchors: list[str] = []
        neighbors: list[str] = []
        final_query = query

        # Case 1: StructuredPrompt object
        if hasattr(prompt, "to_messages") and callable(getattr(prompt, "to_messages")):
            messages = prompt.to_messages()
            anchors = getattr(prompt, "included_anchors", [])
            neighbors = getattr(prompt, "included_neighbors", [])
            # Extract query from user prompt if not passed explicitly
            if not final_query and hasattr(prompt, "user_prompt"):
                m = re.search(r"<user_query>\s*(.*?)\s*</user_query>", prompt.user_prompt, re.DOTALL)
                if m:
                    final_query = m.group(1).strip()
            return messages, final_query, anchors, neighbors

        # Case 2: Pre-formatted list of role-message dicts
        if isinstance(prompt, list):
            messages = prompt
            return messages, final_query, anchors, neighbors

        # Case 3: Raw string prompt
        raw_str = str(prompt)
        messages = [{"role": "user", "content": raw_str}]
        return messages, final_query, anchors, neighbors


# ---------------------------------------------------------------------------
# Convenience Facade Functions
# ---------------------------------------------------------------------------

def synthesize_architecture_explanation(
    prompt: Union[StructuredPrompt, str, list[dict[str, str]]],
    query: str = "",
    config: Optional[OllamaConfig] = None,
    model: Optional[str] = None,
) -> SynthesisResult:
    """Convenience function to synthesize an architectural explanation using default engine."""
    engine = LLMEngine(config=config)
    return engine.synthesize(prompt=prompt, query=query, model=model)


def stream_architecture_explanation(
    prompt: Union[StructuredPrompt, str, list[dict[str, str]]],
    query: str = "",
    config: Optional[OllamaConfig] = None,
    model: Optional[str] = None,
) -> Iterator[Tuple[str, Optional[SynthesisResult]]]:
    """Convenience generator to stream explanation tokens in real-time."""
    engine = LLMEngine(config=config)
    return engine.stream_synthesize(prompt=prompt, query=query, model=model)


# ---------------------------------------------------------------------------
# Synthetic / Mock Generation for Offline Verification
# ---------------------------------------------------------------------------

def generate_mock_explanation(
    query: str,
    anchors: list[str],
    neighbors: list[str],
    model: str = "mock-codellama",
) -> SynthesisResult:
    """Generate a realistic, synthetic architecture explanation for offline testing and verification.

    Emits grounded citations obeying the [file_path::func::<name>] format.
    """
    anchor_str = ", ".join(anchors) if anchors else "the relevant functions"
    mock_narrative = (
        f"### Code Architecture Explanation\n\n"
        f"In response to your query regarding: **\"{query}\"**\n\n"
        f"#### Core Execution Flow\n"
    )

    valid_cits: list[str] = []
    for a in anchors:
        clean = a.split(" (")[0]
        if "::func::" in clean:
            path, func = clean.split("::func::")
            valid_cits.append(f"[{path}::func::{func}]")
            mock_narrative += (
                f"1. **Anchor Analysis**: The primary operation is handled by `{func}` in {valid_cits[-1]}. "
                f"It manages resources with defensive error trapping and boundary validation.\n"
            )

    if neighbors:
        mock_narrative += "\n#### Structural Dependencies & Collaborators\n"
        for n in neighbors:
            clean = n.split(" (")[0]
            if "::func::" in clean:
                path, func = clean.split("::func::")
                cit = f"[{path}::func::{func}]"
                valid_cits.append(cit)
                mock_narrative += (
                    f"- The component collaborates with `{func}` ({cit}) to fulfill downstream "
                    f"cleanup or upstream orchestration.\n"
                )

    mock_narrative += (
        f"\n#### Architectural Summary\n"
        f"All state transitions occur within bounded transactional guarantees, ensuring that "
        f"exceptions trigger immediate rollbacks before propagating.\n"
    )

    t_usage = TokenUsage(
        prompt_tokens=420,
        completion_tokens=185,
        total_tokens=605,
        eval_duration_ms=450.0,
        prompt_eval_duration_ms=80.0,
        total_duration_ms=530.0,
        tokens_per_second=411.1,
    )

    grounding_rep = evaluate_grounding(mock_narrative, anchors, neighbors)

    return SynthesisResult(
        content=mock_narrative,
        query=query,
        model=model,
        token_usage=t_usage,
        grounding_report=grounding_rep,
        raw_response={"mock": True, "done": True},
    )


# ---------------------------------------------------------------------------
# CLI / Interactive Runner
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI runner to test LLM engine connection, stream inference, or run mock demo."""
    parser = argparse.ArgumentParser(
        description="Day 27: LM Service Integration & Ollama Orchestrator"
    )
    parser.add_argument(
        "--query",
        type=str,
        default="How does clone_repository clean up when an error occurs?",
        help="Developer question to synthesize explanation for",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_OLLAMA_MODEL,
        help="Ollama model name (e.g. codellama, llama3, qwen2.5-coder)",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=DEFAULT_OLLAMA_BASE_URL,
        help="Ollama daemon URL (default: http://localhost:11434)",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream tokens in real-time to console",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check Ollama connectivity and list local models",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run synthesis in mock mode without requiring a live Ollama daemon",
    )

    args = parser.parse_args()

    config = OllamaConfig(base_url=args.base_url, model=args.model)
    client = OllamaClient(config=config)

    print("=" * 80)
    print("CODEBASE VISUALIZER — LM SERVICE ORCHESTRATION ENGINE (DAY 27)")
    print("=" * 80)
    print(f"Ollama Target URL : {config.base_url}")
    print(f"Target Model      : {config.model}")
    print("=" * 80)

    if args.check:
        print("\nChecking Ollama daemon status...")
        healthy = client.health_check()
        if healthy:
            print("  Status: ONLINE (Ollama is active)")
            try:
                models = client.list_models()
                print(f"  Available Models ({len(models)}):")
                for m in models:
                    print(f"    • {m}")
            except Exception as e:
                print(f"  Failed to retrieve models: {e}")
        else:
            print("  Status: OFFLINE")
            print("  To start Ollama locally, run:")
            print("    ollama serve")
            print(f"  To pull the target model, run:\n    ollama pull {config.model}")
        return

    # Build a synthetic Graph RAG prompt using prompt_builder
    from rag.prompt_builder import build_rag_prompt

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

    structured_prompt = build_rag_prompt(
        query=args.query,
        hybrid_results=synthetic_results,
        code_cache=mock_code,
        max_tokens=2000,
    )

    print(f"\nUser Query: {args.query}")
    print(f"Context Anchors   : {structured_prompt.included_anchors}")
    print(f"Context Neighbors : {structured_prompt.included_neighbors}")
    print(f"Prompt Tokens     : ~{structured_prompt.estimated_tokens}\n")

    engine = LLMEngine(config=config, client=client)

    # Check if Ollama is available or mock requested
    is_live = not args.mock and client.health_check(timeout=1.5)

    if not is_live:
        if not args.mock:
            print("[INFO] Local Ollama daemon not reachable at specified URL.")
            print("[INFO] Falling back to high-fidelity Mock Mode for demonstration.\n")
        result = generate_mock_explanation(
            query=args.query,
            anchors=structured_prompt.included_anchors,
            neighbors=structured_prompt.included_neighbors,
            model=f"{config.model} (mock)",
        )
    else:
        resolved_model = client.resolve_model(args.model)
        print(f"[INFO] Connected to Ollama! Synthesizing explanation via {resolved_model}...\n")
        try:
            if args.stream:
                print("--- Streaming Output ---")
                final_res: Optional[SynthesisResult] = None
                for token, res in engine.stream_synthesize(structured_prompt, model=resolved_model):
                    if token:
                        sys.stdout.write(token)
                        sys.stdout.flush()
                    if res is not None:
                        final_res = res
                print("\n------------------------")
                result = final_res  # type: ignore
            else:
                result = engine.synthesize(structured_prompt, model=resolved_model)
        except Exception as e:
            print(f"[ERROR] Inference failed: {e}")
            print("[INFO] Generating synthetic fallback explanation...")
            result = generate_mock_explanation(
                query=args.query,
                anchors=structured_prompt.included_anchors,
                neighbors=structured_prompt.included_neighbors,
            )

    print("\n" + "=" * 80)
    print("SYNTHESIS RESULT")
    print("=" * 80)
    print(result.content)
    print("=" * 80)
    print("ORCHESTRATION & GROUNDING TELEMETRY")
    print("=" * 80)
    print(f"Model Used        : {result.model}")
    print(f"Prompt Tokens     : {result.token_usage.prompt_tokens}")
    print(f"Completion Tokens : {result.token_usage.completion_tokens}")
    print(f"Tokens / Second   : {result.token_usage.tokens_per_second:.1f}")
    print(f"Latency           : {result.token_usage.total_duration_ms:.1f} ms")
    print(f"Total Citations   : {result.grounding_report.total_citations}")
    print(f"Valid Citations   : {len(result.grounding_report.valid_citations)}")
    print(f"Hallucinated      : {len(result.grounding_report.hallucinated_citations)}")
    print(f"Adherence Score   : {result.grounding_report.adherence_score * 100:.1f}%")
    print(f"Fully Grounded    : {'YES' if result.is_grounded else 'NO'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
