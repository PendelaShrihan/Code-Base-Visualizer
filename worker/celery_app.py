"""
worker/celery_app.py
--------------------
Celery application entry-point for the CodeBase Visualizer project.

Architecture
------------
  ┌──────────────┐   task message   ┌─────────┐   result   ┌────────────────┐
  │  FastAPI app │ ───────────────► │  Redis  │ ◄───────── │  Celery worker │
  │  (producer)  │                  │ broker  │ ──────────► │  (consumer)    │
  └──────────────┘                  └─────────┘            └────────────────┘
                                         │
                                   result backend
                                   (same Redis DB)

Configuration decisions
-----------------------
  • broker_url       – Redis DB 0 (redis://localhost:6379/0)
    result_backend  – Redis DB 1 (redis://localhost:6379/1)
    Keeping them on separate logical databases means KEYS * in Redis CLI
    only shows the namespace you care about. Override at runtime with
    CELERY_BROKER_URL / CELERY_RESULT_BACKEND env vars (Docker / prod).

  • task_serializer / result_serializer / accept_content – JSON only.
    JSON is human-readable, language-agnostic and avoids the security
    issues of pickle (Celery's legacy default).

  • timezone / enable_utc – All timestamps stored in UTC to avoid
    daylight-saving ambiguity across distributed workers.

  • task_track_started – Lets callers poll STARTED state via AsyncResult
    before the task completes (handy for long-running analysis jobs).
"""

import os

from celery import Celery

# ---------------------------------------------------------------------------
# Connection URLs
# ---------------------------------------------------------------------------
# Docker Compose exposes Redis under the service name "redis".
# Fall back to localhost for local development without Docker.

# Broker: DB 0 — holds queued task messages (lists, sorted sets).
_BROKER_URL: str = os.getenv(
    "CELERY_BROKER_URL",
    "redis://localhost:6379/0",
)

# Backend: DB 1 — holds task results / state (string keys per task-id).
# A separate logical DB keeps result keys out of the broker namespace,
# making `KEYS *` output in Redis CLI cleaner when debugging.
_RESULT_BACKEND: str = os.getenv(
    "CELERY_RESULT_BACKEND",
    "redis://localhost:6379/1",
)

# ---------------------------------------------------------------------------
# Task 3 - Celery instance
# The first argument ("worker") is the name of the worker module / package.
# This name is embedded in every task message so workers can route correctly.
# ---------------------------------------------------------------------------
celery_app = Celery(
    "worker",          # module / package name (matches this directory)
    broker=_BROKER_URL,
    backend=_RESULT_BACKEND,
)

# ---------------------------------------------------------------------------
# Task 4 - Redis broker & backend already wired above via constructor args.
# The update() call below applies additional tuning in one place.
# ---------------------------------------------------------------------------
celery_app.conf.update(
    # Task 5: Serialization
    # Use JSON for all task messages and stored results.
    # "accept_content" whitelists the formats a worker will deserialise;
    # setting it to JSON-only prevents arbitrary pickle execution.
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],

    # Timezone
    timezone="UTC",
    enable_utc=True,

    # Worker behaviour
    # Report STARTED state so callers can distinguish "queued" vs "running".
    task_track_started=True,

    # Acknowledge the message only after the task succeeds/fails, not before.
    # This prevents message loss if a worker crashes mid-execution.
    task_acks_late=True,

    # Prefetch only 1 task at a time per worker process.
    # Keeps long-running tasks from starving other workers.
    worker_prefetch_multiplier=1,
)

# ---------------------------------------------------------------------------
# Auto-discover tasks
# ---------------------------------------------------------------------------
# Celery will look for a tasks.py (or tasks/ package) inside the "worker"
# package. Add more packages to the list as the project grows.
celery_app.autodiscover_tasks(["worker"])
