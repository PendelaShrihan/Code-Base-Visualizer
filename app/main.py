from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import redis.asyncio as redis

from app.routers import analyze as analyze_router
from app.routers import git as git_router
from app.routers import graph as graph_router
try:
    from app.routers import query as query_router
except ImportError:
    query_router = None
from app.routers import status as status_router

app = FastAPI(title="CodeBase Visualizer")

# ---------------------------------------------------------------------------
# CORS — allow the Vite dev server (and any local origin) to call the API
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(analyze_router.router)
app.include_router(git_router.router)
app.include_router(graph_router.router)
if query_router is not None:
    app.include_router(query_router.router)
app.include_router(status_router.router)

redis_client = redis.Redis(host="redis", port=6379, decode_responses=True)


class HealthResponse(BaseModel):
    status: str
    service: str
    request_count: int


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    count = await redis_client.incr("health_check_count")
    return HealthResponse(
        status="ok",
        service="codebase-visualizer",
        request_count=count,
    )