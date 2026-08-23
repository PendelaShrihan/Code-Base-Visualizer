from fastapi import FastAPI
from pydantic import BaseModel
import redis.asyncio as redis

from app.routers import git as git_router
from app.routers import graph as graph_router
from app.routers import status as status_router

app = FastAPI(title="CodeBase Visualizer")

# Register routers
app.include_router(git_router.router)
app.include_router(graph_router.router)
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