from fastapi import FastAPI
from pydantic import BaseModel
import redis.asyncio as redis

app = FastAPI(title="CodeBase Visualizer")

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
