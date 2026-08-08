from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="CodeBase Visualizer")


class HealthResponse(BaseModel):
    status: str
    service: str


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    return HealthResponse(status="ok", service="codebase-visualizer")
