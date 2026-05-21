from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.models import ResearchRequest, ResearchResult
from app.pipeline import run_research_pipeline

app = FastAPI(title="Academic Research Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/research/run", response_model=ResearchResult)
async def run_research(request: ResearchRequest) -> ResearchResult:
    return await run_research_pipeline(request)
