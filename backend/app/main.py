from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.experiment import run_single_topic_experiment
from app.models import ExperimentRequest, ExperimentResult, ResearchRequest, ResearchResult
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


@app.post("/api/experiment/run", response_model=ExperimentResult)
async def run_experiment(request: ExperimentRequest) -> ExperimentResult:
    return await run_single_topic_experiment(request)
