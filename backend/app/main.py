from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.agent_rl import load_policy
from app.agent_training import run_agent_training
from app.experiment import run_single_topic_experiment
from app.models import (
    AgentPolicyState,
    AgentTrainingRequest,
    AgentTrainingResult,
    ExperimentRequest,
    ExperimentResult,
    ResearchRequest,
    ResearchResult,
)
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


@app.get("/api/agent/policy", response_model=AgentPolicyState)
async def get_agent_policy() -> AgentPolicyState:
    return load_policy()


@app.post("/api/agent/train", response_model=AgentTrainingResult)
async def train_agent(request: AgentTrainingRequest) -> AgentTrainingResult:
    return await run_agent_training(request)
