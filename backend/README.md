# Backend Guide

FastAPI backend for Member A responsibilities:

- arXiv literature retrieval
- Semantic Scholar and Crossref literature retrieval
- multi-stage prompt chaining
- LLM API abstraction for OpenAI-compatible providers
- structured JSON responses
- self-evaluation
- single-prompt baseline support

## Run

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Set `LLM_API_KEY`, `LLM_BASE_URL`, and `LLM_MODEL` in `.env` for real LLM output. Without an API key, the backend still returns deterministic fallback output so the frontend can be tested.

Example `.env` for OpenAI-compatible APIs:

```env
LLM_API_KEY=your-api-key
LLM_BASE_URL=https://api.example.com/v1
LLM_MODEL=your-model-name
SEMANTIC_SCHOLAR_API_KEY=
ARXIV_MAX_RESULTS=5
LLM_TEMPERATURE=0.3
```

For providers that are OpenAI-compatible, only these three values usually need to change. The old `OPENAI_API_KEY` and `OPENAI_MODEL` names are still accepted for backward compatibility.

The literature search distributes the requested paper count across arXiv, Semantic Scholar, and Crossref, then removes duplicate papers by DOI URL or normalized title. `SEMANTIC_SCHOLAR_API_KEY` is optional for local testing.

## Local Data

The backend writes local JSON files for experiment tracking:

- `backend/data/cache/literature/`: cached retrieval results by topic and requested paper count. This avoids repeated external API calls during demos. The cache directory is ignored by Git.
- `backend/data/runs/`: full pipeline outputs saved after each successful request. These files can be used for evaluation tables and baseline vs multi-stage analysis.

If you need fresh retrieval results for a topic, delete the matching file under `backend/data/cache/literature/` and run the request again.

## API

```text
POST /api/research/run
```

Request:

```json
{
  "topic": "LLM-based feedback generation for programming education",
  "max_papers": 5
}
```

Response matches the frontend contract in `README_B_FRONTEND.md`.
