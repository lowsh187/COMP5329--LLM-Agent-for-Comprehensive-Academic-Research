# Member B Frontend Guide

This repository is currently focused on Member B responsibilities only.

Member B is responsible for:

- Frontend UI
- Frontend-backend integration
- Result visualization
- Testing
- Evaluation support
- Bug fixing

Backend, prompt chaining, OpenAI API calls, arXiv retrieval, and local JSON storage are Member A responsibilities.

## Frontend Scope

The UI follows the project prompt markdown:

- Simple and clean UI
- Research topic input box
- Generate button
- Multi-stage result cards
- Loading state
- Error handling
- Retrieved paper display
- Proposal display
- Self-evaluation score visualization
- Baseline vs multi-stage comparison support

## Run

```bash
cd frontend
npm install
npm run dev
```

Open:

```text
http://127.0.0.1:5173/
```

## Backend Integration

By default the frontend calls:

```text
http://127.0.0.1:8000/api/research/run
```

You can override it with:

```bash
VITE_API_BASE_URL=http://127.0.0.1:8000 npm run dev
```

If the backend is not running, the UI can switch to demo data mode. This lets Member B continue UI testing before Member A finishes backend integration.

## Expected Backend Response

The frontend expects a structured JSON response:

```json
{
  "topic": "LLM-based feedback generation for programming education",
  "papers": [
    {
      "title": "Paper title",
      "authors": ["Author A", "Author B"],
      "abstract": "Paper abstract",
      "published": "2025-01-01",
      "url": "https://arxiv.org/abs/example"
    }
  ],
  "summary": "...",
  "gap": "...",
  "question": "...",
  "hypothesis": "...",
  "methodology": "...",
  "proposal": "...",
  "evaluation": {
    "clarity": 4,
    "logic": 4,
    "novelty": 3,
    "feasibility": 5,
    "literature_alignment": 4,
    "comments": "Brief evaluation comment"
  }
}
```

## Build Check

```bash
cd frontend
npm run build
```

