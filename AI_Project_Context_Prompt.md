# AI Development Context Prompt
# 用于 Codex / Claude / Cursor / GPT 的项目上下文 Prompt

You are assisting in the development of an LLM-based Academic Research Agent.

Project goal:
Build a multi-stage academic research assistant system that can:
1. Retrieve related literature
2. Summarize papers
3. Identify research gaps
4. Generate research questions
5. Generate hypotheses
6. Suggest methodologies
7. Generate proposal drafts
8. Perform self-evaluation

Tech stack:
- Frontend: React + Vite
- Backend: FastAPI or Flask
- LLM API: OpenAI API
- Retrieval: arXiv API
- Storage: Local JSON
- Evaluation: Manual rubric + GPT evaluation

Important constraints:
- DO NOT train custom models
- DO NOT use fine-tuning
- DO NOT introduce overly complex multi-agent systems
- Focus on prompt engineering and workflow design
- Keep implementation lightweight and stable

Final pipeline:

User Topic
→ arXiv Retrieval
→ Literature Summary
→ Research Gap
→ Research Question
→ Hypothesis
→ Methodology
→ Proposal Draft
→ Self-Evaluation

Expected output format:
Use structured JSON outputs whenever possible.

Example:
{
  "summary": "...",
  "gap": "...",
  "question": "...",
  "hypothesis": "...",
  "methodology": "...",
  "proposal": "...",
  "evaluation": {
    "clarity": 4,
    "novelty": 3,
    "feasibility": 5
  }
}

Frontend expectations:
- Simple and clean UI
- Input box
- Generate button
- Multi-stage result cards
- Loading states
- Error handling

Backend expectations:
- Modular pipeline
- Prompt chaining
- Retry mechanism
- Token control
- API abstraction

Prompt engineering expectations:
Each stage should have an independent prompt template:
- literature_summary.txt
- research_gap.txt
- hypothesis.txt
- methodology.txt
- proposal.txt

Evaluation expectations:
Compare:
1. Single-prompt baseline
2. Multi-stage pipeline

Evaluation dimensions:
- Clarity
- Logic
- Novelty
- Feasibility
- Literature alignment

Important:
Prioritize:
1. Stability
2. Readability
3. Modularity
4. Structured outputs

Avoid unnecessary complexity.

中文说明：
这是一个基于现有 LLM API 的 Academic Research Agent 项目。
项目重点是：
- Prompt Engineering
- Multi-stage Reasoning
- Literature-Augmented Generation
- Evaluation Framework

不要尝试：
- 自训练模型
- 微调
- 复杂 autonomous agents
- 复杂数据库系统

目标是快速完成一个稳定、可演示、结构清晰的系统。
