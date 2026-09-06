# Academic Research Agent · 学术研究智能体

**From a research topic to an evidence-informed, testable research proposal.**  
**从研究主题出发，形成有文献依据、可检验的研究方案。**

[中文介绍](#中文介绍) · [English](#english) · [快速开始 / Quick start](#quick-start) · [API](#api)

> 本文依据 `agent-rl-learning` 分支的当前实现编写。项目是已具备前后端、研究生成流程和策略学习接口的研究原型；历史实验结果与功能实现状态分别说明。  
> This README describes the current implementation on `agent-rl-learning`: a research prototype with a frontend, a research-generation backend, and strategy-learning APIs. Implementation status and historical experimental evidence are reported separately.

## 中文介绍

### 项目定位与使用范围

Academic Research Agent 面向科研选题与研究方案设计。用户输入一个研究主题后，系统检索相关文献，整理证据与研究分歧，逐步生成研究缺口、问题、假设、方法和 proposal，并通过结构化评审与反馈辅助改进。

适用场景包括：

- **研究生与课程项目选题**：将宽泛兴趣收敛为范围明确、可检验的研究问题，为开题讨论提供初稿。
- **文献探索与研究定位**：按主题整理相关论文，识别既有工作的局限、假设冲突和潜在研究机会。
- **研究设计辅助**：梳理理论机制、变量操作化、基线、消融实验、评价指标和负结果的意义。
- **Agent 方法实验**：比较单提示词、多阶段、无检索及候选选择模式，观察质量、稳定性和资源消耗。

当前示例与历史实验主要围绕 LLM、AI、编程教育等主题。系统接受其他领域主题，但尚不能据此认定其在所有学科具有相同效果。文献处理主要基于标题、摘要与元数据；尚未实现通用论文全文阅读、实际科研实验执行或研究结论验证。

### 已实现的核心能力

| 能力 | 当前实现 |
| --- | --- |
| 多源文献检索 | 接入 arXiv、Semantic Scholar、Crossref；包含查询清理、来源配额、去重、缓存和检索快照 |
| 文献组织与证据整理 | 向量相似度重排、凝聚式聚类、主题总结，以及逐篇结构化证据笔记 |
| 分阶段研究生成 | 从学术定位、研究范围、缺口与问题，推进到机制、假设、方法、贡献和 proposal |
| 研究判断 | 分析跨论文张力，检查研究问题有效性，选择主要贡献类型，识别失败模式并检查证据与主张的对应关系 |
| 方案检查与修订 | 完整模式包含草稿批评与修订；支持可选的目标评分诊断与再次修订、逻辑图检查和最终评分 |
| 候选方案选择 | 在研究缺口、假设和方法阶段生成多个候选，结合模型评价与向量对齐分数进行选择 |
| 轻量判断模式 | `judgement_light` 已实现：以结构化研究蓝图约束 proposal，检查蓝图遵循情况，并跳过部分批评与检查环节 |
| 策略学习 | 基于 epsilon-greedy bandit 更新外部策略统计、探索策略变体，并调整评分维度的提示关注权重 |
| 可视化与实验记录 | React 界面支持单次运行、实验对比和演示数据；后端记录结果、检索快照、评分、tokens、耗时及回退信息 |

### 主要技术

- **后端**：Python、FastAPI、Pydantic、HTTPX、Uvicorn，提供异步 API 与结构化数据模型。
- **模型接入**：通过 OpenAI Python SDK 调用兼容接口，聊天模型、服务地址和 embedding 模型均可配置。
- **检索增强生成**：将论文主题和证据笔记传入后续提示链；当前使用外部学术 API 与本地 JSON 缓存，无需向量数据库。
- **文献向量分析**：使用 embedding、余弦相似度与凝聚式聚类；embedding 服务不可用时退回本地 128 维词元哈希向量。
- **流程编排**：以 Python 显式组织提示链、JSON 输出、候选选择、检查和修订，提示模板保存在 `backend/prompts/`。
- **前端**：React 19、TypeScript、Vite、Lucide React，展示论文、中间研究结果、proposal、评分与实验指标。
- **学习与持久化**：外部 bandit 策略、评分反馈、弱项标签及本地 JSON 策略状态。

### 设计创新与技术探索

这里的“创新”指本项目已实现的设计特点与研究探索，不代表已证明的学术首创或性能领先。

1. **从文献总结推进到研究问题形成。** 流程显式分析文献之间的矛盾、未解释机制和假设差异，再检查问题是否具体、有意义且可检验，使输出更聚焦研究设计。
2. **用研究蓝图衔接判断与写作。** 蓝图集中表达研究机制、最强基线、负结果意义、必要证据及必须包含或避免的内容；轻量模式额外检查 proposal 是否遵循蓝图，目标是减少中间判断在写作阶段丢失。
3. **在关键决策点比较候选。** 对缺口、假设和方法进行候选生成与选择，将显式评价标准与语义对齐信号结合，保留候选及选择信息供检查。
4. **在 Agent 层学习执行策略。** 系统根据历史运行反馈调整流程条件、文献数量、提示风格和目标修订配置，并优先探索更高的已观察质量分数。该机制不更新 LLM 参数，也不是 PPO、GRPO 或模型微调。
5. **把评审弱项反馈到后续提示。** 按清晰度、逻辑、新颖性、可行性、文献一致性、博士级研究质量和表达七个维度更新关注权重。这是外部提示控制，不是 Transformer 内部 attention。
6. **同时观察质量与代价。** 保存检索快照并支持重复对照实验，记录 token、延迟、字段完整度和回退情况，便于分析“增加步骤是否值得”。

### 完成情况与结果边界

核心研究流程、前后端展示、实验接口、策略学习、研究蓝图和轻量判断模式均已在当前代码中实现。策略训练及高级流程参数通过后端 API 使用，前端目前主要提供单次生成与实验对比。

当前限制：

- 自动评分由 LLM 及内部 rubric 产生，不能替代领域专家评审，也不能证明真实新颖性或博士研究水平。
- 更多文献和更多步骤尚未表现出稳定的质量收益；完整流程可能产生较高 token 消耗与等待时间。
- embedding 哈希回退、文献回退和生成回退会影响结果解释。应结合后端日志、论文来源及 `metrics.fallback_reason` 判断运行状态；单一字段不能覆盖所有局部降级。
- 前端演示数据与后端回退结果用于联调，不能计入真实实验结论。文献回退结果也可能进入缓存，重新检索时需检查对应缓存。
- 当前以本地运行和 JSON 文件保存状态为主，尚未提供用户认证、数据库或后台任务队列。

## English

### Purpose and scope

Academic Research Agent supports research ideation and proposal design. Starting from a topic, it retrieves related literature, organizes evidence and disagreements, and develops a research gap, question, hypothesis, methodology, and proposal through staged generation and structured review.

It is intended for:

- **Graduate and course-project ideation:** turning broad interests into scoped, testable questions and drafts for supervisor discussions.
- **Literature exploration:** organizing related papers and identifying limitations, conflicting assumptions, and possible research opportunities.
- **Research-design assistance:** articulating mechanisms, operationalized variables, baselines, ablations, metrics, and the meaning of negative results.
- **Agent experiments:** comparing single-prompt, multi-stage, no-retrieval, and candidate-selection conditions in terms of quality, stability, and resource use.

Current examples and historical experiments mainly concern LLMs, AI, and programming education. Other topics are accepted, but equal effectiveness across disciplines has not been established. Evidence processing primarily uses titles, abstracts, and metadata. General full-paper reading, execution of scientific experiments, and verification of research findings are not implemented.

### Implemented capabilities

| Capability | Current implementation |
| --- | --- |
| Multi-source retrieval | arXiv, Semantic Scholar, and Crossref, with query cleaning, source quotas, deduplication, caching, and snapshots |
| Literature organization | Embedding-based reranking, agglomerative clustering, theme summaries, and structured per-paper evidence notes |
| Staged research generation | Scholarly positioning, scope, gap, question, mechanism, hypothesis, methodology, contribution, and proposal |
| Research judgement | Cross-paper tension analysis, problem-validity checks, contribution routing, failure-pattern detection, and evidence–claim alignment |
| Review and revision | Draft critique and revision in full mode; optional target-score diagnosis and revision, logic-graph checks, and final evaluation |
| Candidate selection | Multiple alternatives for gaps, hypotheses, and methodologies, selected using model assessments and embedding alignment |
| Lightweight judgement | Implemented `judgement_light` mode uses a structured research blueprint and a compliance check while skipping selected review stages |
| Strategy learning | An epsilon-greedy bandit updates external strategy statistics, explores variants, and adjusts rubric-focused prompt guidance |
| Interface and records | React views for single runs, comparisons, and demo data; backend records outputs, snapshots, scores, tokens, latency, and fallback information |

### Technology stack

- **Backend:** Python, FastAPI, Pydantic, HTTPX, and Uvicorn for asynchronous endpoints and structured data contracts.
- **Model access:** the OpenAI Python SDK with configurable compatible endpoints, chat models, and embedding models.
- **Retrieval-augmented generation:** literature themes and evidence notes feed downstream prompts; external scholarly APIs and local JSON caches require no vector database.
- **Vector analysis:** embeddings, cosine similarity, and agglomerative clustering, with a local 128-dimensional token-hashing fallback when embeddings are unavailable.
- **Orchestration:** explicit Python prompt chains, JSON outputs, candidate selection, checks, and revisions; templates live in `backend/prompts/`.
- **Frontend:** React 19, TypeScript, Vite, and Lucide React for papers, intermediate outputs, proposals, scores, and experiment metrics.
- **Learning and persistence:** external bandit strategies, rubric feedback, weakness tags, and local JSON policy state.

### Design contributions and technical exploration

These contributions describe implemented project designs, not established claims of academic priority or superior performance.

1. **From literature summaries to problem formation.** Explicit tension analysis and validity checks connect conflicting assumptions and unexplained mechanisms to specific, meaningful, testable questions.
2. **A blueprint between judgement and drafting.** A structured blueprint captures the mechanism, strongest baseline, negative-result interpretation, required evidence, and proposal constraints. Lightweight mode checks compliance to help preserve earlier decisions during drafting.
3. **Candidate comparison at key decisions.** Gaps, hypotheses, and methodologies are generated as alternatives and assessed with explicit criteria and semantic-alignment signals; selection records remain available for inspection.
4. **Learning at the agent level.** Historical feedback adapts workflow conditions, retrieval depth, prompt profiles, and target-revision settings. Selection prioritizes higher observed quality. It does not update LLM weights and is not PPO, GRPO, or model fine-tuning.
5. **Feedback-directed prompt focus.** Weaknesses update attention weights over clarity, logic, novelty, feasibility, literature alignment, PhD-level quality, and presentation. This is external prompt control, not Transformer attention.
6. **Quality and cost measured together.** Retrieval snapshots and repeated comparisons support analysis of quality alongside token use, latency, field completeness, and fallback behavior.

### Completion status and evidence limits

The current code implements the core pipeline, frontend/backend integration, experiment APIs, strategy learning, research blueprints, and lightweight judgement mode. Training and advanced pipeline settings are accessed through backend APIs; the frontend primarily provides single-run and experiment views.

Current limitations:

- Scores come from an LLM and an internal rubric, not independent expert review or proof of novelty or doctoral-level quality.
- More papers and stages have not demonstrated consistent gains; full runs can be expensive and slow.
- Embedding, retrieval, and generation fallbacks affect interpretation. Inspect backend logs, paper sources, and `metrics.fallback_reason`; one field does not capture every local degradation.
- Demo data and backend fallback outputs support integration testing, not empirical quality claims. Retrieval fallback results can also be cached; inspect the matching cache when requesting fresh evidence.
- The implementation targets local use with JSON state; user authentication, a database, and a background job queue are not provided.

<a id="quick-start"></a>
## 快速开始 / Quick start

以下命令使用 Windows PowerShell。从包含 `backend/` 和 `frontend/` 的仓库根目录开始。需要 Python 3.10+（代码语法要求）以及可运行本项目 Vite 6 的 Node.js/npm 环境。  
The commands below use Windows PowerShell, starting at the repository root containing `backend/` and `frontend/`. Use Python 3.10+ as required by the code syntax, and a Node.js/npm environment compatible with the project's Vite 6 dependency.

### 1. 后端 / Backend

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
```

编辑 `backend/.env`，将以下占位值替换为你的服务配置。若文件已存在，保留原配置，不必重复复制。  
Edit `backend/.env`, replacing these placeholders with your provider settings. If it already exists, keep your configuration instead of copying over it.

```dotenv
LLM_API_KEY=your-api-key
LLM_BASE_URL=https://your-provider.example/v1
LLM_MODEL=your-chat-model
EMBEDDING_MODEL=your-supported-embedding-model
SEMANTIC_SCHOLAR_API_KEY=
LLM_TEMPERATURE=0.3
```

聊天与 embedding 使用同一套 API key 和服务地址；请确认服务提供所填 embedding 模型，否则系统将使用本地哈希回退。Semantic Scholar key 可选。未配置 LLM key 时可使用后端回退结果进行联调，但不是真实模型生成。  
Chat and embeddings share the API key and endpoint. Configure an embedding model supported by that provider, or local hashing will be used. The Semantic Scholar key is optional. Without an LLM key, backend fallback outputs support integration testing rather than real model generation.

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

交互 API 文档 / Interactive API docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### 2. 前端 / Frontend

在新的终端中，从仓库根目录运行：  
In a second terminal, start from the repository root:

```powershell
cd frontend
npm install
npm run dev -- --host 127.0.0.1
```

打开 [http://127.0.0.1:5173](http://127.0.0.1:5173)。界面默认启用演示数据；连接真实后端时，取消勾选 **Use demo data while backend is unavailable**，输入主题并运行。`Experiment` 页面支持选择条件与重复次数。  
Open [http://127.0.0.1:5173](http://127.0.0.1:5173). Demo mode is enabled by default. Uncheck **Use demo data while backend is unavailable** to use the backend, enter a topic, and run. The `Experiment` page supports condition selection and repeated runs.

如需更改后端地址，在 `frontend/.env.local` 中设置 `VITE_API_BASE_URL=http://127.0.0.1:8000`，然后重启前端。  
To change the backend address, set `VITE_API_BASE_URL=http://127.0.0.1:8000` in `frontend/.env.local` and restart the frontend.

## API

| Method | Endpoint | 用途 / Purpose |
| --- | --- | --- |
| GET | `/health` | 健康检查 / Health check |
| POST | `/api/research/run` | 单次研究生成 / Single research run |
| POST | `/api/experiment/run` | 多条件重复对比 / Repeated condition comparisons |
| GET | `/api/agent/policy` | 查看策略与关注权重 / Inspect policy and attention weights |
| POST | `/api/agent/train` | 执行学习 episode 并保存策略 / Run learning episodes and persist the policy |

### 运行模式 / Research conditions

| `condition` | 实际行为 / Actual behavior |
| --- | --- |
| `baseline_with_retrieval` | 历史命名；当前为不检索文献的单提示词基线 / Legacy name: currently a single-prompt baseline without retrieval |
| `multi_stage_with_retrieval` | 文献增强的多阶段流程，默认条件 / Literature-grounded multi-stage pipeline; default |
| `multi_stage_without_retrieval` | 无文献输入的多阶段对照 / Multi-stage control without literature input |
| `multi_stage_group_selection` | 带检索与关键阶段候选选择 / Retrieval with candidate selection at key stages |

`main_comparison` 是保留的单次运行兼容模式，会在多阶段结果中附带 baseline；正式条件对比可使用 `/api/experiment/run`。  
`main_comparison` remains a legacy single-run mode that includes a baseline alongside multi-stage output. Use `/api/experiment/run` for explicit condition comparisons.

### 生成研究方案 / Generate a proposal

在 `/docs` 中向 `/api/research/run` 提交以下 JSON，或使用下方 PowerShell 示例。  
Submit this JSON to `/api/research/run` through `/docs`, or use the PowerShell example below.

```json
{
  "topic": "LLM-based feedback generation for programming education",
  "max_papers": 5,
  "condition": "multi_stage_with_retrieval",
  "prompt_profile": "compact_85",
  "pipeline_mode": "judgement_light",
  "target_revision": true
}
```

```powershell
$researchBody = @{
    topic = "LLM-based feedback generation for programming education"
    max_papers = 5
    condition = "multi_stage_with_retrieval"
    prompt_profile = "compact_85"
    pipeline_mode = "judgement_light"
    target_revision = $true
} | ConvertTo-Json
Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/research/run" -Method Post -ContentType "application/json" -Body $researchBody
```

`max_papers` 接受 1–10；`pipeline_mode` 默认 `full`，可选 `judgement_light`；`target_revision` 默认 `false`。响应包括 `papers`、中间研究字段、`proposal`、`evaluation` 和 `metrics`，部分字段随模式省略或为空。  
`max_papers` accepts 1–10; `pipeline_mode` defaults to `full` and also supports `judgement_light`; `target_revision` defaults to `false`. Responses include `papers`, intermediate research fields, `proposal`, `evaluation`, and `metrics`, with some fields omitted or empty depending on the mode.

### 对照实验 / Comparison experiment

提交至 / Submit to `POST /api/experiment/run`:

```json
{
  "topic": "LLM-based feedback generation for programming education",
  "max_papers": 5,
  "conditions": [
    "baseline_with_retrieval",
    "multi_stage_with_retrieval",
    "multi_stage_without_retrieval",
    "multi_stage_group_selection"
  ],
  "repeats": 2
}
```

条件与重复运行顺序执行；检索条件复用同一实验的论文集合。返回逐次结果、质量对比与稳定性/成本表。此接口目前使用默认流程参数，不接收 `pipeline_mode` 或 `prompt_profile`。  
Conditions and repeats run sequentially; retrieval-enabled conditions reuse the experiment's paper set. The response includes individual runs, quality comparisons, and stability/cost tables. This endpoint currently uses default pipeline settings and does not accept `pipeline_mode` or `prompt_profile`.

### 策略学习 / Strategy learning

提交至 / Submit to `POST /api/agent/train`:

```json
{
  "topics": [
    "LLM-based feedback generation for programming education",
    "Retrieval augmented generation for academic literature review"
  ],
  "episodes": 4,
  "max_papers": 5,
  "epsilon": 0.2,
  "warm_start": true,
  "warm_start_max_runs": 60
}
```

训练会调用模型并更新 `backend/data/agent_policy.json`。`warm_start` 从历史运行补充策略统计；`warm_start: false` 仍加载已有策略，不会清空状态。实际文献数量由所选策略优先决定。普通 `/api/research/run` 按请求参数执行，不自动调用 bandit 选择策略。  
Training calls the model and updates `backend/data/agent_policy.json`. Warm start incorporates historical runs; `warm_start: false` still loads the existing policy rather than resetting it. The selected strategy takes precedence for retrieval depth. Ordinary `/api/research/run` calls follow request parameters and do not automatically invoke bandit strategy selection.

## 项目结构 / Repository structure

```text
.
├── README.md
├── backend/
│   ├── app/
│   │   ├── main.py                 # API endpoints
│   │   ├── pipeline.py             # Research orchestration
│   │   ├── literature_client.py    # Multi-source retrieval
│   │   ├── literature_clustering.py# Embeddings and clustering
│   │   ├── llm_client.py           # Compatible model client
│   │   ├── experiment.py           # Condition comparisons
│   │   ├── agent_rl.py             # Strategy policy and feedback
│   │   ├── agent_training.py       # Learning episodes
│   │   ├── rubric.py              # Evaluation rubric
│   │   └── models.py              # Request/response schemas
│   ├── prompts/                   # Stage prompt templates
│   ├── docs/                      # Historical learning/experiment notes
│   ├── data/                      # Runtime records and policy state
│   ├── .env.example
│   └── requirements.txt
└── frontend/
    ├── src/                       # UI, API client, types, demo data
    └── package.json
```

运行记录 / Runtime records:

- `backend/data/cache/literature/`: 文献缓存 / Literature cache.
- `backend/data/retrieval_snapshots/`: 检索快照 / Retrieval snapshots.
- `backend/data/runs/`: 逐次研究结果 / Individual research outputs.
- `backend/data/agent_policy.json`: 策略统计与关注权重 / Policy statistics and attention weights.

更多实现背景见 [Agent learning notes](backend/docs/agent_rl_learning_summary.md) 和 [Historical experiment results](backend/docs/recent_agent_experiment_results.md)。这些文档记录特定开发阶段，部分建议（例如实现轻量模式）已被当前代码落实。  
See [Agent learning notes](backend/docs/agent_rl_learning_summary.md) and [Historical experiment results](backend/docs/recent_agent_experiment_results.md) for development context. They describe particular milestones; some proposed changes, including lightweight mode, are already implemented in the current code.
