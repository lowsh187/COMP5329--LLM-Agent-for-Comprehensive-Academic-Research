# Academic Research Agent · 学术研究智能体

[中文](#chinese) | [English](#english)

<a id="chinese"></a>
## 中文

[Switch to English](#english)

**从研究主题出发，形成有文献依据、可检验的研究方案。**

本文对应 `agent-rl-learning` 分支已实现的研究原型。

### 团队分工与个人贡献

#### Member A — 王士鸿（Shihong Wang）· 核心开发

- **系统与后端**：系统架构、后端开发、模型与文献 API 集成、Prompt Engineering、Agent 工作流及最终系统整合。
- **RL learning 设计**：设计 Agent 层的强化学习式策略优化流程，组织训练 episode、运行反馈与学习状态持久化；不更新 LLM 参数。
- **Policy 制定与策略空间设计**：设计质量优先的 epsilon-greedy bandit policy，定义检索深度、提示风格、流程模式和目标修订等策略配置，以及探索与策略变体机制。
- **奖励与反馈机制**：设计质量、目标接近程度、资源成本和回退状态的奖励信号，结合弱项标签更新评分维度关注权重，指导后续生成。
- **评价与验证**：研究评价框架、单提示词与多阶段对照实验，以及质量、稳定性和成本分析。

代码入口：[研究流程](backend/app/pipeline.py)、[提示模板](backend/prompts/)、[文献检索](backend/app/literature_client.py)、[模型接入](backend/app/llm_client.py)、[Policy 与策略学习](backend/app/agent_rl.py)、[训练流程](backend/app/agent_training.py)、[评价规则](backend/app/rubric.py)、[对照实验](backend/app/experiment.py)。

#### Member B — 朱宴岚（Yanlan Zhu）· 前端与测试支持

负责前端界面、结果可视化、前后端联调、测试、评价支持与问题修复。代码入口：[frontend/src/](frontend/src/)。

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

### 快速开始

从包含 `backend/` 和 `frontend/` 的仓库根目录开始。以下命令使用 Windows PowerShell，需要 Python 3.10+ 和兼容 Vite 6 的 Node.js/npm 环境。

#### 1. 后端

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

首次配置时，在 `backend/` 运行 `Copy-Item .env.example .env`。如果 `.env` 已存在，保留原配置。编辑该文件，替换以下占位值：

```dotenv
LLM_API_KEY=your-api-key
LLM_BASE_URL=https://your-provider.example/v1
LLM_MODEL=your-chat-model
EMBEDDING_MODEL=your-supported-embedding-model
SEMANTIC_SCHOLAR_API_KEY=
LLM_TEMPERATURE=0.3
```

聊天与 embedding 共用 API key 和服务地址。需要服务支持所填 embedding 模型，否则退回本地哈希向量；Semantic Scholar key 可选。未配置 LLM key 时，后端回退结果仅用于联调。

启动后端：

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

交互 API 文档：[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)。

#### 2. 前端

在新终端中，从仓库根目录运行：

```powershell
cd frontend
npm install
npm run dev -- --host 127.0.0.1
```

打开 [http://127.0.0.1:5173](http://127.0.0.1:5173)。界面默认启用演示数据；真实运行时，取消勾选 **Use demo data while backend is unavailable**，然后输入主题运行。`Experiment` 页面支持条件选择和重复运行。

如需更改后端地址，在 `frontend/.env.local` 设置 `VITE_API_BASE_URL=http://127.0.0.1:8000` 并重启前端。前端构建检查命令为 `npm run build`。

### API

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| GET | `/health` | 健康检查 |
| POST | `/api/research/run` | 单次研究生成 |
| POST | `/api/experiment/run` | 多条件重复对比 |
| GET | `/api/agent/policy` | 查看策略与关注权重 |
| POST | `/api/agent/train` | 执行训练并保存策略 |

#### 运行条件

| `condition` | 实际行为 |
| --- | --- |
| `baseline_with_retrieval` | 历史名称；当前为无检索的单提示词基线 |
| `multi_stage_with_retrieval` | 默认条件：文献增强的多阶段流程 |
| `multi_stage_without_retrieval` | 无文献输入的多阶段对照 |
| `multi_stage_group_selection` | 带检索及关键阶段候选选择 |

`main_comparison` 是兼容模式，在多阶段结果中附带 baseline。显式条件对比请使用 `/api/experiment/run`。

#### 生成研究方案

向 `POST /api/research/run` 提交：

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

`max_papers` 接受 1–10；`pipeline_mode` 默认 `full`，也支持 `judgement_light`；`target_revision` 默认 `false`。响应包含论文、中间研究字段、proposal、评价和运行指标，部分字段随模式为空。

#### 对照实验

向 `POST /api/experiment/run` 提交：

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

条件与重复运行顺序执行，检索条件复用该实验的论文集合。响应包括逐次结果、质量对比与稳定性/成本表。此接口使用默认流程配置，不接收 `pipeline_mode` 和 `prompt_profile`。

#### 策略学习

向 `POST /api/agent/train` 提交：

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

训练调用模型并更新 `backend/data/agent_policy.json`。`warm_start` 利用历史运行补充统计；设为 `false` 仍加载已有策略，不会清空状态。实际文献数量由所选策略优先决定。普通研究接口按请求参数运行，不自动使用 bandit 选择策略。

完整字段以运行中后端的 `/docs` 和[数据模型](backend/app/models.py)为准。

### 项目结构与文档

| 路径 | 内容 |
| --- | --- |
| `backend/app/` | API、检索、研究流程、策略学习及评价 |
| `backend/prompts/` | 各阶段提示模板 |
| `backend/docs/` | Agent 技术说明 |
| `frontend/src/` | 界面、API 客户端、类型及演示数据 |
| `backend/data/cache/literature/` | 文献缓存 |
| `backend/data/retrieval_snapshots/` | 检索快照 |
| `backend/data/runs/` | 逐次研究结果 |
| `backend/data/agent_policy.json` | 策略统计与关注权重 |

实现细节见 [Agent 技术说明](backend/docs/agent_rl_learning_summary.md#chinese)。

---

<a id="english"></a>
## English

[切换到中文](#chinese)

**From a research topic to an evidence-informed, testable research proposal.**

This guide describes the research prototype implemented on `agent-rl-learning`.

### Team and contributions

#### Member A — Shihong Wang · Lead developer

- **System and backend:** system architecture, backend development, model and literature API integration, prompt engineering, agent workflows, and final system integration.
- **RL learning design:** design of the agent-level reinforcement-style optimization loop, including training episodes, runtime feedback, and persistent learning state, without updating LLM weights.
- **Policy and strategy-space design:** a quality-oriented epsilon-greedy bandit policy, strategy configurations for retrieval depth, prompt profiles, pipeline modes, and target revision, plus exploration and strategy-variant mechanisms.
- **Reward and feedback design:** reward signals covering quality, target proximity, resource costs, and fallback behavior; weakness-driven updates to rubric attention weights that guide subsequent generation.
- **Evaluation and validation:** the research evaluation framework, single-prompt and multi-stage comparisons, and quality, stability, and cost analysis.

Source entry points: [research pipeline](backend/app/pipeline.py), [prompt templates](backend/prompts/), [literature retrieval](backend/app/literature_client.py), [model client](backend/app/llm_client.py), [policy and strategy learning](backend/app/agent_rl.py), [training loop](backend/app/agent_training.py), [evaluation rubric](backend/app/rubric.py), and [comparison experiments](backend/app/experiment.py).

#### Member B — Yanlan Zhu · Frontend and testing support

Responsible for frontend UI, result visualization, frontend–backend integration, testing, evaluation support, and bug fixing. Source entry point: [frontend/src/](frontend/src/).

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

### Quick start

Start at the repository root containing `backend/` and `frontend/`. These Windows PowerShell commands require Python 3.10+ and a Node.js/npm environment compatible with Vite 6.

#### 1. Backend

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

For first-time setup, run `Copy-Item .env.example .env` in `backend/`. Preserve an existing `.env` instead of overwriting it. Replace these placeholders with your provider settings:

```dotenv
LLM_API_KEY=your-api-key
LLM_BASE_URL=https://your-provider.example/v1
LLM_MODEL=your-chat-model
EMBEDDING_MODEL=your-supported-embedding-model
SEMANTIC_SCHOLAR_API_KEY=
LLM_TEMPERATURE=0.3
```

Chat and embeddings share the API key and endpoint. Configure a supported embedding model or the system will fall back to local hashing vectors. The Semantic Scholar key is optional. Without an LLM key, backend fallback output is for integration testing.

Start the backend:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Interactive API docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

#### 2. Frontend

In a second terminal, start from the repository root:

```powershell
cd frontend
npm install
npm run dev -- --host 127.0.0.1
```

Open [http://127.0.0.1:5173](http://127.0.0.1:5173). Demo mode is enabled by default. Uncheck **Use demo data while backend is unavailable** for real backend requests, then enter a topic and run. The `Experiment` page supports condition selection and repeated runs.

To change the backend address, set `VITE_API_BASE_URL=http://127.0.0.1:8000` in `frontend/.env.local` and restart the frontend. Run `npm run build` to check the frontend build.

### API

| Method | Endpoint | Purpose |
| --- | --- | --- |
| GET | `/health` | Health check |
| POST | `/api/research/run` | Single research run |
| POST | `/api/experiment/run` | Repeated condition comparisons |
| GET | `/api/agent/policy` | Inspect policy and attention weights |
| POST | `/api/agent/train` | Run learning episodes and persist the policy |

#### Research conditions

| `condition` | Actual behavior |
| --- | --- |
| `baseline_with_retrieval` | Legacy name: a single-prompt baseline without retrieval |
| `multi_stage_with_retrieval` | Default: literature-grounded multi-stage generation |
| `multi_stage_without_retrieval` | Multi-stage control without literature input |
| `multi_stage_group_selection` | Retrieval and candidate selection at key stages |

`main_comparison` is a legacy mode that includes a baseline alongside multi-stage output. Use `/api/experiment/run` for explicit condition comparisons.

#### Generate a proposal

Submit to `POST /api/research/run`:

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

`max_papers` accepts 1–10; `pipeline_mode` defaults to `full` and also supports `judgement_light`; `target_revision` defaults to `false`. Responses include papers, intermediate research fields, a proposal, evaluation, and metrics, with some fields empty depending on the mode.

#### Comparison experiment

Submit to `POST /api/experiment/run`:

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

Conditions and repeats run sequentially. Retrieval-enabled conditions reuse the experiment's paper set. The response includes individual runs, quality comparisons, and stability/cost tables. This endpoint uses default pipeline settings and does not accept `pipeline_mode` or `prompt_profile`.

#### Strategy learning

Submit to `POST /api/agent/train`:

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

Training calls the model and updates `backend/data/agent_policy.json`. Warm start incorporates historical runs; `warm_start: false` still loads the existing policy rather than resetting it. The selected strategy takes precedence for retrieval depth. Ordinary research requests follow explicit parameters without automatic bandit selection.

For complete fields, use the running backend's `/docs` and the [data models](backend/app/models.py).

### Repository structure and documentation

| Path | Contents |
| --- | --- |
| `backend/app/` | APIs, retrieval, research pipelines, strategy learning, and evaluation |
| `backend/prompts/` | Stage prompt templates |
| `backend/docs/` | Agent technical documentation |
| `frontend/src/` | UI, API client, types, and demo data |
| `backend/data/cache/literature/` | Literature cache |
| `backend/data/retrieval_snapshots/` | Retrieval snapshots |
| `backend/data/runs/` | Individual research outputs |
| `backend/data/agent_policy.json` | Policy statistics and attention weights |

See [Agent technical notes](backend/docs/agent_rl_learning_summary.md#english) for implementation details.
