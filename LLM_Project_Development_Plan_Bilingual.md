# LLM Academic Research Agent Development Plan
# LLM 学术研究 Agent 开发计划

## Project Timeline / 项目时间线
2026/05/08 – 2026/05/24

---

# 1. Project Goal / 项目目标

## English
Develop a multi-stage LLM-based Academic Research Agent capable of:
- Literature retrieval
- Research gap identification
- Hypothesis generation
- Methodology suggestion
- Proposal generation
- Self-evaluation

The project will use existing LLM APIs instead of training custom models.

## 中文
开发一个基于大语言模型（LLM）的多阶段 Academic Research Agent，能够完成：
- 文献检索
- Research Gap 分析
- 假设生成
- 方法推荐
- Proposal 生成
- 自我评估

项目将直接调用现有 LLM API，而不是训练自己的模型。

---

# 2. Recommended Tech Stack / 推荐技术栈

| Component | Technology |
|---|---|
| Frontend | React + Vite |
| Backend | FastAPI / Flask |
| LLM API | OpenAI API |
| Retrieval | arXiv API |
| Storage | JSON / Local |
| Evaluation | Manual Rubric + GPT Self-Evaluation |

---

# 3. Final System Pipeline / 最终系统流程

```text
User Topic
    ↓
Paper Retrieval (arXiv)
    ↓
Literature Summary
    ↓
Research Gap Identification
    ↓
Research Question Generation
    ↓
Hypothesis Generation
    ↓
Methodology Suggestion
    ↓
Proposal Draft Generation
    ↓
Self Evaluation
```

---

# 4. Team Responsibilities / 团队分工

## Member A（Main Developer）

### English
Responsible for:
- System architecture
- Prompt engineering
- Backend development
- API integration
- Agent workflow
- Evaluation framework
- Final integration

### 中文
负责：
- 系统架构设计
- Prompt Engineering
- 后端开发
- API 集成
- Agent 工作流设计
- Evaluation Framework
- 最终系统整合

---

## Member B

### English
Responsible for:
- Frontend UI
- Frontend-backend integration
- Result visualization
- Testing
- Evaluation support
- Bug fixing

### 中文
负责：
- 前端界面开发
- 前后端联调
- 结果可视化
- 测试
- Evaluation 支持
- Bug 修复

---

# 5. Daily Schedule / 每日开发计划

## 2026/05/08 — System Planning Day
### A
- Create project structure
- Test OpenAI API
- Design pipeline

### B
- Initialize React frontend
- Create homepage and input page

---

## 2026/05/09 — Prompt Engineering Day
### A
- Create prompts
- Implement initial backend pipeline

### B
- Create frontend result components
- Connect frontend to backend

---

## 2026/05/10 — Backend Workflow Day
### A
- Implement prompt chaining
- Design JSON output structure

### B
- Dynamic rendering
- Add loading and error states

---

## 2026/05/11 — arXiv Integration Day
### A
- Integrate arXiv API
- Inject retrieved papers into prompts

### B
- Display retrieved papers

---

## 2026/05/12 — Proposal Generation Day
### A
- Generate proposal sections
- Optimize prompts

### B
- Create proposal display page

---

## 2026/05/13 — Self-Evaluation Day
### A
- Build proposal evaluator
- Create rubric scoring

### B
- Create score visualization

---

## 2026/05/14 — Full Pipeline Integration
### A
- Integrate all modules
- Fix backend bugs

### B
- Frontend-backend integration testing

---

## 2026/05/15 — Internal Demo Day
### A
- Test multiple topics
- Optimize prompts

### B
- Improve UI readability

---

## 2026/05/16 — Evaluation Framework Day
### A
- Build baseline system
- Prepare comparison experiments

### B
- Create evaluation tables

---

## 2026/05/17 — Experimental Day
### A
- Run experiments
- Save outputs

### B
- Manual scoring

---

## 2026/05/18 — Result Analysis Day
### A
- Analyze results

### B
- Organize evaluation data

---

## 2026/05/19 — Stability Day
### A
- Add retry mechanisms
- Optimize token usage

### B
- Improve frontend UX

---

## 2026/05/20 — Feature Freeze
### A
- Fix pipeline issues

### B
- Fix UI bugs

---

## 2026/05/21 — Demo Simulation
### Team
- Perform full demo testing

---

## 2026/05/22 — Final Optimization
### A
- Improve output quality

### B
- Improve UI details

---

## 2026/05/23 — Final Stable Build
### Team
- Final testing
- Clean code
- Verify API keys

---

## 2026/05/24 — Submission Day
### Team
- Final verification only
