import type { ResearchResult } from "./types";

export const demoResult: ResearchResult = {
  topic: "LLM-based feedback generation for programming education",
  condition: "multi_stage_with_retrieval",
  retrieval_query: "LLM-based feedback generation for programming education",
  retrieval_snapshot_id: "demo-snapshot-llm-feedback-5",
  papers: [
    {
      title: "Large Language Models for Automated Programming Feedback",
      authors: ["Demo Author A", "Demo Author B"],
      abstract:
        "This demo paper represents literature retrieved from arXiv. It helps the frontend show paper metadata before the real backend is connected.",
      published: "2025-01-01",
      url: "https://arxiv.org/",
      source: "arXiv"
    },
    {
      title: "Retrieval-Augmented Generation in Educational Support Systems",
      authors: ["Demo Author C"],
      abstract:
        "This demo paper is used for UI testing of abstract text, author display, and external paper links.",
      published: "2025-02-10",
      url: "https://arxiv.org/",
      source: "Semantic Scholar"
    }
  ],
  literature_themes: [
    {
      theme: "Programming Feedback With LLMs",
      paper_indices: [1, 2],
      summary: "This cluster focuses on LLM-generated feedback and retrieval-augmented educational support.",
      limitations: "Existing work often lacks controlled comparisons of feedback grounding.",
      gap_relevance: "The cluster supports a gap around whether literature-grounded feedback improves learning outcomes."
    }
  ],
  summary:
    "Existing work suggests that LLMs can provide scalable programming feedback, but quality varies across task types and student skill levels.",
  gap:
    "Current systems often evaluate feedback accuracy, but fewer studies examine whether literature-grounded LLM feedback improves students' debugging strategies over time.",
  question:
    "How does literature-grounded LLM feedback affect novice programmers' debugging performance compared with generic LLM feedback?",
  hypothesis:
    "Students receiving literature-grounded LLM feedback will show greater improvement in debugging accuracy and explanation quality than students receiving generic feedback.",
  methodology:
    "Run a controlled classroom study with two feedback conditions. Compare pre-test and post-test debugging performance, feedback usefulness ratings, and qualitative reflections.",
  proposal:
    "This proposal investigates whether grounding LLM feedback in relevant educational literature improves programming learning outcomes. The study will compare generic LLM feedback with literature-grounded feedback in a novice programming context. Expected contributions include design guidance for educational LLM systems and empirical evidence about feedback quality.",
  evaluation: {
    clarity: 4,
    logic: 4,
    novelty: 3,
    feasibility: 5,
    literature_alignment: 4,
    phd_level_quality: 4,
    presentation: 4,
    weighted_total: 82,
    rubric_scores: [
      {
        criterion: "Research Clarity & Problem Definition",
        weight: 15,
        score: 13,
        level: "Very Good",
        justification: "The problem and target use case are clear."
      },
      {
        criterion: "Logical Coherence & Multi-Stage Reasoning",
        weight: 15,
        score: 13,
        level: "Very Good",
        justification: "The staged reasoning chain is coherent."
      },
      {
        criterion: "Novelty & Research Contribution",
        weight: 20,
        score: 15,
        level: "Very Good",
        justification: "The literature-grounded comparison gives a clear contribution."
      },
      {
        criterion: "Feasibility & Experimental Design",
        weight: 20,
        score: 17,
        level: "Very Good",
        justification: "The study design has clear groups and metrics."
      },
      {
        criterion: "Literature Alignment & Scholarly Positioning",
        weight: 15,
        score: 12,
        level: "Very Good",
        justification: "The proposal uses retrieved work but could compare more deeply."
      },
      {
        criterion: "PhD-Level Research Quality & Academic Depth",
        weight: 10,
        score: 8,
        level: "Very Good",
        justification: "The proposal shows research depth with manageable scope."
      },
      {
        criterion: "Presentation & Academic Communication",
        weight: 5,
        score: 4,
        level: "Very Good",
        justification: "The writing is clear and professional."
      }
    ],
    comments:
      "The proposal is clear and feasible. Novelty is moderate because LLM feedback is an active research area, but the literature-grounded comparison gives it a focused angle."
  },
  baseline: {
    proposal:
      "A single-prompt baseline proposal would directly generate the full research idea from the topic without separating literature summary, gap, question, and methodology stages.",
    evaluation: {
      clarity: 3,
      logic: 3,
      novelty: 3,
      feasibility: 4,
      literature_alignment: 3
    }
  },
  metrics: {
    condition: "multi_stage_with_retrieval",
    model: "demo-model",
    temperature: 0.3,
    retrieval_query: "LLM-based feedback generation for programming education",
    retrieval_snapshot_id: "demo-snapshot-llm-feedback-5",
    latency_ms: 1840,
    prompt_tokens: 3200,
    completion_tokens: 980,
    total_tokens: 4180,
    json_valid: true,
    required_field_completeness: 1,
    failed_requests: 0,
    retry_count: 0
  }
};
