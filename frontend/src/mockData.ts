import type { ResearchResult } from "./types";

export const demoResult: ResearchResult = {
  topic: "LLM-based feedback generation for programming education",
  papers: [
    {
      title: "Large Language Models for Automated Programming Feedback",
      authors: ["Demo Author A", "Demo Author B"],
      abstract:
        "This demo paper represents literature retrieved from arXiv. It helps the frontend show paper metadata before the real backend is connected.",
      published: "2025-01-01",
      url: "https://arxiv.org/"
    },
    {
      title: "Retrieval-Augmented Generation in Educational Support Systems",
      authors: ["Demo Author C"],
      abstract:
        "This demo paper is used for UI testing of abstract text, author display, and external paper links.",
      published: "2025-02-10",
      url: "https://arxiv.org/"
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
  }
};
