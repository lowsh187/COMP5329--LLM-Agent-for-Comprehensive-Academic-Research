import {
  AlertCircle,
  BarChart3,
  BookOpen,
  BrainCircuit,
  ClipboardList,
  FileText,
  FlaskConical,
  Lightbulb,
  Loader2,
  Search,
  Sparkles
} from "lucide-react";
import { FormEvent, useMemo, useState } from "react";
import { runResearch } from "./api";
import type { Evaluation, Paper, ResearchResult } from "./types";

const sampleTopics = [
  "LLM-based feedback generation for programming education",
  "Retrieval augmented generation for academic literature review",
  "AI-assisted research question generation in higher education"
];

export function App() {
  const [topic, setTopic] = useState(sampleTopics[0]);
  const [maxPapers, setMaxPapers] = useState(5);
  const [demoMode, setDemoMode] = useState(true);
  const [result, setResult] = useState<ResearchResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError("");
    setLoading(true);

    try {
      const data = await runResearch({ topic: topic.trim(), max_papers: maxPapers }, demoMode);
      setResult(data);
    } catch (err) {
      setResult(null);
      setError(err instanceof Error ? err.message : "Unknown frontend integration error");
    } finally {
      setLoading(false);
    }
  }

  const stages = useMemo(() => {
    if (!result) return [];
    return [
      { title: "Literature Summary", value: result.summary, icon: BookOpen },
      { title: "Research Gap", value: result.gap, icon: Search },
      { title: "Research Question", value: result.question, icon: BrainCircuit },
      { title: "Hypothesis", value: result.hypothesis, icon: Lightbulb },
      { title: "Methodology", value: result.methodology, icon: FlaskConical },
      { title: "Proposal Draft", value: result.proposal, icon: FileText }
    ];
  }, [result]);

  return (
    <main className="app">
      <section className="layout">
        <aside className="sidebar">
          <div className="product-title">
            <Sparkles aria-hidden="true" />
            <div>
              <h1>Academic Research Agent</h1>
              <p>Frontend workspace for Member B</p>
            </div>
          </div>

          <form className="input-panel" onSubmit={handleSubmit}>
            <label htmlFor="topic">Research topic</label>
            <textarea
              id="topic"
              rows={5}
              value={topic}
              onChange={(event) => setTopic(event.target.value)}
              placeholder="Enter a research topic"
            />

            <div className="number-row">
              <label htmlFor="max-papers">Papers</label>
              <input
                id="max-papers"
                type="number"
                min={1}
                max={10}
                value={maxPapers}
                onChange={(event) => setMaxPapers(Number(event.target.value))}
              />
            </div>

            <label className="check-row">
              <input type="checkbox" checked={demoMode} onChange={(event) => setDemoMode(event.target.checked)} />
              Use demo data while backend is unavailable
            </label>

            <button type="submit" disabled={loading || topic.trim().length < 3}>
              {loading ? <Loader2 className="spin" aria-hidden="true" /> : <Search aria-hidden="true" />}
              Generate
            </button>
          </form>

          <div className="samples">
            {sampleTopics.map((sample) => (
              <button type="button" key={sample} onClick={() => setTopic(sample)}>
                {sample}
              </button>
            ))}
          </div>
        </aside>

        <section className="content">
          {error ? (
            <div className="message error">
              <AlertCircle aria-hidden="true" />
              <span>{error}</span>
            </div>
          ) : null}

          {!result && !loading ? <EmptyState /> : null}
          {loading ? <LoadingState /> : null}

          {result ? (
            <div className="result-stack">
              <header className="result-header">
                <div>
                  <p className="overline">{demoMode ? "Demo data" : "Backend integration"}</p>
                  <h2>{result.topic}</h2>
                </div>
              </header>

              <PaperSection papers={result.papers} />

              <section className="stage-grid">
                {stages.map((stage) => (
                  <article className="stage-card" key={stage.title}>
                    <div className="card-title">
                      <stage.icon aria-hidden="true" />
                      <h3>{stage.title}</h3>
                    </div>
                    <p>{stage.value}</p>
                  </article>
                ))}
              </section>

              <EvaluationPanel evaluation={result.evaluation} />

              {result.baseline ? <BaselinePanel result={result} /> : null}
            </div>
          ) : null}
        </section>
      </section>
    </main>
  );
}

function EmptyState() {
  return (
    <div className="empty-state">
      <ClipboardList aria-hidden="true" />
      <h2>Ready for frontend-backend testing</h2>
      <p>Enter a topic, run demo data, or connect to the backend endpoint when Member A provides it.</p>
    </div>
  );
}

function LoadingState() {
  return (
    <div className="empty-state">
      <Loader2 className="spin" aria-hidden="true" />
      <h2>Generating results</h2>
      <p>Loading state is visible while the backend or demo data responds.</p>
    </div>
  );
}

function PaperSection({ papers }: { papers: Paper[] }) {
  return (
    <section className="panel-section">
      <h3>Retrieved Papers</h3>
      <div className="paper-grid">
        {papers.map((paper, index) => (
          <article className="paper-card" key={`${paper.title}-${index}`}>
            <a href={paper.url} target="_blank" rel="noreferrer">
              {paper.title}
            </a>
            <p className="paper-meta">
              {[paper.source, paper.authors.join(", ") || "Unknown author"].filter(Boolean).join(" | ")}
            </p>
            <p>{paper.abstract}</p>
          </article>
        ))}
      </div>
    </section>
  );
}

function EvaluationPanel({ evaluation }: { evaluation: Evaluation }) {
  const scores = [
    ["Clarity", evaluation.clarity],
    ["Logic", evaluation.logic],
    ["Novelty", evaluation.novelty],
    ["Feasibility", evaluation.feasibility],
    ["Literature Alignment", evaluation.literature_alignment]
  ] as const;

  return (
    <section className="evaluation-panel">
      <div className="card-title">
        <BarChart3 aria-hidden="true" />
        <h3>Self-Evaluation</h3>
      </div>

      <div className="score-grid">
        {scores.map(([label, score]) => (
          <div className="score-card" key={label}>
            <span>{label}</span>
            <strong>{score}/5</strong>
            <div className="score-track">
              <i style={{ width: `${score * 20}%` }} />
            </div>
          </div>
        ))}
      </div>

      <p>{evaluation.comments}</p>
    </section>
  );
}

function BaselinePanel({ result }: { result: ResearchResult }) {
  const baseline = result.baseline;
  if (!baseline) return null;

  const pipelineAverage = averageScore(result.evaluation);
  const baselineAverage = averageScore(baseline.evaluation);

  return (
    <section className="comparison-panel">
      <div className="card-title">
        <BarChart3 aria-hidden="true" />
        <h3>Evaluation Support</h3>
      </div>

      <div className="comparison-grid">
        <div>
          <span>Multi-stage pipeline</span>
          <strong>{pipelineAverage.toFixed(1)}/5</strong>
        </div>
        <div>
          <span>Single-prompt baseline</span>
          <strong>{baselineAverage.toFixed(1)}/5</strong>
        </div>
      </div>

      {baseline.proposal ? <p>{baseline.proposal}</p> : null}
    </section>
  );
}

function averageScore(evaluation?: Partial<Evaluation>): number {
  if (!evaluation) return 0;
  const values = [
    evaluation.clarity,
    evaluation.logic,
    evaluation.novelty,
    evaluation.feasibility,
    evaluation.literature_alignment
  ].filter((value): value is number => typeof value === "number");

  if (values.length === 0) return 0;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}
