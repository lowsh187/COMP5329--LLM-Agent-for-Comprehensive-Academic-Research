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
  Play,
  Search,
  Sparkles
} from "lucide-react";
import { FormEvent, useMemo, useState } from "react";
import { runExperiment, runResearch } from "./api";
import type {
  Evaluation,
  ExperimentCondition,
  ExperimentResult,
  IndependentExperimentCondition,
  Paper,
  ResearchResult,
  RunMetrics
} from "./types";

const sampleTopics = [
  "LLM-based feedback generation for programming education",
  "Retrieval augmented generation for academic literature review",
  "AI-assisted research question generation in higher education"
];

const independentConditions: IndependentExperimentCondition[] = [
  "baseline_with_retrieval",
  "multi_stage_with_retrieval",
  "multi_stage_without_retrieval",
  "multi_stage_group_selection"
];

type Page = "single" | "experiment";

export function App() {
  const [page, setPage] = useState<Page>("single");
  const [demoMode, setDemoMode] = useState(true);

  return (
    <main className="app">
      <section className="layout">
        <aside className="sidebar">
          <div className="product-title">
            <Sparkles aria-hidden="true" />
            <div>
              <h1>Academic Research Agent</h1>
              <p>Experiment workspace</p>
            </div>
          </div>

          <div className="nav-switch">
            <button className={page === "single" ? "active" : ""} type="button" onClick={() => setPage("single")}>
              Single Run
            </button>
            <button
              className={page === "experiment" ? "active" : ""}
              type="button"
              onClick={() => setPage("experiment")}
            >
              Experiment
            </button>
          </div>

          <label className="check-row">
            <input type="checkbox" checked={demoMode} onChange={(event) => setDemoMode(event.target.checked)} />
            Use demo data while backend is unavailable
          </label>
        </aside>

        <section className="content">
          {page === "single" ? <SingleRunPage demoMode={demoMode} /> : <ExperimentPage demoMode={demoMode} />}
        </section>
      </section>
    </main>
  );
}

function SingleRunPage({ demoMode }: { demoMode: boolean }) {
  const [topic, setTopic] = useState(sampleTopics[0]);
  const [maxPapers, setMaxPapers] = useState(5);
  const [condition, setCondition] = useState<ExperimentCondition>("multi_stage_with_retrieval");
  const [result, setResult] = useState<ResearchResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError("");
    setLoading(true);

    try {
      const data = await runResearch({ topic: topic.trim(), max_papers: maxPapers, condition }, demoMode);
      setResult(data);
    } catch (err) {
      setResult(null);
      setError(err instanceof Error ? err.message : "Unknown frontend integration error");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="page-stack">
      <form className="input-panel run-form" onSubmit={handleSubmit}>
        <label htmlFor="topic">Research topic</label>
        <textarea
          id="topic"
          rows={4}
          value={topic}
          onChange={(event) => setTopic(event.target.value)}
          placeholder="Enter a research topic"
        />

        <div className="form-grid">
          <div className="select-row">
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

          <div className="select-row">
            <label htmlFor="condition">Condition</label>
            <select
              id="condition"
              value={condition}
              onChange={(event) => setCondition(event.target.value as ExperimentCondition)}
            >
              {independentConditions.map((item) => (
                <option key={item} value={item}>
                  {conditionLabel(item)}
                </option>
              ))}
            </select>
          </div>
        </div>

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

      {error ? <ErrorMessage message={error} /> : null}
      {!result && !loading ? <EmptyState /> : null}
      {loading ? <LoadingState label="Generating results" /> : null}
      {result ? <ResearchResultView result={result} demoMode={demoMode} /> : null}
    </div>
  );
}

function ExperimentPage({ demoMode }: { demoMode: boolean }) {
  const [topic, setTopic] = useState("atom physics");
  const [maxPapers, setMaxPapers] = useState(5);
  const [repeats, setRepeats] = useState(1);
  const [conditions, setConditions] = useState<IndependentExperimentCondition[]>([
    "baseline_with_retrieval",
    "multi_stage_with_retrieval"
  ]);
  const [result, setResult] = useState<ExperimentResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError("");
    setLoading(true);

    try {
      const data = await runExperiment(
        {
          topic: topic.trim(),
          max_papers: maxPapers,
          repeats,
          conditions
        },
        demoMode
      );
      setResult(data);
    } catch (err) {
      setResult(null);
      setError(err instanceof Error ? err.message : "Unknown experiment error");
    } finally {
      setLoading(false);
    }
  }

  function toggleCondition(condition: IndependentExperimentCondition) {
    setConditions((current) => {
      if (current.includes(condition)) {
        return current.length === 1 ? current : current.filter((item) => item !== condition);
      }
      return [...current, condition];
    });
  }

  return (
    <div className="page-stack">
      <form className="input-panel run-form" onSubmit={handleSubmit}>
        <label htmlFor="experiment-topic">Experiment topic</label>
        <textarea
          id="experiment-topic"
          rows={4}
          value={topic}
          onChange={(event) => setTopic(event.target.value)}
          placeholder="Run one topic through selected conditions"
        />

        <div className="form-grid">
          <div className="select-row">
            <label htmlFor="experiment-papers">Papers</label>
            <input
              id="experiment-papers"
              type="number"
              min={1}
              max={10}
              value={maxPapers}
              onChange={(event) => setMaxPapers(Number(event.target.value))}
            />
          </div>

          <div className="select-row">
            <label htmlFor="repeats">Repeated runs</label>
            <input
              id="repeats"
              type="number"
              min={1}
              max={5}
              value={repeats}
              onChange={(event) => setRepeats(Number(event.target.value))}
            />
          </div>
        </div>

        <div className="condition-list">
          {independentConditions.map((item) => (
            <label className="check-row condition-option" key={item}>
              <input type="checkbox" checked={conditions.includes(item)} onChange={() => toggleCondition(item)} />
              {conditionLabel(item)}
            </label>
          ))}
        </div>

        <button type="submit" disabled={loading || topic.trim().length < 3 || conditions.length === 0}>
          {loading ? <Loader2 className="spin" aria-hidden="true" /> : <Play aria-hidden="true" />}
          Run Experiment
        </button>
      </form>

      {error ? <ErrorMessage message={error} /> : null}
      {!result && !loading ? <EmptyState /> : null}
      {loading ? <LoadingState label="Running experiment" /> : null}
      {result ? <ExperimentView result={result} /> : null}
    </div>
  );
}

function ResearchResultView({ result, demoMode }: { result: ResearchResult; demoMode: boolean }) {
  const stages = useMemo(() => buildStages(result), [result]);

  return (
    <div className="result-stack">
      <header className="result-header">
        <div>
          <p className="overline">
            {demoMode ? "Demo data" : "Backend integration"} | {conditionLabel(result.condition)}
          </p>
          <h2>{result.topic}</h2>
        </div>
      </header>

      {result.metrics ? <MetricsPanel metrics={result.metrics} /> : null}
      <PaperSection papers={result.papers} />
      {result.literature_themes?.length ? <LiteratureThemesPanel themes={result.literature_themes} /> : null}
      {result.evidence_notes?.length ? <EvidenceNotesPanel notes={result.evidence_notes} /> : null}
      {result.embedding_analysis ? <EmbeddingEvidencePanel analysis={result.embedding_analysis} /> : null}

      {stages.length ? (
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
      ) : null}

      <section className="stage-card">
        <div className="card-title">
          <FileText aria-hidden="true" />
          <h3>{result.draft_proposal ? "Revised Proposal" : "Proposal Draft"}</h3>
        </div>
        <p>{result.proposal}</p>
      </section>

      {result.draft_proposal ? (
        <section className="stage-card">
          <div className="card-title">
            <FileText aria-hidden="true" />
            <h3>Original Draft</h3>
          </div>
          <p>{result.draft_proposal}</p>
        </section>
      ) : null}

      {result.proposal_critique ? <CritiquePanel critique={result.proposal_critique} /> : null}
      {result.proposal_logic_graph ? <LogicGraphPanel graph={result.proposal_logic_graph} /> : null}
      {result.group_selection ? <GroupSelectionPanel selection={result.group_selection} /> : null}
      <EvaluationPanel evaluation={result.evaluation} fallbackReason={result.metrics?.fallback_reason} />
    </div>
  );
}

function ExperimentView({ result }: { result: ExperimentResult }) {
  const [selectedRunIndex, setSelectedRunIndex] = useState(0);
  const selectedRun = result.runs[selectedRunIndex];

  return (
    <div className="result-stack">
      <header className="result-header">
        <div>
          <p className="overline">Experiment {result.experiment_id}</p>
          <h2>{result.topic}</h2>
          <p>Retrieval query: {result.retrieval_query}</p>
        </div>
      </header>

      <SummaryTable title="Quality Comparison" rows={result.quality_comparison} />
      <SummaryTable title="Stability And Cost" rows={result.stability_cost} />
      <RunTable runs={result.runs} selectedIndex={selectedRunIndex} onSelect={setSelectedRunIndex} />
      {selectedRun ? (
        <section className="selected-run">
          <header className="result-header compact">
            <div>
              <p className="overline">
                Selected Run {selectedRun.run_index || selectedRunIndex + 1} | {conditionLabel(selectedRun.condition)}
              </p>
              <h3>{selectedRun.topic}</h3>
            </div>
          </header>
          <ResearchResultView result={selectedRun} demoMode={false} />
        </section>
      ) : null}
    </div>
  );
}

function SummaryTable({ title, rows }: { title: string; rows: Array<Record<string, string | number | null>> }) {
  const columns = Array.from(new Set(rows.flatMap((row) => Object.keys(row))));
  return (
    <section className="table-panel">
      <div className="card-title">
        <BarChart3 aria-hidden="true" />
        <h3>{title}</h3>
      </div>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              {columns.map((column) => (
                <th key={column}>{formatColumn(column)}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row, index) => (
              <tr key={`${title}-${index}`}>
                {columns.map((column) => (
                  <td key={column}>{formatCell(row[column])}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function RunTable({
  runs,
  selectedIndex,
  onSelect
}: {
  runs: ResearchResult[];
  selectedIndex: number;
  onSelect: (index: number) => void;
}) {
  return (
    <section className="table-panel">
      <div className="card-title">
        <ClipboardList aria-hidden="true" />
        <h3>Experiment Runs</h3>
      </div>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Run</th>
              <th>Condition</th>
              <th>Average Score</th>
              <th>Latency</th>
              <th>Tokens</th>
              <th>Papers</th>
              <th>Completeness</th>
              <th>Result</th>
            </tr>
          </thead>
          <tbody>
            {runs.map((run, index) => (
              <tr className={selectedIndex === index ? "selected-row" : ""} key={`${run.condition}-${run.run_index || index}`}>
                <td>{run.run_index || index + 1}</td>
                <td>{conditionLabel(run.condition)}</td>
                <td>{averageScore(run.evaluation).toFixed(2)}</td>
                <td>{run.metrics ? `${run.metrics.latency_ms} ms` : "N/A"}</td>
                <td>{run.metrics?.total_tokens || "N/A"}</td>
                <td>{run.papers.length}</td>
                <td>
                  {run.metrics ? `${Math.round(run.metrics.required_field_completeness * 100)}%` : "N/A"}
                </td>
                <td>
                  <button className="table-action" type="button" onClick={() => onSelect(index)}>
                    View
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function MetricsPanel({ metrics }: { metrics: RunMetrics }) {
  const rows = [
    ["Condition", conditionLabel(metrics.condition)],
    ["Retrieval query", metrics.retrieval_query || "Same as topic"],
    ["Retrieval snapshot", metrics.retrieval_snapshot_id || "Not saved"],
    ["Model", metrics.model],
    ["Temperature", metrics.temperature.toString()],
    ["Latency", `${metrics.latency_ms} ms`],
    ["Total tokens", metrics.total_tokens ? metrics.total_tokens.toString() : "Unavailable"],
    ["Field completeness", `${Math.round(metrics.required_field_completeness * 100)}%`],
    ["JSON valid", metrics.json_valid ? "Yes" : "No"],
    ["Fallback reason", metrics.fallback_reason || "None"]
  ];

  return (
    <section className="metrics-panel">
      <div className="card-title">
        <BarChart3 aria-hidden="true" />
        <h3>Experiment Record</h3>
      </div>
      <div className="metrics-grid">
        {rows.map(([label, value]) => (
          <div key={label}>
            <span>{label}</span>
            <strong>{value}</strong>
          </div>
        ))}
      </div>
    </section>
  );
}

function CritiquePanel({ critique }: { critique: Record<string, unknown> }) {
  return (
    <section className="group-panel">
      <div className="card-title">
        <ClipboardList aria-hidden="true" />
        <h3>Stage-Aware Draft Critique</h3>
      </div>
      <div className="metrics-grid">
        {Object.entries(critique).map(([key, value]) => (
          <CritiqueItem key={key} label={key} value={value} />
        ))}
      </div>
    </section>
  );
}

function CritiqueItem({ label, value }: { label: string; value: unknown }) {
  const displayLabel = label.replaceAll("_", " ");
  if (Array.isArray(value)) {
    return (
      <div className="critique-block">
        <span>{displayLabel}</span>
        <div className="critique-subgrid">
          {value.map((item, index) => (
            <CritiqueItem key={`${label}-${index}`} label={`${index + 1}`} value={item} />
          ))}
        </div>
      </div>
    );
  }

  if (value && typeof value === "object" && !Array.isArray(value)) {
    return (
      <div className="critique-block">
        <span>{displayLabel}</span>
        <div className="critique-subgrid">
          {Object.entries(value as Record<string, unknown>).map(([childKey, childValue]) => (
            <CritiqueItem key={childKey} label={childKey} value={childValue} />
          ))}
        </div>
      </div>
    );
  }

  return (
    <div>
      <span>{displayLabel}</span>
      <strong>{String(value ?? "Not recorded")}</strong>
    </div>
  );
}

function GroupSelectionPanel({ selection }: { selection: NonNullable<ResearchResult["group_selection"]> }) {
  const stages = [
    ["Gap Candidates", selection.gap],
    ["Hypothesis Candidates", selection.hypothesis],
    ["Methodology Candidates", selection.methodology]
  ] as const;

  return (
    <section className="group-panel">
      <div className="card-title">
        <BarChart3 aria-hidden="true" />
        <h3>Group-Based Candidate Selection</h3>
      </div>
      <div className="candidate-stage-grid">
        {stages.map(([title, stage]) =>
          stage ? (
            <article className="candidate-stage" key={title}>
              <h4>{title}</h4>
              <p className="selected-text">{stage.selected_text || "No selected candidate recorded."}</p>
              {stage.candidates?.length ? (
                <div className="candidate-list">
                  {stage.candidates.map((candidate, index) => (
                    <div
                      className={candidate.id === stage.selected_id ? "candidate selected-candidate" : "candidate"}
                      key={candidate.id || `${title}-${index}`}
                    >
                      <strong>
                        {candidate.id || `candidate_${index + 1}`}
                        {candidate.total !== null && candidate.total !== undefined ? ` | total ${candidate.total}` : ""}
                      </strong>
                      <p>{candidate.text || ""}</p>
                      {candidate.reason ? <span>{candidate.reason}</span> : null}
                    </div>
                  ))}
                </div>
              ) : null}
            </article>
          ) : null
        )}
      </div>
    </section>
  );
}

function EmptyState() {
  return (
    <div className="empty-state">
      <ClipboardList aria-hidden="true" />
      <h2>Ready for testing</h2>
      <p>Enter a topic, run demo data, or connect to the backend endpoint.</p>
    </div>
  );
}

function LoadingState({ label }: { label: string }) {
  return (
    <div className="empty-state">
      <Loader2 className="spin" aria-hidden="true" />
      <h2>{label}</h2>
      <p>The backend runs selected conditions sequentially to reduce rate-limit risk.</p>
    </div>
  );
}

function ErrorMessage({ message }: { message: string }) {
  return (
    <div className="message error">
      <AlertCircle aria-hidden="true" />
      <span>{message}</span>
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

function LiteratureThemesPanel({ themes }: { themes: NonNullable<ResearchResult["literature_themes"]> }) {
  return (
    <section className="panel-section">
      <div className="card-title">
        <BookOpen aria-hidden="true" />
        <h3>Literature Themes</h3>
      </div>
      <div className="theme-grid">
        {themes.map((theme, index) => (
          <article className="theme-card" key={`${theme.theme}-${index}`}>
            <h4>{theme.theme}</h4>
            <p className="paper-meta">Papers: {theme.paper_indices.join(", ")}</p>
            <p>{theme.summary}</p>
            {theme.limitations ? <p><strong>Limitations:</strong> {theme.limitations}</p> : null}
            {theme.gap_relevance ? <p><strong>Gap relevance:</strong> {theme.gap_relevance}</p> : null}
          </article>
        ))}
      </div>
    </section>
  );
}

function EvidenceNotesPanel({ notes }: { notes: NonNullable<ResearchResult["evidence_notes"]> }) {
  return (
    <section className="panel-section">
      <div className="card-title">
        <BookOpen aria-hidden="true" />
        <h3>Evidence Notes</h3>
      </div>
      <div className="theme-grid">
        {notes.map((note) => (
          <article className="theme-card" key={note.paper_id}>
            <h4>{note.paper_id}: {note.title || "Untitled paper"}</h4>
            {note.problem ? <p><strong>Problem:</strong> {note.problem}</p> : null}
            {note.method ? <p><strong>Method:</strong> {note.method}</p> : null}
            {note.evidence ? <p><strong>Evidence:</strong> {note.evidence}</p> : null}
            {note.limitation ? <p><strong>Limitation:</strong> {note.limitation}</p> : null}
            {note.gap_relevance ? <p><strong>Gap relevance:</strong> {note.gap_relevance}</p> : null}
          </article>
        ))}
      </div>
    </section>
  );
}

function EmbeddingEvidencePanel({ analysis }: { analysis: Record<string, unknown> }) {
  return (
    <section className="group-panel">
      <div className="card-title">
        <Search aria-hidden="true" />
        <h3>Embedding Evidence</h3>
      </div>
      <div className="metrics-grid">
        {Object.entries(analysis).map(([key, value]) => (
          <CritiqueItem key={key} label={key} value={value} />
        ))}
      </div>
    </section>
  );
}

function LogicGraphPanel({ graph }: { graph: NonNullable<ResearchResult["proposal_logic_graph"]> }) {
  return (
    <section className="group-panel">
      <div className="card-title">
        <BrainCircuit aria-hidden="true" />
        <h3>Proposal Logic Graph</h3>
      </div>
      <div className="metrics-grid">
        {Object.entries(graph).map(([key, value]) => (
          <CritiqueItem key={key} label={key} value={value} />
        ))}
      </div>
    </section>
  );
}

function EvaluationPanel({ evaluation, fallbackReason }: { evaluation: Evaluation; fallbackReason?: string | null }) {
  const scores = [
    ["Clarity", evaluation.clarity],
    ["Logic", evaluation.logic],
    ["Novelty", evaluation.novelty],
    ["Feasibility", evaluation.feasibility],
    ["Literature Alignment", evaluation.literature_alignment],
    ["PhD-Level Quality", evaluation.phd_level_quality],
    ["Presentation", evaluation.presentation]
  ].filter(([, score]) => typeof score === "number") as Array<[string, number]>;

  return (
    <section className="evaluation-panel">
      <div className="card-title">
        <BarChart3 aria-hidden="true" />
        <h3>Self-Evaluation</h3>
      </div>

      {typeof evaluation.weighted_total === "number" ? (
        <div className="final-score-card">
          <span>Final Proposal Score</span>
          <strong>{evaluation.weighted_total.toFixed(1)}/100</strong>
        </div>
      ) : (
        <div className="final-score-card unavailable">
          <span>Final Proposal Score</span>
          <strong>Not available</strong>
          {fallbackReason ? <p>LLM evaluation fell back: {fallbackReason}</p> : null}
        </div>
      )}

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
      {evaluation.rubric_scores?.length ? <RubricScoreTable scores={evaluation.rubric_scores} /> : null}
    </section>
  );
}

function RubricScoreTable({ scores }: { scores: NonNullable<Evaluation["rubric_scores"]> }) {
  return (
    <div className="table-wrap rubric-table">
      <table>
        <thead>
          <tr>
            <th>Criterion</th>
            <th>Weight</th>
            <th>Score</th>
            <th>Level</th>
            <th>Justification</th>
          </tr>
        </thead>
        <tbody>
          {scores.map((score) => (
            <tr key={score.criterion}>
              <td>{score.criterion}</td>
              <td>{score.weight}</td>
              <td>{score.score}</td>
              <td>{score.level}</td>
              <td>{score.justification}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function buildStages(result: ResearchResult) {
  return [
    { title: "Literature Summary", value: result.summary, icon: BookOpen },
    { title: "Scholarly Positioning", value: result.scholarly_positioning, icon: BookOpen },
    { title: "Domain Router", value: result.domain_router, icon: ClipboardList },
    { title: "Research Scope", value: result.research_scope, icon: ClipboardList },
    { title: "Concept Definition", value: result.concept_definition, icon: BrainCircuit },
    { title: "Technical Feasibility", value: result.technical_feasibility, icon: ClipboardList },
    { title: "Research Gap", value: result.gap, icon: Search },
    { title: "Research Question", value: result.question, icon: BrainCircuit },
    { title: "Theoretical Mechanism", value: result.theoretical_mechanism, icon: BrainCircuit },
    { title: "Hypothesis", value: result.hypothesis, icon: Lightbulb },
    { title: "Operationalization & Causal Check", value: result.operationalization_causal_check, icon: ClipboardList },
    { title: "Methodology", value: result.methodology, icon: FlaskConical },
    { title: "Novelty & Contribution", value: result.novelty_contribution, icon: Lightbulb },
    { title: "Failure Analysis", value: result.failure_analysis, icon: AlertCircle }
  ].filter((stage): stage is { title: string; value: string; icon: typeof BookOpen } => Boolean(stage.value));
}

function conditionLabel(condition?: ExperimentCondition): string {
  if (condition === "baseline_with_retrieval") return "C0 raw single prompt";
  if (condition === "multi_stage_with_retrieval") return "C1 multi-stage with retrieval";
  if (condition === "multi_stage_without_retrieval") return "C2 multi-stage without retrieval";
  if (condition === "multi_stage_group_selection") return "C3 group-based selection";
  return "Legacy C0 vs C1 main comparison";
}

function formatColumn(column: string): string {
  if (column === "metric") return "Metric";
  if (column === "multi_stage_minus_baseline") return "C1 - C0";
  if (column === "group_selection_minus_multi_stage") return "C3 - C1";
  return conditionLabel(column as ExperimentCondition).replace("Legacy ", "");
}

function formatCell(value: string | number | null | undefined): string {
  if (value === null || value === undefined) return "N/A";
  if (typeof value === "number") return Number.isInteger(value) ? value.toString() : value.toFixed(3);
  return value.replaceAll("_", " ");
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
