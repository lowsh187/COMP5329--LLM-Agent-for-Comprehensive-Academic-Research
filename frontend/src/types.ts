export type Paper = {
  title: string;
  authors: string[];
  abstract: string;
  published?: string | null;
  url: string;
};

export type Evaluation = {
  clarity: number;
  logic: number;
  novelty: number;
  feasibility: number;
  literature_alignment: number;
  comments: string;
};

export type ResearchResult = {
  topic: string;
  papers: Paper[];
  summary: string;
  gap: string;
  question: string;
  hypothesis: string;
  methodology: string;
  proposal: string;
  evaluation: Evaluation;
  baseline?: {
    proposal?: string;
    evaluation?: Partial<Evaluation>;
  };
};

export type ResearchRequest = {
  topic: string;
  max_papers: number;
};
