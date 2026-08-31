// `group` and `order` carry the canonical card layout the exporter decided:
// which section of the setup the field belongs to, and where it sits. Older
// bakes omit both, and then the cards fall back to the order they arrive in.
export type Field = { label: string; value: string; group?: string; order?: number };

export type Point = { x: number; y: number };

export type PlotPoint = Point & { label?: number };

export type QuestionSummary = {
  id: string;
  type?: string;
  datasetId?: string;
  family?: string;
  budget?: number;
  choices?: number;
  metric?: string;
  profile?: string;
  profileHash?: string;
  track?: string;
  order?: number;
};

export type Choice = {
  letter: string;
  candidateId: string;
  color: string;
  variant: Field[];
  modelLines: string[];
  optimizerLines: string[];
  lossLines: string[];
  files: Record<string, unknown>;
};

export type LlmCotEntry = {
  model: string;
  correct: boolean;
  parsedLetter?: string | null;
  source?: string;
  text: string;
};

export type LlmCot = {
  /** True when at least one model has extractable CoT text. */
  available: boolean;
  reason?: "no_correct" | "no_cot" | string;
  defaultModel?: string | null;
  entries?: LlmCotEntry[];
};

export type BakedQuestion = {
  id: string;
  title: string;
  family: string;
  datasetId: string;
  type: string;
  profile?: string;
  profileHash?: string;
  track?: string;
  sourceRun?: string | null;
  provenance?: Record<string, unknown>;
  budget: Record<string, unknown> | number;
  metric?: string;
  evaluation?: Record<string, unknown>;
  invariantAxes?: string[];
  varyingAxes?: string[];
  numChoices?: number;
  llmDifficulty?: string;
  llmConsensusAcc?: number;
  /** Present after answer reveal: per-model CoT traces for the dropdown. */
  llmCot?: LlmCot;
  detail: {
    prompt: string;
    shared: Field[];
    dataset: {
      family: string;
      datasetId: string;
      selectionMetric?: string;
      params?: Record<string, unknown>;
      plot?: {
        kind: string;
        train?: PlotPoint[];
        test?: PlotPoint[];
        matrix?: number[][];
        xEdges?: number[];
        yEdges?: number[];
        probability?: number[][];
        featurePair?: [number, number];
        selectionNote?: string;
        xLabel?: string;
        yLabel?: string;
        legend?: string;
        min?: number;
        max?: number;
      };
      example?: {
        input: number | number[];
        output: number | number[];
      };
      files?: Record<string, unknown>;
      tensorShapes?: Record<string, unknown>;
    };
    choices: Choice[];
  };
  reveal: {
    correctLetter: string;
    ranked: Array<{
      letter: string;
      candidateId: string;
      metric: string;
      mean: number | null;
      std: number | null;
      label: string;
    }>;
    curves: Array<{
      letter: string;
      samples: number[];
      mean: number[];
      std: number[];
    }>;
    files?: Record<string, Record<string, unknown>>;
  };
};

export type BakeFile = {
  schema_version: number;
  ordered?: boolean;
  collection?: Record<string, unknown> | null;
  questions: QuestionSummary[];
  byId: Record<string, BakedQuestion>;
  /** Present when deployed as index.json + by-id/*.json (Cloudflare 25 MiB limit). */
  split?: boolean;
};

/** The dataset and the choices share one screen, so there are two stages. */
export type Stage = "study" | "reveal";

/** Yes/No vote for "Is this a good problem?" */
export type ProblemVote = "yes" | "no";
