export type Field = { label: string; value: string };

export type Point = { x: number; y: number };

export type QuestionSummary = {
  id: string;
  type?: string;
  datasetId?: string;
  family?: string;
  budget?: number;
  choices?: number;
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

export type BakedQuestion = {
  id: string;
  title: string;
  family: string;
  datasetId: string;
  type: string;
  profile?: string;
  budget: Record<string, unknown> | number;
  metric?: string;
  evaluation?: Record<string, unknown>;
  invariantAxes?: string[];
  varyingAxes?: string[];
  numChoices?: number;
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
        train?: Point[];
        test?: Point[];
        matrix?: number[][];
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
  questions: QuestionSummary[];
  byId: Record<string, BakedQuestion>;
};

export type Stage = "observe" | "compare" | "reveal";
