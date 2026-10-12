import type { BakeFile } from "./types";

export type QuizResult = { correct: boolean; picked: string };

export type LeaderboardEntry = {
  id: string;
  label: string;
  kind: "ai" | "human";
  correct: number;
  total: number;
  accuracy: number | null;
  rank: number | null;
  source: "question_pack" | "human";
};

type Score = { correct: number; total: number };

const MODEL_LABELS: Record<string, string> = {
  "gpt-5.6-sol": "GPT-5.6 Sol",
  "gpt-5.6-luna": "GPT-5.6 Luna",
  "claude-opus-5": "Claude Opus 5",
  "claude-sonnet-5": "Claude Sonnet 5",
  "deepseek-v4-flash-0731": "DeepSeek V4 Flash",
  "deepseek-v4-pro-0813": "DeepSeek V4 Pro",
  "deepseek_deepseek-r1": "DeepSeek R1",
  "gemini-2.5-pro": "Gemini 2.5 Pro",
  "gemini-3.1-pro-preview": "Gemini 3.1 Pro",
  "gemini-3.1-pro-preview-high": "Gemini 3.1 Pro",
  "glm-5.2-low": "GLM-5.2 Low",
  "gpt-4o": "GPT-4o",
  "minicpm5-1b": "MiniCPM 5 1B",
  "qwen3.5-0.8b": "Qwen 3.5 0.8B"
};

export function formatModelLabel(model: string): string {
  return MODEL_LABELS[model] ?? model;
}

function compareEntries(a: LeaderboardEntry, b: LeaderboardEntry): number {
  if (a.accuracy === null) return b.accuracy === null ? 0 : 1;
  if (b.accuracy === null) return -1;
  if (a.accuracy !== b.accuracy) return b.accuracy - a.accuracy;
  if (a.correct !== b.correct) return b.correct - a.correct;
  return a.label.localeCompare(b.label);
}

export function buildLeaderboard(
  bake: BakeFile,
  results: Record<string, QuizResult>,
  humanLabel = "You"
): LeaderboardEntry[] {
  const modelScores = new Map<string, Score>();
  const seen = new Set<string>();

  for (const question of Object.values(bake.byId)) {
    for (const entry of question.llmCot?.entries ?? []) {
      if (typeof entry.correct === "boolean" && !modelScores.has(entry.model)) {
        modelScores.set(entry.model, { correct: 0, total: 0 });
      }
    }
  }

  for (const questionId of Object.keys(results)) {
    const question = bake.byId[questionId];
    for (const entry of question?.llmCot?.entries ?? []) {
      if (typeof entry.correct !== "boolean") continue;
      const key = `${questionId}\u0000${entry.model}`;
      if (seen.has(key)) continue;
      seen.add(key);
      const score = modelScores.get(entry.model) ?? { correct: 0, total: 0 };
      score.total += 1;
      if (entry.correct) score.correct += 1;
      modelScores.set(entry.model, score);
    }
  }

  const entries: LeaderboardEntry[] = Array.from(modelScores, ([model, score]) => ({
    id: `ai:${model}`,
    label: formatModelLabel(model),
    kind: "ai" as const,
    ...score,
    accuracy: score.total ? score.correct / score.total : null,
    rank: null,
    source: "question_pack" as const
  }));

  const humanScore = Object.values(results).reduce<Score>(
    (score, result) => ({
      correct: score.correct + (result.correct ? 1 : 0),
      total: score.total + 1
    }),
    { correct: 0, total: 0 }
  );
  entries.push({
    id: "human",
    label: humanLabel,
    kind: "human",
    ...humanScore,
    accuracy: humanScore.total ? humanScore.correct / humanScore.total : null,
    rank: null,
    source: "human"
  });
  entries.sort(compareEntries);

  let previousAccuracy: number | null | undefined;
  let previousRank = 0;
  return entries.map((entry, index) => {
    const tied = index > 0 && entry.accuracy === previousAccuracy;
    const rank = entry.accuracy === null ? null : tied ? previousRank : index + 1;
    previousAccuracy = entry.accuracy;
    if (rank !== null) previousRank = rank;
    return { ...entry, rank };
  });
}

export function formatAccuracy(accuracy: number | null): string {
  return accuracy === null ? "-" : `${Math.round(accuracy * 100)}%`;
}
