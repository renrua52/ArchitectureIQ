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
  source: "answered_questions" | "benchmark_500" | "human";
};

type Score = { correct: number; total: number };

const BENCHMARK_500: Array<{ label: string; accuracy: number }> = [
  { label: "GPT-5.6 Sol", accuracy: 0.764 },
  { label: "Claude Opus 5", accuracy: 0.76 },
  { label: "Claude Sonnet 5", accuracy: 0.678 },
  { label: "Gemini 3.1 Pro", accuracy: 0.666 },
  { label: "GPT-5.6 Luna", accuracy: 0.654 },
  { label: "DeepSeek R1", accuracy: 0.59 },
  { label: "Gemini 2.5 Pro", accuracy: 0.558 },
  { label: "DeepSeek R1 32B", accuracy: 0.46 },
  { label: "Llama 3.1 70B", accuracy: 0.42 }
];

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

  const entries: LeaderboardEntry[] = modelScores.size
    ? Array.from(modelScores, ([model, score]) => ({
        id: `ai:${model}`,
        label: model,
        kind: "ai" as const,
        ...score,
        accuracy: score.total ? score.correct / score.total : null,
        rank: null,
        source: "answered_questions" as const
      }))
    : BENCHMARK_500.map(({ label, accuracy }) => ({
        id: `ai:${label}`,
        label,
        kind: "ai" as const,
        correct: Math.round(accuracy * 500),
        total: 500,
        accuracy,
        rank: null,
        source: "benchmark_500" as const
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
