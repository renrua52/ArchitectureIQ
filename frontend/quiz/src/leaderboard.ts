import type { BakeFile } from "./types";

export type LeaderboardEntry = {
  id: string;
  label: string;
  kind: "ai" | "human";
  correct: number;
  total: number;
  accuracy: number | null;
  rank: number | null;
};

type RunningScore = { correct: number; total: number };

function compareEntries(a: LeaderboardEntry, b: LeaderboardEntry): number {
  if (a.accuracy === null) return b.accuracy === null ? 0 : 1;
  if (b.accuracy === null) return -1;
  if (a.accuracy !== b.accuracy) return b.accuracy - a.accuracy;
  if (a.correct !== b.correct) return b.correct - a.correct;
  if (a.total !== b.total) return b.total - a.total;
  return a.label.localeCompare(b.label);
}

/** Build one ranking from the active pack's baked model results and live human score. */
export function buildLeaderboard(
  bake: BakeFile,
  human: RunningScore,
  humanLabel = "You"
): LeaderboardEntry[] {
  const modelScores = new Map<string, RunningScore>();
  const seen = new Set<string>();

  for (const summary of bake.questions) {
    const question = bake.byId[summary.id];
    for (const entry of question?.llmCot?.entries ?? []) {
      if (typeof entry.correct !== "boolean") continue;
      const key = `${question.id}\u0000${entry.model}`;
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
    label: model,
    kind: "ai",
    ...score,
    accuracy: score.total ? score.correct / score.total : null,
    rank: null
  }));

  entries.push({
    id: "human",
    label: humanLabel,
    kind: "human",
    ...human,
    accuracy: human.total ? human.correct / human.total : null,
    rank: null
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
