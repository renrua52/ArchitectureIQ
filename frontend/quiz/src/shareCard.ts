import type { LeaderboardEntry } from "./leaderboard";
import { formatAccuracy } from "./leaderboard";

export type ShareCardData = {
  correct: number;
  total: number;
  rank: number | null;
  participantCount: number;
  leaders: LeaderboardEntry[];
};

export async function createShareCardBlob(data: ShareCardData): Promise<Blob> {
  const canvas = document.createElement("canvas");
  canvas.width = 1200;
  canvas.height = 630;
  const context = canvas.getContext("2d");
  if (!context) throw new Error("Canvas is unavailable in this browser.");

  context.fillStyle = "#141817";
  context.fillRect(0, 0, 1200, 630);
  context.fillStyle = "#4ee3a4";
  context.fillRect(0, 0, 18, 630);
  context.fillStyle = "#f4f7f5";
  context.font = "700 34px system-ui, sans-serif";
  context.fillText("ArchitectureIQ", 78, 78);
  context.fillStyle = "#95a09b";
  context.font = "500 20px system-ui, sans-serif";
  context.fillText("Deep learning modeling intuition benchmark", 78, 112);

  context.fillStyle = "#202724";
  context.beginPath();
  context.roundRect(72, 160, 610, 388, 18);
  context.fill();
  context.fillStyle = "#a7b1ac";
  context.font = "600 19px system-ui, sans-serif";
  context.fillText("Human result", 112, 214);
  context.fillStyle = "#f4f7f5";
  context.font = "750 112px system-ui, sans-serif";
  context.fillText(`${data.correct}/${data.total}`, 108, 342, 520);
  const accuracy = data.total ? data.correct / data.total : null;
  context.fillStyle = "#4ee3a4";
  context.font = "700 42px system-ui, sans-serif";
  context.fillText(`${formatAccuracy(accuracy)} accuracy`, 112, 404);
  context.fillStyle = "#a7b1ac";
  context.font = "600 25px system-ui, sans-serif";
  context.fillText(
    data.rank === null
      ? "Unranked"
      : `Rank #${data.rank} of ${data.participantCount}`,
    112,
    462
  );

  context.fillStyle = "#f4f7f5";
  context.font = "700 25px system-ui, sans-serif";
  context.fillText("Leaderboard", 750, 192);
  context.fillStyle = "#95a09b";
  context.font = "500 17px system-ui, sans-serif";
  context.fillText("Top AI benchmark results", 750, 222);
  data.leaders.slice(0, 4).forEach((entry, index) => {
    const y = 276 + index * 66;
    context.fillStyle = index === 0 ? "#4ee3a4" : "#8c9792";
    context.font = "700 20px system-ui, sans-serif";
    context.fillText(`#${entry.rank ?? index + 1}`, 750, y);
    context.fillStyle = "#f4f7f5";
    context.font = "600 19px system-ui, sans-serif";
    context.fillText(entry.label, 810, y, 250);
    context.fillStyle = "#c9d0cd";
    context.textAlign = "right";
    context.fillText(formatAccuracy(entry.accuracy), 1110, y);
    context.textAlign = "left";
  });
  context.fillStyle = "#6f7a75";
  context.font = "600 18px system-ui, sans-serif";
  context.fillText("architecture-iq.com", 750, 540);

  return new Promise((resolve, reject) => {
    canvas.toBlob(
      (blob) => (blob ? resolve(blob) : reject(new Error("Could not create score image."))),
      "image/png"
    );
  });
}

export function downloadShareCard(blob: Blob): void {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = "architectureiq-score.png";
  anchor.click();
  window.setTimeout(() => URL.revokeObjectURL(url), 0);
}
