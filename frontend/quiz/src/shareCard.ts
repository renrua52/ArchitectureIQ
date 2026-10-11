import type { LeaderboardEntry } from "./leaderboard";
import { formatAccuracy } from "./leaderboard";

export type ShareCardData = {
  username: string;
  correct: number;
  total: number;
  rank: number | null;
  participantCount: number;
  leaders: LeaderboardEntry[];
};

const WIDTH = 1200;
const HEIGHT = 630;

function drawText(
  context: CanvasRenderingContext2D,
  text: string,
  x: number,
  y: number,
  maxWidth?: number
) {
  if (maxWidth === undefined) context.fillText(text, x, y);
  else context.fillText(text, x, y, maxWidth);
}

export async function createShareCardBlob(data: ShareCardData): Promise<Blob> {
  const canvas = document.createElement("canvas");
  canvas.width = WIDTH;
  canvas.height = HEIGHT;
  const context = canvas.getContext("2d");
  if (!context) throw new Error("Canvas is unavailable in this browser.");

  context.fillStyle = "#141817";
  context.fillRect(0, 0, WIDTH, HEIGHT);
  context.fillStyle = "#4ee3a4";
  context.fillRect(0, 0, 18, HEIGHT);

  context.fillStyle = "#f4f7f5";
  context.font = "700 34px system-ui, -apple-system, sans-serif";
  drawText(context, "ArchitectureIQ", 78, 78);
  context.fillStyle = "#95a09b";
  context.font = "500 20px system-ui, -apple-system, sans-serif";
  drawText(context, "Deep learning modeling intuition benchmark", 78, 112);

  context.fillStyle = "#202724";
  context.beginPath();
  context.roundRect(72, 160, 610, 388, 18);
  context.fill();

  context.fillStyle = "#a7b1ac";
  context.font = "600 19px system-ui, -apple-system, sans-serif";
  drawText(context, data.username, 112, 214, 520);
  context.fillStyle = "#f4f7f5";
  context.font = "750 112px system-ui, -apple-system, sans-serif";
  drawText(context, `${data.correct}/${data.total}`, 108, 342);
  const accuracy = data.total ? data.correct / data.total : null;
  context.fillStyle = "#4ee3a4";
  context.font = "700 42px system-ui, -apple-system, sans-serif";
  drawText(context, `${formatAccuracy(accuracy)} accuracy`, 112, 404);
  context.fillStyle = "#a7b1ac";
  context.font = "600 25px system-ui, -apple-system, sans-serif";
  const rank = data.rank === null ? "Unranked" : `Rank #${data.rank}`;
  drawText(context, `${rank} of ${data.participantCount}`, 112, 462);

  context.fillStyle = "#f4f7f5";
  context.font = "700 25px system-ui, -apple-system, sans-serif";
  drawText(context, "Leaderboard", 750, 192);
  context.fillStyle = "#95a09b";
  context.font = "500 17px system-ui, -apple-system, sans-serif";
  drawText(context, "Top AI results in this question pack", 750, 222);

  data.leaders.slice(0, 4).forEach((entry, index) => {
    const y = 276 + index * 66;
    context.fillStyle = index === 0 ? "#4ee3a4" : "#8c9792";
    context.font = "700 20px system-ui, -apple-system, sans-serif";
    drawText(context, `#${entry.rank ?? index + 1}`, 750, y);
    context.fillStyle = "#f4f7f5";
    context.font = "600 19px system-ui, -apple-system, sans-serif";
    drawText(context, entry.label, 810, y, 260);
    context.fillStyle = "#c9d0cd";
    context.textAlign = "right";
    drawText(context, formatAccuracy(entry.accuracy), 1110, y);
    context.textAlign = "left";
  });

  context.fillStyle = "#6f7a75";
  context.font = "600 18px system-ui, -apple-system, sans-serif";
  drawText(context, "architecture-iq.com", 750, 540);

  return new Promise((resolve, reject) => {
    canvas.toBlob(
      (blob) => (blob ? resolve(blob) : reject(new Error("Could not create score image."))),
      "image/png"
    );
  });
}

export function scoreCardFileName(username: string): string {
  const safeName = username.trim().replace(/[^a-zA-Z0-9_-]+/g, "-").replace(/^-|-$/g, "");
  return `architectureiq-${safeName || "score"}.png`;
}

export function downloadShareCard(blob: Blob, username: string): void {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = scoreCardFileName(username);
  anchor.click();
  window.setTimeout(() => URL.revokeObjectURL(url), 0);
}
