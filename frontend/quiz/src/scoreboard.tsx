import { useEffect, useMemo, useRef, useState } from "react";
import { formatAccuracy, type LeaderboardEntry } from "./leaderboard";
import { createShareCardBlob, downloadShareCard } from "./shareCard";

export function LiveScoreboard({
  entries,
  answered,
  settleMinimum,
  onSettle
}: {
  entries: LeaderboardEntry[];
  answered: number;
  settleMinimum: number;
  onSettle: () => void;
}) {
  const unlocked = answered >= settleMinimum;
  const usesLiveAi = entries.some((entry) => entry.source === "answered_questions");
  return (
    <aside className="scoreboard" aria-label="Live leaderboard">
      <div className="scoreboard-head">
        <div>
          <p className="scoreboard-kicker">Live ranking</p>
          <h2>Human vs AI</h2>
        </div>
        <span className="scoreboard-source">{usesLiveAi ? "Live" : "500Q"}</span>
      </div>
      <ol className="scoreboard-list" aria-live="polite">
        {entries.map((entry) => (
          <li key={entry.id} className={entry.kind === "human" ? "is-human" : undefined}>
            <span className="scoreboard-rank">{entry.rank === null ? "-" : `#${entry.rank}`}</span>
            <span className="scoreboard-player" title={entry.label}>
              <strong>{entry.label}</strong>
              <small>
                {entry.correct}/{entry.total} · {formatAccuracy(entry.accuracy)}
              </small>
            </span>
          </li>
        ))}
      </ol>
      <div className="settle-control">
        <button
          type="button"
          className="settle-button"
          disabled={!unlocked}
          onClick={onSettle}
        >
          {unlocked ? "Settle score" : `Settle at ${settleMinimum}`}
        </button>
        <p>{answered}/{settleMinimum} answers</p>
      </div>
    </aside>
  );
}

export function SettlementOverlay({
  entries,
  correct,
  total,
  complete,
  onClose
}: {
  entries: LeaderboardEntry[];
  correct: number;
  total: number;
  complete: boolean;
  onClose: () => void;
}) {
  const [blob, setBlob] = useState<Blob | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [shareStatus, setShareStatus] = useState("");
  const human = entries.find((entry) => entry.kind === "human");
  const leaders = useMemo(
    () => entries.filter((entry) => entry.kind === "ai").slice(0, 4),
    [entries]
  );
  const closeButton = useRef<HTMLButtonElement | null>(null);

  useEffect(() => {
    closeButton.current?.focus();
  }, []);

  useEffect(() => {
    let disposed = false;
    let url: string | null = null;
    void createShareCardBlob({
      correct,
      total,
      rank: human?.rank ?? null,
      participantCount: entries.length,
      leaders
    }).then((nextBlob) => {
      if (disposed) return;
      url = URL.createObjectURL(nextBlob);
      setBlob(nextBlob);
      setPreviewUrl(url);
    }).catch(() => {
      if (!disposed) setShareStatus("Could not generate the score image.");
    });
    return () => {
      disposed = true;
      if (url) URL.revokeObjectURL(url);
    };
  }, [correct, entries, human?.rank, leaders, total]);

  async function shareScore() {
    if (!blob) return;
    const file = new File([blob], "architectureiq-score.png", { type: "image/png" });
    const shareData: ShareData = {
      title: "My ArchitectureIQ score",
      text: `I scored ${correct}/${total} on ArchitectureIQ.`,
      files: [file]
    };
    if (navigator.share && (!navigator.canShare || navigator.canShare(shareData))) {
      try {
        await navigator.share(shareData);
        setShareStatus("Shared.");
        return;
      } catch (error) {
        if (error instanceof DOMException && error.name === "AbortError") return;
      }
    }
    downloadShareCard(blob);
    setShareStatus("Sharing is unavailable here, so the image was downloaded.");
  }

  const accuracy = total ? correct / total : null;
  return (
    <div className="completion-overlay" role="dialog" aria-modal="true" aria-label="Score settlement">
      <div className="completion-card">
        <section className="settlement-summary">
          <p className="scoreboard-kicker">{complete ? "Pack complete" : "Score settled"}</p>
          <h2>{complete ? "All done" : "Your result"}</h2>
          <div className="settlement-score">
            <strong>{correct}</strong><span>/{total}</span>
          </div>
          <p className="settlement-accuracy">{formatAccuracy(accuracy)} accuracy</p>
          <div className="settlement-rank">
            <span>Live rank</span>
            <strong>{human?.rank == null ? "-" : `#${human.rank}`}</strong>
            <small>of {entries.length}</small>
          </div>
          <div className="settlement-comparison">
            <h3>Top AI comparison</h3>
            {leaders.slice(0, 3).map((entry) => (
              <div key={entry.id}>
                <span>{entry.label}</span>
                <strong>{formatAccuracy(entry.accuracy)}</strong>
              </div>
            ))}
          </div>
          <button ref={closeButton} type="button" className="completion-btn" onClick={onClose}>
            Back to questions
          </button>
        </section>
        <section className="share-panel" aria-label="Share score">
          <div className="share-panel-head">
            <div>
              <p className="scoreboard-kicker">Share</p>
              <h3>Your score card</h3>
            </div>
            <span>PNG</span>
          </div>
          <div className="share-preview">
            {previewUrl ? (
              <img src={previewUrl} alt="Preview of downloadable ArchitectureIQ score card" />
            ) : (
              <span>Generating image...</span>
            )}
          </div>
          <div className="share-actions">
            <button type="button" className="completion-btn primary" disabled={!blob} onClick={shareScore}>
              Share score
            </button>
            <button type="button" className="completion-btn" disabled={!blob} onClick={() => blob && downloadShareCard(blob)}>
              Download image
            </button>
          </div>
          <p className="share-status" aria-live="polite">{shareStatus}</p>
        </section>
      </div>
    </div>
  );
}
