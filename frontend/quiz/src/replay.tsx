/**
 * Replay player for session recordings.
 *
 * Loads a recording JSON (exported locally by the user or pulled by an
 * admin) and re-animates it over a read-only reconstruction of the quiz
 * UI: pointer dot, click ripples, stage/question transitions, tab-leave
 * banners, answers. Pointer coordinates are permille of the recorded
 * viewport, so replay scales to any window size.
 */

import React, { useEffect, useMemo, useRef, useState } from "react";
import type { BakeFile, BakedQuestion } from "./types";
import type { Recording } from "./recorder";

type AnswerMark = { letter: string; correct: boolean };

type ReplayState = {
  questionId: string | null;
  stage: string;
  answers: Record<string, AnswerMark>;
  cursor: { x: number; y: number } | null;
  visible: boolean;
  scrollY: number;
  viewport: { w: number; h: number };
  clicks: Array<{ x: number; y: number; at: number }>;
};

const RIPPLE_MS = 700;

function stateAt(rec: Recording, t: number): ReplayState {
  const state: ReplayState = {
    questionId: null,
    stage: "observe",
    answers: {},
    cursor: null,
    visible: true,
    scrollY: 0,
    viewport: rec.meta.viewport,
    clicks: []
  };
  for (const ev of rec.events) {
    const dt = ev[0] as number;
    if (dt > t) break;
    const type = ev[1] as string;
    if (type === "m" || type === "c") {
      state.cursor = { x: ev[2] as number, y: ev[3] as number };
      if (type === "c") state.clicks.push({ x: ev[2] as number, y: ev[3] as number, at: dt });
    } else if (type === "s") {
      state.scrollY = ev[2] as number;
    } else if (type === "v") {
      state.visible = (ev[2] as number) === 1;
    } else if (type === "q") {
      state.questionId = String(ev[2]);
      state.stage = "observe";
    } else if (type === "g") {
      state.stage = String(ev[2]);
    } else if (type === "a") {
      if (state.questionId) {
        state.answers[state.questionId] = {
          letter: String(ev[2]),
          correct: (ev[3] as number) === 1
        };
      }
    } else if (type === "r") {
      state.viewport = { w: ev[2] as number, h: ev[3] as number };
    }
  }
  state.clicks = state.clicks.filter((c) => t - c.at < RIPPLE_MS);
  return state;
}

function semanticEvents(rec: Recording): Array<{ dt: number; label: string }> {
  const out: Array<{ dt: number; label: string }> = [];
  for (const ev of rec.events) {
    const dt = ev[0] as number;
    const type = ev[1] as string;
    if (type === "q") out.push({ dt, label: `View ${String(ev[2])}` });
    else if (type === "g") out.push({ dt, label: `Stage → ${String(ev[2])}` });
    else if (type === "a") {
      out.push({ dt, label: `Answer ${String(ev[2])} ${(ev[3] as number) === 1 ? "✓" : "✗"}` });
    } else if (type === "v") {
      out.push({ dt, label: (ev[2] as number) === 1 ? "Returned to tab" : "Left tab" });
    }
  }
  return out;
}

function fmtTime(ms: number): string {
  const total = Math.floor(ms / 1000);
  const m = Math.floor(total / 60);
  const s = total % 60;
  return `${m}:${String(s).padStart(2, "0")}`;
}

export function ReplayPlayer({
  recording,
  bake,
  onBack
}: {
  recording: Recording;
  bake: BakeFile | null;
  onBack: () => void;
}) {
  const duration = useMemo(
    () => recording.events.reduce((max, ev) => Math.max(max, ev[0] as number), 0),
    [recording]
  );
  const log = useMemo(() => semanticEvents(recording), [recording]);
  const [t, setT] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const clock = useRef({ startWall: 0, startT: 0 });
  const boxRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!playing) return;
    clock.current = { startWall: performance.now(), startT: t };
    let raf = 0;
    const tick = () => {
      const next = clock.current.startT + (performance.now() - clock.current.startWall) * speed;
      if (next >= duration) {
        setT(duration);
        setPlaying(false);
        return;
      }
      setT(next);
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [playing, speed, duration]);

  const state = useMemo(() => stateAt(recording, t), [recording, t]);
  const question: BakedQuestion | null =
    (state.questionId && bake?.byId?.[state.questionId]) || null;
  const answer = state.questionId ? state.answers[state.questionId] : undefined;

  const box = boxRef.current;
  const boxH = box?.clientHeight ?? 1;
  const scrollScale = state.viewport.h > 0 ? boxH / state.viewport.h : 1;

  return (
    <main className="shell replay">
      <header className="topnav">
        <button type="button" className="brand-btn" onClick={onBack}>
          ArchitectureIQ
        </button>
        <div />
        <button type="button" onClick={onBack}>
          Back
        </button>
      </header>

      <section className="panel replay-panel">
        <h1 className="panel-title">Session replay</h1>
        <p className="replay-meta">
          {recording.meta.username ? `${recording.meta.username} · ` : ""}
          {recording.meta.pack ?? "unknown pack"} · started{" "}
          {recording.meta.started_at ? new Date(recording.meta.started_at).toLocaleString() : "—"}{" "}
          · {fmtTime(duration)} · {recording.events.length} events
        </p>

        <div className="replay-body">
          <div className="replay-viewport" ref={boxRef}>
            <div
              className="replay-page"
              style={{ transform: `translateY(${-state.scrollY * scrollScale}px)` }}
            >
              {question ? (
                <ReplayQuestion question={question} state={state} answer={answer} />
              ) : (
                <p className="replay-empty">
                  {state.questionId
                    ? `Question ${state.questionId} is not in the currently loaded pack.`
                    : "Waiting for the first question…"}
                </p>
              )}
            </div>

            {state.cursor ? (
              <div
                className="replay-cursor"
                style={{
                  left: `${(state.cursor.x / 1000) * 100}%`,
                  top: `${(state.cursor.y / 1000) * 100}%`
                }}
              />
            ) : null}
            {state.clicks.map((click) => (
              <div
                key={click.at}
                className="replay-ripple"
                style={{
                  left: `${(click.x / 1000) * 100}%`,
                  top: `${(click.y / 1000) * 100}%`
                }}
              />
            ))}
            {!state.visible ? (
              <div className="replay-away">User left the tab / window</div>
            ) : null}
            <span className="replay-scrollhint">scroll {Math.round(state.scrollY)}px</span>
          </div>

          <aside className="replay-log">
            <p className="stage-kicker">Events</p>
            <div className="replay-log-list">
              {log.map((item, i) => (
                <button
                  key={`${item.dt}-${i}`}
                  type="button"
                  className={item.dt <= t ? "seen" : ""}
                  onClick={() => {
                    setPlaying(false);
                    setT(item.dt);
                  }}
                >
                  <span>{fmtTime(item.dt)}</span>
                  {item.label}
                </button>
              ))}
            </div>
          </aside>
        </div>

        <div className="replay-transport">
          <button
            type="button"
            className="cta"
            onClick={() => {
              if (!playing && t >= duration) setT(0);
              setPlaying(!playing);
            }}
          >
            {playing ? "Pause" : "Play"}
          </button>
          <input
            type="range"
            min={0}
            max={duration}
            step={50}
            value={Math.round(t)}
            onChange={(ev) => {
              setPlaying(false);
              setT(Number(ev.target.value));
            }}
          />
          <span className="replay-clock">
            {fmtTime(t)} / {fmtTime(duration)}
          </span>
          <select value={speed} onChange={(ev) => setSpeed(Number(ev.target.value))}>
            <option value={1}>1×</option>
            <option value={2}>2×</option>
            <option value={4}>4×</option>
          </select>
        </div>
      </section>
    </main>
  );
}

function ReplayQuestion({
  question,
  state,
  answer
}: {
  question: BakedQuestion;
  state: ReplayState;
  answer?: AnswerMark;
}) {
  const correct = question.reveal.correctLetter;
  return (
    <div className="replay-question">
      <h1 className="question-title">
        <span>{question.family.replace(/_/g, " ")}</span>
        <span className="dot">·</span>
        <span>{(question.metric ?? "").replace(/_/g, " ") || "metric"}</span>
        <span className="dot">·</span>
        <span className="tag">{question.type.replace(/_/g, " ")}</span>
        <span className="dot">·</span>
        <span className="tag">stage: {state.stage}</span>
      </h1>
      {answer ? (
        <p className={`verdict ${answer.correct ? "ok" : "bad"}`}>
          {answer.correct
            ? `Answered ${answer.letter} — correct.`
            : `Answered ${answer.letter}. Correct is ${correct}.`}
        </p>
      ) : null}
      <div className="choice-grid">
        {question.detail.choices.map((choice) => {
          const isCorrect = choice.letter === correct && state.stage === "reveal";
          const isWrongPick =
            state.stage === "reveal" && answer?.letter === choice.letter && !answer.correct;
          return (
            <div
              key={choice.letter}
              className={`choice-card${isCorrect ? " correct" : ""}${isWrongPick ? " wrong" : ""}`}
              style={{ "--choice": choice.color } as React.CSSProperties}
            >
              <span className="choice-letter">{choice.letter}</span>
              <div className="choice-fields">
                {choice.variant.map((field) => (
                  <div key={field.label} className="field vary">
                    <span>{field.label}</span>
                    <strong>{field.value}</strong>
                  </div>
                ))}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
