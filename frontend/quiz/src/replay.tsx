/**
 * Replay player for session recordings.
 *
 * Loads a recording JSON (exported locally by the user or pulled by an
 * admin) and re-animates it over a pixel-faithful reconstruction of the
 * quiz UI. The replica renders at the RECORDED viewport size using the
 * exact same components as the live app (questionView.tsx), then the
 * whole stage is CSS-scaled to fit the replay box. Pointer coordinates
 * are permille of the recorded viewport, so they map back to the same
 * on-screen elements regardless of layout reflow.
 *
 * Attention events:
 *   v = tab hidden/visible, f = window blur/focus, o = pointer left/re-entered.
 * The banner and the event log distinguish all three (legacy recordings
 * only have v, which covers both tab switches and window blur).
 */

import { useEffect, useMemo, useRef, useState } from "react";
import type { BakeFile, BakedQuestion } from "./types";
import type { Recording } from "./recorder";
import {
  AnswerStage,
  ChoicesStage,
  DatasetStage,
  humanFamily,
  humanMetric,
  humanType,
  type FeedbackDraft
} from "./questionView";

type AnswerMark = { letter: string; correct: boolean };

type ReplayState = {
  questionId: string | null;
  stage: string;
  answers: Record<string, AnswerMark>;
  cursor: { x: number; y: number } | null;
  tabVisible: boolean;
  windowFocused: boolean;
  pointerIn: boolean;
  scrollY: number;
  viewport: { w: number; h: number };
  clicks: Array<{ x: number; y: number; at: number }>;
};

const RIPPLE_MS = 700;

const EMPTY_FEEDBACK: FeedbackDraft = {
  confidence: null,
  decision: null,
  comment: "",
  submitted: false
};
const noop = () => {};

function stateAt(rec: Recording, t: number): ReplayState {
  const state: ReplayState = {
    questionId: null,
    stage: "observe",
    answers: {},
    cursor: null,
    tabVisible: true,
    windowFocused: true,
    pointerIn: true,
    scrollY: 0,
    viewport: rec.meta.viewport ?? { w: 1280, h: 800 },
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
      state.tabVisible = (ev[2] as number) === 1;
    } else if (type === "f") {
      state.windowFocused = (ev[2] as number) === 1;
    } else if (type === "o") {
      state.pointerIn = (ev[2] as number) === 1;
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

type Attention = "tab" | "focus" | "pointer" | null;

function attentionOf(state: ReplayState): Attention {
  if (!state.tabVisible) return "tab";
  if (!state.windowFocused) return "focus";
  if (!state.pointerIn) return "pointer";
  return null;
}

const ATTENTION_TEXT: Record<Exclude<Attention, null>, string> = {
  tab: "User switched away from the tab",
  focus: "Window lost focus (clicked outside the app)",
  pointer: "Pointer left the page"
};

function semanticEvents(rec: Recording): Array<{ dt: number; label: string; kind: string }> {
  const out: Array<{ dt: number; label: string; kind: string }> = [];
  for (const ev of rec.events) {
    const dt = ev[0] as number;
    const type = ev[1] as string;
    if (type === "q") out.push({ dt, label: `View ${String(ev[2])}`, kind: "q" });
    else if (type === "g") out.push({ dt, label: `Stage → ${String(ev[2])}`, kind: "g" });
    else if (type === "a") {
      out.push({
        dt,
        label: `Answer ${String(ev[2])} ${(ev[3] as number) === 1 ? "✓" : "✗"}`,
        kind: "a"
      });
    } else if (type === "v") {
      out.push({
        dt,
        kind: "v",
        label: (ev[2] as number) === 1 ? "Returned to tab" : "Left tab (hidden)"
      });
    } else if (type === "f") {
      out.push({
        dt,
        kind: "f",
        label: (ev[2] as number) === 1 ? "Window refocused" : "Window lost focus"
      });
    } else if (type === "o") {
      out.push({
        dt,
        kind: "o",
        label: (ev[2] as number) === 1 ? "Pointer returned" : "Pointer left page"
      });
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

/** The exact page the examinee saw, built from the live app's components. */
function ReplicaPage({
  question,
  stage,
  answer
}: {
  question: BakedQuestion | null;
  stage: string;
  answer?: AnswerMark;
}) {
  return (
    <main className="shell quiz">
      <header className="topnav">
        <span className="brand-btn">ArchitectureIQ</span>
        <div className="progress">
          <span>—</span>
          <div className="progress-track">
            <div style={{ width: "0%" }} />
          </div>
        </div>
        <div className="top-actions">
          <span className="score-text">Score —</span>
          <button type="button">Next</button>
          <button type="button">Random</button>
          <button type="button">Questions</button>
          <button type="button">Export</button>
        </div>
      </header>
      {question ? (
        <>
          <h1 className="question-title">
            <span>{humanFamily(question.family)}</span>
            <span className="dot">·</span>
            <span>{humanMetric(question.metric)}</span>
            <span className="dot">·</span>
            <span className="tag">{humanType(question.type)}</span>
            <span className="dot">·</span>
            <span className="tag">{question.track ?? "default"}</span>
            <span className="dot">·</span>
            <span>{question.detail.choices.length} choices</span>
          </h1>
          <section className="stage-screen">
            <div className="provenance">
              <span>Track: {question.track ?? "default"}</span>
              <span>Profile: {question.profile ?? "legacy/unknown"}</span>
              <span>Hash: {question.profileHash ?? "legacy/unknown"}</span>
            </div>
            <DatasetStage question={question} onSeeChoices={noop} onInfo={noop} />
            {stage !== "reveal" ? (
              <ChoicesStage question={question} onPick={noop} onInfo={noop} />
            ) : (
              <AnswerStage
                question={question}
                selected={answer?.letter ?? null}
                feedback={EMPTY_FEEDBACK}
                onFeedbackChange={noop}
                onSubmitFeedback={noop}
                onNext={noop}
                onInfo={noop}
                onDatasetInfo={noop}
              />
            )}
          </section>
        </>
      ) : (
        <p className="replay-empty">
          {stage === "observe"
            ? "Waiting for the first question…"
            : "Question is not in the currently loaded pack."}
        </p>
      )}
    </main>
  );
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
  const warnMarks = useMemo(
    () =>
      recording.events
        .filter((ev) => ["v", "f", "o"].includes(ev[1] as string) && (ev[2] as number) === 0)
        .map((ev) => ev[0] as number),
    [recording]
  );
  const [t, setT] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const clock = useRef({ startWall: 0, startT: 0 });
  const boxRef = useRef<HTMLDivElement | null>(null);
  const [boxSize, setBoxSize] = useState({ w: 0, h: 0 });

  useEffect(() => {
    const el = boxRef.current;
    if (!el || typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver((entries) => {
      const rect = entries[0].contentRect;
      setBoxSize({ w: rect.width, h: rect.height });
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

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

  const vp = state.viewport;
  const fit =
    vp.w > 0 && vp.h > 0 && boxSize.w > 0
      ? Math.min(boxSize.w / vp.w, boxSize.h / vp.h)
      : 1;
  const attention = attentionOf(state);
  const px = (permille: number, total: number) => (permille / 1000) * total;

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
          · {fmtTime(duration)} · {recording.events.length} events · viewport{" "}
          {vp.w}×{vp.h} ({Math.round(fit * 100)}%)
        </p>

        <div className="replay-body">
          <div className="replay-viewport" ref={boxRef}>
            <div
              className="replay-stage"
              style={{
                width: vp.w,
                height: vp.h,
                transform: `scale(${fit})`,
                opacity: attention ? 0.45 : 1
              }}
            >
              <div
                className="replay-content"
                style={{ width: vp.w, transform: `translateY(${-state.scrollY}px)` }}
              >
                <ReplicaPage question={question} stage={state.stage} answer={answer} />
              </div>

              {state.cursor ? (
                <div
                  className="replay-cursor"
                  style={{
                    left: px(state.cursor.x, vp.w),
                    top: px(state.cursor.y, vp.h),
                    width: 14 / fit,
                    height: 14 / fit,
                    margin: `${-7 / fit}px 0 0 ${-7 / fit}px`
                  }}
                />
              ) : null}
              {state.clicks.map((click) => (
                <div
                  key={click.at}
                  className="replay-ripple"
                  style={{
                    left: px(click.x, vp.w),
                    top: px(click.y, vp.h),
                    width: 36 / fit,
                    height: 36 / fit,
                    margin: `${-18 / fit}px 0 0 ${-18 / fit}px`
                  }}
                />
              ))}
              {attention ? (
                <div className={`replay-away warn-${attention}`}>
                  {ATTENTION_TEXT[attention]}
                </div>
              ) : null}
              <span className="replay-scrollhint">scroll {Math.round(state.scrollY)}px</span>
            </div>
          </div>

          <aside className="replay-log">
            <p className="stage-kicker">Events</p>
            <div className="replay-log-list">
              {log.map((item, i) => (
                <button
                  key={`${item.dt}-${i}`}
                  type="button"
                  className={`${item.kind}${item.dt <= t ? " seen" : ""}`}
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
        {warnMarks.length ? (
          <div className="replay-marks">
            {warnMarks.map((mark, i) => (
              <button
                key={`${mark}-${i}`}
                type="button"
                title="Attention event — click to jump"
                style={{ left: `${(mark / Math.max(duration, 1)) * 100}%` }}
                onClick={() => {
                  setPlaying(false);
                  setT(Math.max(0, mark - 1500));
                }}
              />
            ))}
          </div>
        ) : null}
      </section>
    </main>
  );
}
