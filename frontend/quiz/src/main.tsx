import React, { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";
import { newSessionId, track } from "./telemetry";
import { parseRecording, SessionRecorder, type Recording } from "./recorder";
import {
  apiConfigured,
  clearAuth,
  drainQueue,
  listAnswers,
  loadAuth,
  recordAnswer,
  registerUser,
  upsertSession,
  uploadChunk,
  type Auth
} from "./api";
import { ReplayPlayer } from "./replay";
import type {
  BakeFile,
  BakedQuestion,
  Stage
} from "./types";

import {
  AnswerStage,
  DatasetStage,
  ChoicesStage,
  humanFamily,
  humanMetric,
  humanMetricByFamily,
  humanType,
  type FeedbackDraft
} from "./questionView";


type Screen = "home" | "quiz" | "menu" | "contact";
type InfoTarget =
  | { kind: "dataset" }
  | { kind: "choice"; letter: string }
  | null;

const EMPTY_FEEDBACK: FeedbackDraft = { confidence: null, decision: null, comment: "", submitted: false };


function App() {
  const [bake, setBake] = useState<BakeFile | null>(null);
  const [screen, setScreen] = useState<Screen>("home");
  const [index, setIndex] = useState(0);
  const [stage, setStage] = useState<Stage>("observe");
  const [selected, setSelected] = useState<string | null>(null);
  const [answered, setAnswered] = useState(false);
  const [info, setInfo] = useState<InfoTarget>(null);
  const [error, setError] = useState<string | null>(null);
  const sessionId = useRef(newSessionId());
  const viewStartedAt = useRef(Date.now());
  const startedTracked = useRef(false);
  const results = useRef<Record<string, { correct: boolean; picked: string; repeat?: boolean }>>({});
  const feedbackByQuestion = useRef<Record<string, FeedbackDraft>>({});
  const [feedback, setFeedback] = useState<FeedbackDraft>(EMPTY_FEEDBACK);
  const [, bump] = useState(0);
  const [auth, setAuth] = useState<Auth | null>(() => loadAuth());
  const [showAuth, setShowAuth] = useState(false);
  const [replay, setReplay] = useState<Recording | null>(null);
  const [packId, setPackId] = useState<string | null>(null);
  const recorderRef = useRef<SessionRecorder | null>(null);
  const authRef = useRef<Auth | null>(auth);
  authRef.current = auth;
  const pendingBegin = useRef<number | null>(null);
  // Server-persisted answers for the signed-in user: refresh restores them.
  const answeredMapRef = useRef<Record<string, { picked: string; correct: boolean; attempts: number }>>({});
  const [answeredMapVersion, setAnsweredMapVersion] = useState(0);
  // One-shot completion celebration: fires when every question in the pack
  // has an answer (live or restored from the server).
  const [celebrate, setCelebrate] = useState(false);
  const celebratedRef = useRef(false);

  useEffect(() => {
    if (authRef.current) void drainQueue(authRef.current);
  }, []);

  // On user change: if the current session/recorder belongs to a different
  // user, stop it and wipe in-memory state so answers can not leak across
  // users. A recorder created for THIS user (fresh sign-in begins the quiz
  // before this effect commits) is kept. Server rows stay untouched.
  const lastAuthUserId = useRef<string | null>(auth?.user_id ?? null);
  const sessionUserId = useRef<string | null>(null);
  // Lets the mount-time deep-link effect call the latest beginQuiz closure.
  const beginQuizRef = useRef<((at: number) => void) | null>(null);
  useEffect(() => {
    const uid = auth?.user_id ?? null;
    if (uid === lastAuthUserId.current) {
      return;
    }
    lastAuthUserId.current = uid;
    if (sessionUserId.current !== uid) {
      recorderRef.current?.stop();
      recorderRef.current = null;
      sessionUserId.current = uid;
      results.current = {};
      feedbackByQuestion.current = {};
      answeredMapRef.current = {};
      celebratedRef.current = false;
      setCelebrate(false);
      sessionId.current = newSessionId();
      setIndex(0);
      setSelected(null);
      setAnswered(false);
      setStage("observe");
      setAnsweredMapVersion((v) => v + 1);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [auth?.user_id]);

  useEffect(() => {
    const current = authRef.current;
    if (!current || !apiConfigured()) {
      return;
    }
    let cancelled = false;
    // Fetch ALL answers for this user, not just the active pack: a stale
    // link (e.g. an old ?question_pack= bookmark) must still restore and
    // lock previously answered questions — records are per (user, question).
    listAnswers(current)
      .then((records) => {
        if (cancelled) return;
        const map: typeof answeredMapRef.current = {};
        for (const record of records) {
          map[record.question_id] = {
            picked: record.picked,
            correct: record.correct,
            attempts: record.attempts
          };
        }
        answeredMapRef.current = map;
        setAnsweredMapVersion((v) => v + 1);
      })
      .catch(() => {
        /* offline: keep whatever we have; re-answering remains blocked by
           local results until the list arrives */
      });
    return () => {
      cancelled = true;
    };
  }, [auth?.user_id]);

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const packId = params.get("question_pack");
    const packUrl = packId
      ? `data/packs/${encodeURIComponent(packId)}.json`
      : "data/packs/v15-launch50-seed20260905.json";
    setPackId(packId ?? "v15-launch50-seed20260905");
    fetch(packUrl)
      .then((response) => {
        if (!response.ok) {
          throw new Error(`Missing baked questions at ${packUrl}`);
        }
        return response.json();
      })
      .then((data: BakeFile) => {
        setBake(data);
        const target = params.get("q");
        if (target) {
          const at = data.questions.findIndex((item) => item.id === target);
          if (at >= 0) {
            // Enter through beginQuiz so sign-in and the session recorder
            // are guaranteed (deep links used to bypass both).
            beginQuizRef.current?.(at);
          }
        }
      })
      .catch((err) => setError(err instanceof Error ? err.message : String(err)));
  }, []);

  const summaries = bake?.questions ?? [];
  const currentId = summaries[index]?.id;
  const question = currentId && bake ? bake.byId[currentId] : null;

  const score = useMemo(() => {
    const values = Object.values(results.current);
    const total = values.length;
    const correct = values.filter((item) => item.correct).length;
    return { correct, total };
  }, [answered, index, screen, bump]);

  // Completion celebration: every question in the pack has an answer.
  useEffect(() => {
    if (!bake || !summaries.length || screen !== "quiz") return;
    if (score.total < summaries.length || celebratedRef.current) return;
    celebratedRef.current = true;
    setCelebrate(true);
    recorderRef.current?.mark("g", "complete");
  }, [bake, score.total, summaries.length, screen]);

  useEffect(() => {
    if (screen !== "quiz" || !question) {
      return;
    }
    const persisted = answeredMapRef.current[question.id];
    if (persisted && results.current[question.id] === undefined) {
      // Resume a previously submitted answer (page refresh / new device):
      // locked, cannot be re-answered, flagged as a repeat in the export.
      results.current[question.id] = {
        correct: persisted.correct,
        picked: persisted.picked,
        repeat: true
      };
    }
    const prior = results.current[question.id];
    const already = prior !== undefined;
    setStage(already ? "reveal" : "observe");
    setFeedback({ ...(feedbackByQuestion.current[question.id] ?? EMPTY_FEEDBACK) });
    setSelected(prior?.picked ?? null);
    setAnswered(already);
    setInfo(null);
    viewStartedAt.current = Date.now();
    recorderRef.current?.mark("q", question.id);
    track({
      session_id: sessionId.current,
      event_type: "question_view",
      question_id: question.id
    });
  }, [screen, question?.id, answeredMapVersion]);

  useEffect(() => {
    if (screen === "quiz" && question) {
      recorderRef.current?.mark("g", stage);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [stage, screen, question?.id]);

  function ensureRecorder(): SessionRecorder {
    if (!recorderRef.current) {
      const recorder = new SessionRecorder(sessionId.current);
      recorder.onFlush = (seq, events, final) => {
        const current = authRef.current;
        if (!current) return;
        void uploadChunk(
          current,
          { session_id: recorder.sessionId, seq, events },
          final
        ).then(() => drainQueue(current));
      };
      recorderRef.current = recorder;
    }
    return recorderRef.current;
  }

  function currentScore() {
    const values = Object.values(results.current);
    return {
      correct: values.filter((item) => item.correct).length,
      total: values.length
    };
  }

  function ensureSessionStart() {
    if (startedTracked.current) {
      return;
    }
    startedTracked.current = true;
    track({
      session_id: sessionId.current,
      event_type: "session_start",
      payload: { app_version: "quiz-0.3" }
    });
  }

  function exportSession() {
    const payload = {
      schema_version: 1,
      exported_at: new Date().toISOString(),
      session_id: sessionId.current,
      collection: bake?.collection ?? null,
      username: authRef.current?.username ?? null,
      results: results.current,
      audit_feedback: feedbackByQuestion.current,
      recording: recorderRef.current?.snapshot() ?? null
    };
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `architectureiq-session-${sessionId.current}.json`;
    anchor.click();
    window.setTimeout(() => URL.revokeObjectURL(url), 0);
  }

  function beginQuiz(atIndex = 0) {
    if (apiConfigured() && !authRef.current) {
      pendingBegin.current = atIndex;
      setShowAuth(true);
      return;
    }
    ensureSessionStart();
    const recorder = ensureRecorder();
    recorder.userId = authRef.current?.user_id ?? null;
    if (!recorder.isRunning) {
      recorder.start({ username: authRef.current?.username, pack: packId ?? undefined });
    }
    const current = authRef.current;
    sessionUserId.current = current?.user_id ?? null;
    if (current && apiConfigured()) {
      const scoreNow = currentScore();
      void upsertSession(current, {
        session_id: recorder.sessionId,
        pack: packId ?? undefined,
        score_correct: scoreNow.correct,
        score_total: scoreNow.total,
        meta: { user_agent: navigator.userAgent }
      });
    }
    setIndex(atIndex);
    setScreen("quiz");
  }
  beginQuizRef.current = beginQuiz;

  function goHome() {
    setScreen("home");
    setInfo(null);
  }

  function leaveAndSwitch(nextIndex: number) {
    if (question && !answered && results.current[question.id] === undefined) {
      track({
        session_id: sessionId.current,
        event_type: "question_leave",
        question_id: question.id,
        duration_ms: Date.now() - viewStartedAt.current
      });
    }
    setIndex(nextIndex);
    setInfo(null);
  }

  function nextQuestion() {
    if (!summaries.length || (bake?.ordered && index >= summaries.length - 1)) {
      return;
    }
    const next = bake?.ordered ? index + 1 : (index + 1) % summaries.length;
    leaveAndSwitch(next);
  }

  function randomQuestion() {
    if (summaries.length <= 1) {
      return;
    }
    let next = index;
    while (next === index) {
      next = Math.floor(Math.random() * summaries.length);
    }
    leaveAndSwitch(next);
  }

  function updateFeedback(
    patch: Partial<Pick<FeedbackDraft, "confidence" | "decision" | "comment">>
  ) {
    if (!question || feedback.submitted) {
      return;
    }
    const next = { ...feedback, ...patch };
    feedbackByQuestion.current[question.id] = next;
    setFeedback(next);
  }

  function submitAuditFeedback() {
    if (
      !question ||
      feedback.submitted ||
      feedback.confidence === null ||
      feedback.decision === null
    ) {
      return;
    }
    const answer = results.current[question.id];
    const pickedLetter = answer?.picked ?? selected;
    const correct = answer?.correct ??
      (pickedLetter ? pickedLetter === question.reveal.correctLetter : null);
    const next = { ...feedback, submitted: true };
    feedbackByQuestion.current[question.id] = next;
    setFeedback(next);
    track({
      session_id: sessionId.current,
      event_type: "audit_feedback",
      question_id: question.id,
      duration_ms: Date.now() - viewStartedAt.current,
      payload: {
        confidence: feedback.confidence,
        decision: feedback.decision,
        comment: feedback.comment.trim() || null,
        picked_letter: pickedLetter,
        correct,
        stage: "reveal",
        track: question.track ?? "default",
        profile: question.profile ?? "legacy/unknown",
        profile_hash: question.profileHash ?? "legacy/unknown"
      }
    });
  }
  function pickChoice(letter: string) {
    if (!question || answered || results.current[question.id] !== undefined) {
      return;
    }
    const correct = letter === question.reveal.correctLetter;
    const answeredMap = answeredMapRef.current;
    results.current[question.id] = {
      correct,
      picked: letter,
      repeat: answeredMap[question.id] != null
    };
    setSelected(letter);
    setAnswered(true);
    bump((n) => n + 1);
    recorderRef.current?.mark("a", letter, correct ? 1 : 0);
    const current = authRef.current;
    if (current && apiConfigured()) {
      void recordAnswer(current, {
        question_id: question.id,
        picked: letter,
        correct,
        pack: packId ?? undefined
      })
        .then((res) => {
          answeredMapRef.current[question.id] = {
            picked: res.picked,
            correct: res.correct,
            attempts: 1
          };
          if (res.duplicate) {
            const entry = results.current[question.id];
            if (entry) entry.repeat = true;
            bump((n) => n + 1);
          }
        })
        .catch(() => {
          // offline / raced reload: one deferred retry, then give up — the
          // trajectory chunks still carry the attempt for proctoring.
          window.setTimeout(() => {
            const retry = authRef.current;
            if (!retry) return;
            void recordAnswer(retry, {
              question_id: question.id,
              picked: letter,
              correct,
              pack: packId ?? undefined
            }).catch(() => undefined);
          }, 1_500);
        });
      const scoreNow = currentScore();
      void upsertSession(current, {
        session_id: sessionId.current,
        pack: packId ?? undefined,
        score_correct: scoreNow.correct,
        score_total: scoreNow.total
      });
    }
    track({
      session_id: sessionId.current,
      event_type: "answer_submit",
      question_id: question.id,
      duration_ms: Date.now() - viewStartedAt.current,
      payload: { picked_letter: letter, correct }
    });
    track({
      session_id: sessionId.current,
      event_type: "stage_change",
      question_id: question.id,
      payload: { from: stage, to: "reveal" }
    });
    setStage("reveal");
  }
  function goCompare() {
    if (!question) {
      return;
    }
    track({
      session_id: sessionId.current,
      event_type: "stage_change",
      question_id: question.id,
      payload: { from: "observe", to: "compare" }
    });
    document.getElementById("choices-anchor")?.scrollIntoView({ behavior: "smooth" });
  }

  function openReplayFile(file: File) {
    void file.text().then((text) => {
      try {
        const parsed = JSON.parse(text) as { recording?: unknown };
        const rec = parseRecording(parsed);
        if (!rec) {
          window.alert(
            parsed && typeof parsed === "object" && "recording" in parsed && parsed.recording === null
              ? "This export contains your results but no trajectory: the session was played on a page that never started recording (e.g. opened via a direct question link before this fix)."
              : "That file is not a valid session recording."
          );
          return;
        }
        setReplay(rec);
      } catch {
        window.alert("That file is not a valid session recording.");
      }
    });
  }

  if (error) {
    return (
      <main className="shell">
        <p className="error">{error}</p>
      </main>
    );
  }

  if (!bake) {
    return (
      <main className="shell">
        <p className="loading">Loading…</p>
      </main>
    );
  }

  if (replay) {
    return <ReplayPlayer recording={replay} bake={bake} onBack={() => setReplay(null)} />;
  }

  if (screen === "home") {
    return (
      <>
        <HomeScreen
          ready={Boolean(bake)}
          auth={auth}
          onBegin={() => beginQuiz(0)}
          onMenu={() => setScreen("menu")}
          onContact={() => setScreen("contact")}
          onSwitchUser={() => {
            clearAuth();
            setAuth(null);
            setShowAuth(true);
          }}
          onReplayFile={openReplayFile}
        />
        {showAuth ? (
          <AuthGate
            allowCancel={Boolean(auth)}
            onCancel={() => setShowAuth(false)}
            onSuccess={(next) => {
              authRef.current = next; // setAuth is async; beginQuiz reads the ref
              setAuth(next);
              setShowAuth(false);
              const at = pendingBegin.current ?? 0;
              pendingBegin.current = null;
              beginQuiz(at);
            }}
          />
        ) : null}
      </>
    );
  }

  if (screen === "contact") {
    return (
      <SimpleScreen title="Contact us" onBack={goHome}>
        <p className="body-copy">
          ArchitectureIQ is a research prototype. For questions about the benchmark or this human
          quiz, email{" "}
          <a href="mailto:rzr23@mails.tsinghua.edu.cn">rzr23@mails.tsinghua.edu.cn</a>.
        </p>
      </SimpleScreen>
    );
  }

  if (screen === "menu") {
    return (
      <QuestionMenu
        summaries={summaries}
        onBack={goHome}
        onPick={(itemIndex) => beginQuiz(itemIndex)}
      />
    );
  }

  if (!question) {
    return (
      <main className="shell">
        <p className="loading">No questions available.</p>
      </main>
    );
  }

  const progress = summaries.length ? ((index + 1) / summaries.length) * 100 : 0;
  const accuracy =
    score.total > 0 ? `${Math.round((100 * score.correct) / score.total)}%` : "—";

  return (
    <main className="shell quiz">
      <header className="topnav">
        <button type="button" className="brand-btn" onClick={goHome}>
          ArchitectureIQ
        </button>
        <div className="progress" aria-label={`Question ${index + 1} of ${summaries.length}`}>
          <span>
            {index + 1} / {summaries.length}
          </span>
          <div className="progress-track">
            <div style={{ width: `${progress}%` }} />
          </div>
        </div>
        <div className="top-actions">
          <span className="score-text" title="Session accuracy">
            Score {score.correct}/{score.total} ({accuracy})
          </span>
          <button
            type="button"
            onClick={nextQuestion}
            disabled={Boolean(bake.ordered && index >= summaries.length - 1)}
          >
            {bake.ordered && index >= summaries.length - 1 ? "End" : "Next"}
          </button>
          <button type="button" onClick={randomQuestion}>
            Random
          </button>
          <button type="button" onClick={() => setScreen("menu")}>
            Questions
          </button>
          <button type="button" onClick={exportSession}>
            Export
          </button>
        </div>
      </header>

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

      <section className="stage-screen" key={question.id}>
        <div className="provenance" aria-label="Question provenance">
          <span>Track: {question.track ?? "default"}</span>
          <span>Profile: {question.profile ?? "legacy/unknown"}</span>
          <span>Hash: {question.profileHash ?? "legacy/unknown"}</span>
        </div>
        <DatasetStage
          question={question}
          onSeeChoices={goCompare}
          onInfo={() => setInfo({ kind: "dataset" })}
        />
        {stage !== "reveal" ? (
          <ChoicesStage
            question={question}
            onPick={pickChoice}
            onInfo={(letter) => setInfo({ kind: "choice", letter })}
          />
        ) : (
          <AnswerStage
            question={question}
            selected={selected}
            feedback={feedback}
            onFeedbackChange={updateFeedback}
            onSubmitFeedback={submitAuditFeedback}
            onNext={nextQuestion}
            onInfo={(letter) => setInfo({ kind: "choice", letter })}
            onDatasetInfo={() => setInfo({ kind: "dataset" })}
          />
        )}
      </section>

      {info ? (
        <InfoModal
          question={question}
          target={info}
          answered={answered}
          onClose={() => setInfo(null)}
        />
      ) : null}

      {celebrate ? (
        <CompletionOverlay
          correct={score.correct}
          total={score.total}
          onExport={exportSession}
          onClose={() => setCelebrate(false)}
        />
      ) : null}
    </main>
  );
}

// Score ladder on the same 50-question launch set, mapped to each model's
// accuracy on the full 500-question set (shown in the completion overlay).
const LLM_LADDER: Array<{ min: number; models: string; full: string }> = [
  { min: 33, models: "GPT-5.6 Sol", full: "76.4%" },
  { min: 32, models: "Claude Opus 5", full: "76.0%" },
  {
    min: 30,
    models: "Claude Sonnet 5 · Gemini 3.1 Pro Preview · GPT-5.6 Luna",
    full: "65.4% – 67.8%"
  },
  { min: 28, models: "DeepSeek R1", full: "59.0%" },
  { min: 25, models: "Gemini 2.5 Pro", full: "55.8%" },
  { min: 23, models: "DeepSeek R1-Distill 32B", full: "46.0%" },
  { min: 21, models: "Llama 3.1 70B", full: "42.0%" },
  { min: 0, models: "You — a human", full: "—" }
];

function CompletionOverlay({
  correct,
  total,
  onExport,
  onClose
}: {
  correct: number;
  total: number;
  onExport: () => void;
  onClose: () => void;
}) {
  const pieces = useRef(
    Array.from({ length: 90 }, (_, i) => ({
      left: (i * 37) % 100,
      delay: ((i * 13) % 30) / 10,
      duration: 2.6 + ((i * 7) % 20) / 10,
      color: ["#ffd166", "#ef476f", "#06d6a0", "#118ab2", "#f78c6b", "#b388eb"][i % 6],
      size: 6 + ((i * 5) % 8),
      round: i % 3 === 0
    }))
  ).current;
  const pct = total > 0 ? Math.round((100 * correct) / total) : 0;
  const matchIndex = LLM_LADDER.findIndex((row) => correct >= row.min);
  const match = LLM_LADDER[matchIndex];
  return (
    <div className="completion-overlay" role="dialog" aria-label="Quiz complete">
      <div className="confetti" aria-hidden="true">
        {pieces.map((p, i) => (
          <i
            key={i}
            style={{
              left: `${p.left}%`,
              animationDelay: `${p.delay}s`,
              animationDuration: `${p.duration}s`,
              background: p.color,
              width: p.size,
              height: p.round ? p.size : p.size * 1.8,
              borderRadius: p.round ? "50%" : "2px"
            }}
          />
        ))}
      </div>
      <div className="completion-card">
        <div className="completion-trophy" aria-hidden="true">
          🏆
        </div>
        <h2>All done!</h2>
        <p className="completion-score">
          You answered all {total} questions — score <strong>{correct}</strong>/{total} ({pct}%)
        </p>
        <p className="completion-note">
          Your answers and session have been recorded. Thanks for taking the ArchitectureIQ quiz!
        </p>
        <div className="completion-match">
          <p className="completion-match-line">
            {match.min > 0 ? (
              <>
                This score matches <strong>{match.models}</strong> — {match.full} on the full
                500-question set
              </>
            ) : (
              <>No LLM on our leaderboard scored this low — you are, in fact, a human.</>
            )}
          </p>
          <table className="completion-ladder">
            <thead>
              <tr>
                <th>Your score</th>
                <th>Model</th>
                <th>Acc (500)</th>
              </tr>
            </thead>
            <tbody>
              {LLM_LADDER.map((row, i) => (
                <tr
                  key={row.min}
                  className={i === matchIndex ? "is-you" : i > matchIndex ? "is-above" : ""}
                >
                  <td>{row.min > 0 ? `${row.min}+ / 50` : "< 21 / 50"}</td>
                  <td>{row.models}</td>
                  <td>{row.full}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="completion-actions">
          <button type="button" className="completion-btn primary" onClick={onExport}>
            Export my results
          </button>
          <button type="button" className="completion-btn" onClick={onClose}>
            Back to questions
          </button>
        </div>
      </div>
    </div>
  );
}

function HomeScreen({
  ready,
  auth,
  onBegin,
  onMenu,
  onContact,
  onSwitchUser,
  onReplayFile
}: {
  ready: boolean;
  auth: Auth | null;
  onBegin: () => void;
  onMenu: () => void;
  onContact: () => void;
  onSwitchUser: () => void;
  onReplayFile: (file: File) => void;
}) {
  const fileInput = useRef<HTMLInputElement | null>(null);
  return (
    <main className="shell home">
      <div className="home-block">
        <h1 className="home-title">ArchitectureIQ</h1>
        <p className="home-tagline">
          A human playable edition of an LLM benchmark on deep learning modeling intuition.
        </p>
        <div className="menu-stack">
          <button type="button" className="menu-btn begin" disabled={!ready} onClick={onBegin}>
            Begin
          </button>
          <button type="button" className="menu-btn" onClick={onMenu}>
            Question menu
          </button>
          <button type="button" className="menu-btn" onClick={onContact}>
            Contact us
          </button>
          <button
            type="button"
            className="menu-btn"
            disabled={!ready}
            onClick={() => fileInput.current?.click()}
          >
            Replay a recording
          </button>
        </div>
        <p className="home-auth">
          {auth
            ? `Signed in as ${auth.username} · `
            : "You will be asked for a username and the group password before you begin. "}
          {auth ? (
            <button type="button" className="linklike" onClick={onSwitchUser}>
              Switch user
            </button>
          ) : null}
        </p>
        <input
          ref={fileInput}
          type="file"
          accept="application/json,.json"
          hidden
          onChange={(event) => {
            const file = event.target.files?.[0];
            if (file) onReplayFile(file);
            event.target.value = "";
          }}
        />
      </div>
    </main>
  );
}

function AuthGate({
  allowCancel,
  onCancel,
  onSuccess
}: {
  allowCancel: boolean;
  onCancel: () => void;
  onSuccess: (auth: Auth) => void;
}) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  function submit() {
    if (busy) return;
    setBusy(true);
    setError(null);
    registerUser(username, password)
      .then(onSuccess)
      .catch((err) => {
        const message = err instanceof Error ? err.message : String(err);
        if (message === "invalid_password") setError("Wrong password.");
        else if (message === "invalid_username_length") setError("Pick a username (1-40 characters).");
        else setError("Could not reach the server — check your connection and try again.");
        setBusy(false);
      });
  }

  return (
    <div className="modal-backdrop" onClick={allowCancel ? onCancel : undefined}>
      <div className="modal auth-modal" onClick={(event) => event.stopPropagation()}>
        <h2>Sign in to play</h2>
        <p className="auth-note">
          Pick any username and enter the group password. Your score and an anonymized mouse
          trajectory (moves, clicks, tab switches — nothing outside this page) are recorded for
          research. You can download your own recording any time via Export.
        </p>
        <label className="auth-field">
          <span>Username</span>
          <input
            value={username}
            autoFocus
            maxLength={40}
            placeholder="e.g. ada_lovelace"
            onChange={(event) => setUsername(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter") submit();
            }}
          />
        </label>
        <label className="auth-field">
          <span>Group password</span>
          <input
            type="password"
            value={password}
            onChange={(event) => setPassword(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter") submit();
            }}
          />
        </label>
        {error ? <p className="auth-error">{error}</p> : null}
        <div className="auth-actions">
          {allowCancel ? (
            <button type="button" onClick={onCancel}>
              Cancel
            </button>
          ) : null}
          <button
            type="button"
            className="cta"
            disabled={busy || !username.trim() || !password}
            onClick={submit}
          >
            {busy ? "Signing in…" : "Sign in & begin"}
          </button>
        </div>
      </div>
    </div>
  );
}

function SimpleScreen({
  title,
  onBack,
  children
}: {
  title: string;
  onBack: () => void;
  children: React.ReactNode;
}) {
  return (
    <main className="shell">
      <header className="topnav">
        <button type="button" className="brand-btn" onClick={onBack}>
          ArchitectureIQ
        </button>
        <div />
        <button type="button" onClick={onBack}>
          Back
        </button>
      </header>
      <section className="panel">
        <h1 className="panel-title">{title}</h1>
        {children}
      </section>
    </main>
  );
}

function QuestionMenu({
  summaries,
  onBack,
  onPick
}: {
  summaries: BakeFile["questions"];
  onBack: () => void;
  onPick: (index: number) => void;
}) {
  return (
    <SimpleScreen title="Question menu" onBack={onBack}>
      <ul className="question-list">
        {summaries.map((item, itemIndex) => (
          <li key={item.id}>
            <button type="button" className="question-row" onClick={() => onPick(itemIndex)}>
              <span className="qnum">{itemIndex + 1}</span>
              <span>
                {humanFamily(item.family)} · {humanMetricByFamily(item.family, item.metric)} ·{" "}
                {humanType(item.type)} · {item.track ?? "default"} · {item.choices ?? "?"} choices
              </span>
            </button>
          </li>
        ))}
      </ul>
    </SimpleScreen>
  );
}


function InfoModal({
  question,
  target,
  answered,
  onClose
}: {
  question: BakedQuestion;
  target: Exclude<InfoTarget, null>;
  answered: boolean;
  onClose: () => void;
}) {
  const files = useMemo(() => {
    if (target.kind === "dataset") {
      return question.detail.dataset.files ?? {};
    }
    const choice = question.detail.choices.find((item) => item.letter === target.letter);
    const base = { ...(choice?.files ?? {}) };
    if (answered && target.letter && question.reveal.files?.[target.letter]) {
      Object.assign(base, question.reveal.files[target.letter]);
    }
    return base;
  }, [answered, question, target]);

  const names = Object.keys(files);
  const [fileName, setFileName] = useState(names[0] ?? "");
  useEffect(() => {
    setFileName(names[0] ?? "");
  }, [names.join("|")]);

  const content = fileName ? files[fileName] : null;
  const title =
    target.kind === "dataset" ? "Dataset definition" : `Choice ${target.letter} definition`;

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal" onClick={(event) => event.stopPropagation()}>
        <div className="panel-head">
          <h2>{title}</h2>
          <button type="button" onClick={onClose}>
            Close
          </button>
        </div>
        <select value={fileName} onChange={(event) => setFileName(event.target.value)}>
          {names.map((name) => (
            <option key={name} value={name}>
              {name}
            </option>
          ))}
        </select>
        <pre>{typeof content === "string" ? content : JSON.stringify(content, null, 2)}</pre>
      </div>
    </div>
  );
}

createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
