import React, { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";
import { newSessionId, track } from "./telemetry";
import type {
  AuditDecision,
  BakeFile,
  BakedQuestion,
  Choice,
  ConfidenceRating,
  Field,
  Point,
  Stage
} from "./types";

type Screen = "home" | "quiz" | "menu" | "contact";
type InfoTarget =
  | { kind: "dataset" }
  | { kind: "choice"; letter: string }
  | null;

type CardField = Field & { varying: boolean };

type FeedbackDraft = {
  confidence: ConfidenceRating | null;
  decision: AuditDecision | null;
  comment: string;
  submitted: boolean;
};

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
  const results = useRef<Record<string, { correct: boolean; picked: string }>>({});
  const feedbackByQuestion = useRef<Record<string, FeedbackDraft>>({});
  const [feedback, setFeedback] = useState<FeedbackDraft>(EMPTY_FEEDBACK);
  const [, bump] = useState(0);

  useEffect(() => {
    fetch("/data/questions.json")
      .then((response) => {
        if (!response.ok) {
          throw new Error("Missing baked questions. Run: python tools/export_quiz_static.py");
        }
        return response.json();
      })
      .then((data: BakeFile) => setBake(data))
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

  useEffect(() => {
    if (screen !== "quiz" || !question) {
      return;
    }
    const prior = results.current[question.id];
    const already = prior !== undefined;
    setStage(already ? "reveal" : "observe");
    setSelected(prior?.picked ?? null);
    setAnswered(already);
    setInfo(null);
    viewStartedAt.current = Date.now();
    track({
      session_id: sessionId.current,
      event_type: "question_view",
      question_id: question.id
    });
  }, [screen, question?.id]);

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
      results: results.current,
      audit_feedback: feedbackByQuestion.current
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
    ensureSessionStart();
    setIndex(atIndex);
    setScreen("quiz");
  }

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
    results.current[question.id] = { correct, picked: letter };
    setSelected(letter);
    setAnswered(true);
    bump((n) => n + 1);
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
      payload: { from: "compare", to: "reveal" }
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
    setStage("compare");
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

  if (screen === "home") {
    return (
      <HomeScreen
        ready={Boolean(bake)}
        onBegin={() => beginQuiz(0)}
        onMenu={() => setScreen("menu")}
        onContact={() => setScreen("contact")}
      />
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

      <section className="stage-screen" key={`${question.id}-${stage}`}>
        <div className="provenance" aria-label="Question provenance">
          <span>Track: {question.track ?? "default"}</span>
          <span>Profile: {question.profile ?? "legacy/unknown"}</span>
          <span>Hash: {question.profileHash ?? "legacy/unknown"}</span>
        </div>
        {stage === "observe" ? (
          <DatasetStage
            question={question}
            onSeeChoices={goCompare}
            onInfo={() => setInfo({ kind: "dataset" })}
          />
        ) : null}
        {stage === "compare" ? (
          <ChoicesStage
            question={question}
            onPick={pickChoice}
            onInfo={(letter) => setInfo({ kind: "choice", letter })}
          />
        ) : null}
        {stage === "reveal" ? (
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
        ) : null}
      </section>

      {info ? (
        <InfoModal
          question={question}
          target={info}
          answered={answered}
          onClose={() => setInfo(null)}
        />
      ) : null}
    </main>
  );
}

function HomeScreen({
  ready,
  onBegin,
  onMenu,
  onContact
}: {
  ready: boolean;
  onBegin: () => void;
  onMenu: () => void;
  onContact: () => void;
}) {
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
        </div>
      </div>
    </main>
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

function TaskDescription({ question }: { question: BakedQuestion }) {
  const params = question.detail.dataset.params ?? {};
  const metric = humanMetric(question.metric);
  const train = params.train_size != null ? String(params.train_size) : "—";
  const test = params.test_size != null ? String(params.test_size) : "—";
  let summary: string;
  if (question.family === "synthetic_tabular_classification") {
    const rule = String(params.rule_family ?? "synthetic rule").replace(/_/g, " ");
    const active = Array.isArray(params.active_features)
      ? params.active_features.map((value) => `x_${String(value)}`).join(", ")
      : "the active features";
    summary = `Predict one of ${params.num_classes ?? 2} classes from ${params.input_dim ?? "N"}-dimensional tabular features. Labels follow a ${rule} rule using ${active}; the held-out selection metric is ${metric} (lower is better). The dataset has ${train} training rows and ${test} test rows.`;
  } else if (question.family === "bigram_lm") {
    summary = `Predict the next token in a synthetic bigram language model with vocabulary size ${params.vocab_size ?? "—"} and context length ${params.context_length ?? "—"}. Compare held-out ${metric} after the stated training budget (lower is better).`;
  } else if (question.family === "multivariate_regression") {
    summary = `Fit a scalar target from ${params.input_dim ?? "multiple"}-dimensional inputs. Compare held-out ${metric} after the stated training budget (lower is better); all choices use the same materialized dataset.`;
  } else {
    summary = `Fit a scalar regression target from one-dimensional inputs. Compare held-out ${metric} after the stated training budget (lower is better); all choices use the same materialized dataset.`;
  }
  return (
    <div className="panel task-description">
      <div className="panel-head"><p className="stage-kicker">Task description</p></div>
      <p className="task-summary">{summary}</p>
      <details>
        <summary>Show full benchmark instructions</summary>
        <pre className="prompt-copy">{question.detail.prompt}</pre>
      </details>
    </div>
  );
}

function DatasetStage({
  question,
  onSeeChoices,
  onInfo
}: {
  question: BakedQuestion;
  onSeeChoices: () => void;
  onInfo: () => void;
}) {
  const params = question.detail.dataset.params ?? {};
  return (
    <div className="stage-inner">
      <TaskDescription question={question} />
      <div className="panel dataset-panel">
        <div className="panel-head">
          <p className="stage-kicker">Dataset</p>
          <button type="button" className="ghost-info" onClick={onInfo} aria-label="Dataset files">
            i
          </button>
        </div>
        <div className="dataset-layout">
          <dl className="attr-list">
            <div>
              <dt>Family</dt>
              <dd>{humanFamily(question.family)}</dd>
            </div>
            {params.expression != null ? (
              <div>
                <dt>Target expression</dt>
                <dd className="mono">{String(params.expression)}</dd>
              </div>
            ) : null}
            {params.input_dim != null ? (
              <div>
                <dt>Input dim</dt>
                <dd>{String(params.input_dim)}</dd>
              </div>
            ) : null}
            {params.domain != null ? (
              <div>
                <dt>Domain</dt>
                <dd className="mono">{formatParam(params.domain)}</dd>
              </div>
            ) : null}
            {params.vocab_size != null ? (
              <div>
                <dt>Vocab size</dt>
                <dd>{String(params.vocab_size)}</dd>
              </div>
            ) : null}
            {params.context_length != null ? (
              <div>
                <dt>Context length</dt>
                <dd>{String(params.context_length)}</dd>
              </div>
            ) : null}
            {params.train_size != null ? (
              <div>
                <dt>Train / test size</dt>
                <dd>
                  {String(params.train_size)} / {String(params.test_size ?? "—")}
                </dd>
              </div>
            ) : null}
            {params.noise != null ? (
              <div>
                <dt>Noise</dt>
                <dd className="mono">{formatParam(params.noise)}</dd>
              </div>
            ) : null}
            {question.detail.dataset.example ? (
              <div>
                <dt>Example</dt>
                <dd className="mono example-io">
                  <div>
                    <span className="io-label">in</span>{" "}
                    {formatParam(question.detail.dataset.example.input)}
                  </div>
                  <div>
                    <span className="io-label">out</span>{" "}
                    {formatParam(question.detail.dataset.example.output)}
                  </div>
                </dd>
              </div>
            ) : null}
          </dl>
          <DatasetVisual question={question} />
        </div>
      </div>
      <div className="stage-footer">
        <button type="button" className="cta" onClick={onSeeChoices}>
          See choices →
        </button>
      </div>
    </div>
  );
}

function ChoicesStage({
  question,
  onPick,
  onInfo
}: {
  question: BakedQuestion;
  onPick: (letter: string) => void;
  onInfo: (letter: string) => void;
}) {
  return (
    <div className="stage-inner">
      <p className="stage-kicker">Choices</p>
      <p className="hint">Tap a card to lock that answer. Emphasized rows differ across choices.</p>
      <div className="choice-grid">
        {question.detail.choices.map((choice) => (
          <ChoiceCard
            key={choice.letter}
            choice={choice}
            fields={fieldsForChoice(question, choice)}
            interactive
            onPick={() => onPick(choice.letter)}
            onInfo={() => onInfo(choice.letter)}
          />
        ))}
      </div>
    </div>
  );
}

function AnswerStage({
  question,
  selected,
  feedback,
  onFeedbackChange,
  onSubmitFeedback,
  onNext,
  onInfo,
  onDatasetInfo
}: {
  question: BakedQuestion;
  selected: string | null;
  feedback: FeedbackDraft;
  onFeedbackChange: (
    patch: Partial<Pick<FeedbackDraft, "confidence" | "decision" | "comment">>
  ) => void;
  onSubmitFeedback: () => void;
  onNext: () => void;
  onInfo: (letter: string) => void;
  onDatasetInfo: () => void;
}) {
  const correct = question.reveal.correctLetter;
  const pickedOk = selected === correct;
  const byLetter = Object.fromEntries(question.reveal.ranked.map((row) => [row.letter, row]));

  return (
    <div className="stage-inner">
      <div className="panel-head">
        <p className="stage-kicker">Answer</p>
        <button type="button" className="ghost-info" onClick={onDatasetInfo} aria-label="Dataset files">
          i
        </button>
      </div>
      <p className={`verdict ${pickedOk ? "ok" : "bad"}`}>
        {selected
          ? pickedOk
            ? `Correct — ${correct} is best on ${humanMetric(question.metric)}.`
            : `You picked ${selected}. Correct is ${correct}.`
          : `Correct choice: ${correct}.`}
      </p>
      <div className="choice-grid">
        {question.detail.choices.map((choice) => {
          const row = byLetter[choice.letter];
          return (
            <ChoiceCard
              key={choice.letter}
              choice={choice}
              fields={fieldsForChoice(question, choice)}
              interactive={false}
              correct={choice.letter === correct}
              wrongPick={Boolean(selected && choice.letter === selected && choice.letter !== correct)}
              metricText={
                row ? formatMetric(row.mean, row.std, row.metric) : "unavailable"
              }
              onInfo={() => onInfo(choice.letter)}
            />
          );
        })}
      </div>
      <CurvesPlot question={question} />
      <AuditFeedbackPanel feedback={feedback} onChange={onFeedbackChange} onSubmit={onSubmitFeedback} />
      <div className="stage-footer">
        <p className="hint">Continue when you are ready.</p>
        <button type="button" className="cta" onClick={onNext}>
          Next question →
        </button>
      </div>
    </div>
  );
}

function AuditFeedbackPanel({
  feedback,
  onChange,
  onSubmit
}: {
  feedback: FeedbackDraft;
  onChange: (
    patch: Partial<Pick<FeedbackDraft, "confidence" | "decision" | "comment">>
  ) => void;
  onSubmit: () => void;
}) {
  return (
    <section className="audit-feedback panel" aria-label="Question audit feedback">
      <div className="panel-head">
        <div>
          <p className="stage-kicker">Audit feedback</p>
          <p className="hint">How confident are you, and should this question stay in the collection?</p>
        </div>
        {feedback.submitted ? <span className="feedback-saved">Saved</span> : null}
      </div>
      <div className="feedback-group">
        <span className="feedback-label">Confidence</span>
        <div className="feedback-options" role="group" aria-label="Confidence from 1 to 5">
          {([1, 2, 3, 4, 5] as const).map((value) => (
            <button
              key={value}
              type="button"
              className={feedback.confidence === value ? "selected" : ""}
              aria-pressed={feedback.confidence === value}
              disabled={feedback.submitted}
              onClick={() => onChange({ confidence: value })}
            >
              {value}
            </button>
          ))}
        </div>
      </div>
      <div className="feedback-group">
        <span className="feedback-label">Quality decision</span>
        <div className="feedback-options disposition-options" role="group" aria-label="Question quality">
          {(["keep", "revise", "reject"] as const).map((value) => (
            <button
              key={value}
              type="button"
              className={`${value}${feedback.decision === value ? " selected" : ""}`}
              aria-pressed={feedback.decision === value}
              disabled={feedback.submitted}
              onClick={() => onChange({ decision: value })}
            >
              {value[0].toUpperCase() + value.slice(1)}
            </button>
          ))}
        </div>
      </div>
      <label className="feedback-comment">
        <span className="feedback-label">Comment</span>
        <textarea
          value={feedback.comment}
          disabled={feedback.submitted}
          maxLength={2000}
          rows={3}
          placeholder="What should be clarified or changed?"
          onChange={(event) => onChange({ comment: event.target.value })}
        />
      </label>
      <button
        type="button"
        className="feedback-save"
        disabled={feedback.submitted || feedback.confidence === null || feedback.decision === null}
        onClick={onSubmit}
      >
        Save feedback
      </button>
    </section>
  );
}

function ChoiceCard({
  choice,
  fields,
  interactive,
  onPick,
  onInfo,
  correct,
  wrongPick,
  metricText
}: {
  choice: Choice;
  fields: CardField[];
  interactive: boolean;
  onPick?: () => void;
  onInfo: () => void;
  correct?: boolean;
  wrongPick?: boolean;
  metricText?: string;
}) {
  const className = [
    "choice-card",
    correct ? "correct" : "",
    wrongPick ? "wrong" : ""
  ]
    .filter(Boolean)
    .join(" ");

  return (
    <div
      className={className}
      style={{ "--choice": choice.color } as React.CSSProperties}
      role={interactive ? "button" : undefined}
      tabIndex={interactive ? 0 : undefined}
      onClick={interactive ? onPick : undefined}
      onKeyDown={
        interactive
          ? (event) => {
              if (event.key === "Enter" || event.key === " ") {
                event.preventDefault();
                onPick?.();
              }
            }
          : undefined
      }
    >
      <button
        type="button"
        className="ghost-info on-card"
        aria-label={`Files for choice ${choice.letter}`}
        onClick={(event) => {
          event.stopPropagation();
          onInfo();
        }}
      >
        i
      </button>
      <span className="choice-letter">{choice.letter}</span>
      {metricText ? <div className="choice-metric">{metricText}</div> : null}
      <div className="choice-fields">
        {fields.map((field) => (
          <div key={field.label} className={field.varying ? "field vary" : "field same"}>
            <span>{titleCase(field.label)}</span>
            <strong>{formatFieldValue(field.label, field.value)}</strong>
          </div>
        ))}
      </div>
    </div>
  );
}

function fieldsForChoice(question: BakedQuestion, choice: Choice): CardField[] {
  const shared = question.detail.shared.map((field) => ({ ...field, varying: false }));
  const variant = choice.variant.map((field) => ({ ...field, varying: true }));
  const parameterLabel = "trainable parameter count";
  const hasParameterField = [...shared, ...variant].some(
    (field) => field.label.toLowerCase().replace(/_/g, " ") === parameterLabel
  );
  if (!hasParameterField) {
    const parameterValues = question.detail.choices.map((item) => trainableParameterCount(item));
    const parameterValue = trainableParameterCount(choice);
    const allEqual = parameterValues.every((value) => value === parameterValues[0]);
    const field = { label: parameterLabel, value: parameterValue, varying: !allEqual };
    if (allEqual) {
      shared.push(field);
    } else {
      variant.push(field);
    }
  }
  // Keep a stable key order: shared keys first (as baked), then varying keys.
  const seen = new Set(shared.map((field) => field.label));
  const extra = variant.filter((field) => !seen.has(field.label));
  return [...shared, ...extra];
}

function trainableParameterCount(choice: Choice): string {
  const raw = choice.files?.["candidate_spec.json"];
  const spec =
    raw && typeof raw === "object" && !Array.isArray(raw)
      ? (raw as Record<string, unknown>)
      : typeof raw === "string"
        ? parseCandidateSpec(raw)
        : null;
  const count = spec?.trainable_parameter_count;
  return count == null || count === "" ? "—" : String(count);
}

function parseCandidateSpec(raw: string): Record<string, unknown> | null {
  try {
    const parsed: unknown = JSON.parse(raw);
    return parsed && typeof parsed === "object" && !Array.isArray(parsed)
      ? (parsed as Record<string, unknown>)
      : null;
  } catch {
    return null;
  }
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

function DatasetVisual({ question }: { question: BakedQuestion }) {
  const plot = question.detail.dataset.plot;
  if (!plot || plot.kind === "none") {
    return null;
  }
  if (plot.kind === "classification") {
    return <ClassificationPlot plot={plot} />;
  }
  if (plot.kind === "heatmap" && plot.matrix) {
    return (
      <Heatmap
        matrix={plot.matrix}
        xLabel={plot.xLabel ?? "next token"}
        yLabel={plot.yLabel ?? "current token"}
        legend={plot.legend ?? "probability"}
        min={plot.min}
        max={plot.max}
      />
    );
  }
  const params = question.detail.dataset.params ?? {};
  const trainCount =
    typeof params.train_size === "number" ? params.train_size : (plot.train?.length ?? 0);
  const testCount =
    typeof params.test_size === "number" ? params.test_size : (plot.test?.length ?? 0);
  return (
    <Scatter
      train={plot.train ?? []}
      test={plot.test ?? []}
      trainCount={trainCount}
      testCount={testCount}
    />
  );
}

function Scatter({
  train,
  test,
  trainCount,
  testCount
}: {
  train: Point[];
  test: Point[];
  trainCount: number;
  testCount: number;
}) {
  const all = [...train, ...test];
  if (!all.length) {
    return null;
  }
  const width = 560;
  const height = 260;
  const plot = { x: 48, y: 18, width: 480, height: 190 };
  const domain = pointDomain(all);
  const xTicks = makeTicks(domain.xMin, domain.xMax, 6);
  const yTicks = makeTicks(domain.yMin, domain.yMax, 5);
  const pos = (point: Point) => ({
    x: plot.x + ((point.x - domain.xMin) / (domain.xMax - domain.xMin || 1)) * plot.width,
    y: plot.y + plot.height - ((point.y - domain.yMin) / (domain.yMax - domain.yMin || 1)) * plot.height
  });
  return (
    <div className="viz">
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Dataset scatter">
        <rect x={plot.x} y={plot.y} width={plot.width} height={plot.height} fill="#1a1d24" />
        {xTicks.map((tick) => {
          const x = plot.x + ((tick - domain.xMin) / (domain.xMax - domain.xMin || 1)) * plot.width;
          return (
            <g key={`x-${tick}`}>
              <line x1={x} x2={x} y1={plot.y} y2={plot.y + plot.height} stroke="#2a2e38" />
              <text x={x} y={plot.y + plot.height + 22} textAnchor="middle" fill="#8b919f" fontSize="11">
                {formatTick(tick)}
              </text>
            </g>
          );
        })}
        {yTicks.map((tick) => {
          const y =
            plot.y + plot.height - ((tick - domain.yMin) / (domain.yMax - domain.yMin || 1)) * plot.height;
          return (
            <g key={`y-${tick}`}>
              <line x1={plot.x} x2={plot.x + plot.width} y1={y} y2={y} stroke="#2a2e38" />
              <text x={plot.x - 10} y={y + 4} textAnchor="end" fill="#8b919f" fontSize="11">
                {formatTick(tick)}
              </text>
            </g>
          );
        })}
        <text x={plot.x} y={height - 8} fill="#8b919f" fontSize="12">
          x · train {trainCount} · test {testCount}
        </text>
        <text
          x={18}
          y={plot.y + plot.height / 2}
          fill="#8b919f"
          fontSize="12"
          transform={`rotate(-90 18 ${plot.y + plot.height / 2})`}
        >
          y
        </text>
        {train.map((point, i) => {
          const p = pos(point);
          return <circle key={`tr-${i}`} cx={p.x} cy={p.y} r="3.2" fill="#8b7cff" opacity="0.85" />;
        })}
        {test.map((point, i) => {
          const p = pos(point);
          return <circle key={`te-${i}`} cx={p.x} cy={p.y} r="3.2" fill="#3dcf9a" opacity="0.85" />;
        })}
      </svg>
    </div>
  );
}

function ClassificationPlot({
  plot
}: {
  plot: NonNullable<BakedQuestion["detail"]["dataset"]["plot"]>;
}) {
  const train = (plot.train ?? []) as Array<Point & { label?: number }>;
  const test = (plot.test ?? []) as Array<Point & { label?: number }>;
  const all = [...train, ...test].filter((point) => Number.isFinite(point.x) && Number.isFinite(point.y));
  if (!all.length) {
    return <p className="hint">Classification projection unavailable for this dataset.</p>;
  }
  const fallback = pointDomain(all);
  const xEdges = plot.xEdges && plot.xEdges.length > 1 ? plot.xEdges : [fallback.xMin, fallback.xMax];
  const yEdges = plot.yEdges && plot.yEdges.length > 1 ? plot.yEdges : [fallback.yMin, fallback.yMax];
  const xMin = xEdges[0];
  const xMax = xEdges[xEdges.length - 1];
  const yMin = yEdges[0];
  const yMax = yEdges[yEdges.length - 1];
  const width = 560;
  const height = 310;
  const chart = { x: 52, y: 18, width: 470, height: 230 };
  const mapX = (value: number) => chart.x + ((value - xMin) / (xMax - xMin || 1)) * chart.width;
  const mapY = (value: number) => chart.y + chart.height - ((value - yMin) / (yMax - yMin || 1)) * chart.height;
  const xTicks = makeTicks(xMin, xMax, 5);
  const yTicks = makeTicks(yMin, yMax, 5);
  const probability = plot.probability ?? [];
  const observedLabels = Array.from(
    new Set(
      [...train, ...test]
        .map((point) => point.label)
        .filter((label): label is number => label != null && Number.isFinite(label))
    )
  ).sort((a, b) => a - b);
  const legendLabels = observedLabels.length ? observedLabels : [0, 1];
  const classPalette = ["#1d4ed8", "#b91c1c", "#15803d", "#7e22ce", "#c2410c", "#0f766e"];
  const classColor = (label: number | undefined) => {
    const index = legendLabels.indexOf(label ?? legendLabels[0]);
    return classPalette[(index < 0 ? 0 : index) % classPalette.length];
  };
  const trainPoints = train.filter((point) => Number.isFinite(point.x) && Number.isFinite(point.y));
  const testPoints = test.filter((point) => Number.isFinite(point.x) && Number.isFinite(point.y));
  return (
    <div className="viz">
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Synthetic classification projection">
        <rect x={chart.x} y={chart.y} width={chart.width} height={chart.height} fill="#1a1d24" />
        {probability.map((row, x) => row.map((value, y) => {
          const x0 = xEdges[x];
          const x1 = xEdges[x + 1];
          const y0 = yEdges[y];
          const y1 = yEdges[y + 1];
          if (
            !Number.isFinite(value) ||
            x0 == null || x1 == null || y0 == null || y1 == null ||
            !Number.isFinite(x0) || !Number.isFinite(x1) ||
            !Number.isFinite(y0) || !Number.isFinite(y1)
          ) {
            return null;
          }
          const alpha = 0.08 + Math.min(1, Math.max(0, value)) * 0.7;
          const fill = value >= 0.5 ? `rgba(185,28,28,${alpha})` : `rgba(29,78,216,${1.0 - alpha * 0.35})`;
          return (
            <rect key={`prob-${x}-${y}`} x={mapX(x0)} y={mapY(y1)}
              width={Math.max(0, mapX(x1) - mapX(x0))}
              height={Math.max(0, mapY(y0) - mapY(y1))}
              fill={fill} />
          );
        }))}
        {xTicks.map((tick) => <g key={`x-${tick}`}><line x1={mapX(tick)} x2={mapX(tick)} y1={chart.y} y2={chart.y + chart.height} stroke="#2a2e38" /><text x={mapX(tick)} y={chart.y + chart.height + 18} textAnchor="middle" fill="#8b919f" fontSize="10">{formatTick(tick)}</text></g>)}
        {yTicks.map((tick) => <g key={`y-${tick}`}><line x1={chart.x} x2={chart.x + chart.width} y1={mapY(tick)} y2={mapY(tick)} stroke="#2a2e38" /><text x={chart.x - 8} y={mapY(tick) + 3} textAnchor="end" fill="#8b919f" fontSize="10">{formatTick(tick)}</text></g>)}
        {trainPoints.map((point, i) => <circle key={`train-${i}`} cx={mapX(point.x)} cy={mapY(point.y)} r="3" fill={classColor(point.label)} opacity="0.48" />)}
        {testPoints.map((point, i) => <circle key={`test-${i}`} cx={mapX(point.x)} cy={mapY(point.y)} r="4" fill="none" stroke={classColor(point.label)} strokeWidth="1.8" opacity="0.9" />)}
        <text x={chart.x + chart.width / 2} y={height - 25} textAnchor="middle" fill="#8b919f" fontSize="11">{plot.xLabel ?? "feature x"}</text>
        <text x="14" y={chart.y + chart.height / 2} textAnchor="middle" fill="#8b919f" fontSize="11" transform={`rotate(-90 14 ${chart.y + chart.height / 2})`}>{plot.yLabel ?? "feature y"}</text>
        {legendLabels.map((label, index) => {
          const x = chart.x + 8 + index * 110;
          return (
            <g key={`legend-${label}`}>
              <circle cx={x} cy={height - 8} r="4" fill={classColor(label)} />
              <text x={x + 10} y={height - 4} fill="#c5c9d4" fontSize="10">{`class ${label} · train`}</text>
            </g>
          );
        })}
        <text
          x={legendLabels.length <= 2 ? chart.x + 265 : chart.x + 8 + legendLabels.length * 110 + 8}
          y={height - 4}
          fill="#8b919f"
          fontSize="10"
        >
          test outlined · {plot.selectionNote ?? "rule-aware feature pair"}
        </text>
      </svg>
    </div>
  );
}

function Heatmap({
  matrix,
  xLabel,
  yLabel,
  legend,
  min,
  max
}: {
  matrix: number[][];
  xLabel: string;
  yLabel: string;
  legend: string;
  min?: number;
  max?: number;
}) {
  const rows = matrix.length;
  const cols = matrix[0]?.length ?? 0;
  if (!rows || !cols) {
    return null;
  }
  const flat = matrix.flat().filter((value) => Number.isFinite(value));
  const lo = min ?? Math.min(...flat, 0);
  const hi = max ?? Math.max(...flat, 1);
  const cell = rows > 24 ? 8 : 12;
  const padL = 46;
  const padT = 28;
  const padR = 58;
  const padB = 42;
  const gridW = cols * cell;
  const gridH = rows * cell;
  const width = padL + gridW + padR;
  const height = padT + gridH + padB;
  const tickStep = Math.max(1, Math.floor(Math.max(rows, cols) / 4));
  const norm = (value: number) => (hi === lo ? 0.5 : (value - lo) / (hi - lo));

  return (
    <div className="viz">
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Transition matrix">
        {matrix.map((row, y) =>
          row.map((value, x) => {
            const t = Math.min(1, Math.max(0, norm(value)));
            return (
              <rect
                key={`${x}-${y}`}
                x={padL + x * cell}
                y={padT + y * cell}
                width={cell - 0.6}
                height={cell - 0.6}
                fill={`rgba(139,124,255,${0.08 + t * 0.92})`}
              />
            );
          })
        )}
        {Array.from({ length: Math.floor((cols - 1) / tickStep) + 1 }, (_, i) => i * tickStep).map(
          (tick) => (
            <text
              key={`xt-${tick}`}
              x={padL + tick * cell + cell / 2}
              y={padT + gridH + 16}
              textAnchor="middle"
              fill="#8b919f"
              fontSize="10"
            >
              {tick}
            </text>
          )
        )}
        {Array.from({ length: Math.floor((rows - 1) / tickStep) + 1 }, (_, i) => i * tickStep).map(
          (tick) => (
            <text
              key={`yt-${tick}`}
              x={padL - 8}
              y={padT + tick * cell + cell / 2 + 3}
              textAnchor="end"
              fill="#8b919f"
              fontSize="10"
            >
              {tick}
            </text>
          )
        )}
        <text
          x={padL + gridW / 2}
          y={height - 8}
          textAnchor="middle"
          fill="#8b919f"
          fontSize="11"
        >
          {xLabel}
        </text>
        <text
          x={14}
          y={padT + gridH / 2}
          textAnchor="middle"
          fill="#8b919f"
          fontSize="11"
          transform={`rotate(-90 14 ${padT + gridH / 2})`}
        >
          {yLabel}
        </text>
        {/* color legend */}
        {Array.from({ length: 48 }, (_, i) => {
          const t = i / 47;
          return (
            <rect
              key={`leg-${i}`}
              x={padL + gridW + 14}
              y={padT + (1 - t) * gridH}
              width={10}
              height={gridH / 47 + 0.5}
              fill={`rgba(139,124,255,${0.08 + t * 0.92})`}
            />
          );
        })}
        <text x={padL + gridW + 28} y={padT + 8} fill="#8b919f" fontSize="10">
          {formatTick(hi)}
        </text>
        <text x={padL + gridW + 28} y={padT + gridH} fill="#8b919f" fontSize="10">
          {formatTick(lo)}
        </text>
        <text
          x={padL + gridW + 18}
          y={padT - 10}
          textAnchor="middle"
          fill="#8b919f"
          fontSize="10"
        >
          {legend}
        </text>
      </svg>
    </div>
  );
}

function CurvesPlot({ question }: { question: BakedQuestion }) {
  const curves = question.reveal.curves;
  const width = 920;
  const height = 380;
  const plot = { x: 72, y: 40, width: 780, height: 280 };
  const allY = curves.flatMap((series) =>
    series.mean.filter((value): value is number => Number.isFinite(value))
  );
  const allX = curves.flatMap((series) => series.samples);
  if (!curves.length || !allY.length || !allX.length) {
    return <p className="hint">Learning curves unavailable for this question.</p>;
  }
  const xMin = Math.min(...allX);
  const xMax = Math.max(...allX);
  const yMin = Math.min(...allY);
  const yMax = Math.max(...allY);
  const yPad = Math.max((yMax - yMin) * 0.12, 1e-6);
  const yLo = yMin - yPad;
  const yHi = yMax + yPad;
  const xTicks = makeTicks(xMin, xMax, 6);
  const yTicks = makeTicks(yLo, yHi, 5);
  const colorFor = (letter: string) =>
    question.detail.choices.find((choice) => choice.letter === letter)?.color ?? "#ccc";
  const mapX = (x: number) => plot.x + ((x - xMin) / (xMax - xMin || 1)) * plot.width;
  const mapY = (y: number) => plot.y + plot.height - ((y - yLo) / (yHi - yLo || 1)) * plot.height;
  const metric = humanMetric(question.metric);

  return (
    <div className="viz">
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Ground-truth learning curves">
        <rect x={plot.x} y={plot.y} width={plot.width} height={plot.height} fill="#1a1d24" />
        {xTicks.map((tick) => {
          const x = mapX(tick);
          return (
            <g key={`cx-${tick}`}>
              <line x1={x} x2={x} y1={plot.y} y2={plot.y + plot.height} stroke="#2a2e38" />
              <text x={x} y={plot.y + plot.height + 22} textAnchor="middle" fill="#8b919f" fontSize="11">
                {formatTick(tick)}
              </text>
            </g>
          );
        })}
        {yTicks.map((tick) => {
          const y = mapY(tick);
          return (
            <g key={`cy-${tick}`}>
              <line x1={plot.x} x2={plot.x + plot.width} y1={y} y2={y} stroke="#2a2e38" />
              <text x={plot.x - 10} y={y + 4} textAnchor="end" fill="#8b919f" fontSize="11">
                {formatTick(tick)}
              </text>
            </g>
          );
        })}
        <text x={plot.x + plot.width / 2} y={height - 8} textAnchor="middle" fill="#8b919f" fontSize="12">
          samples seen
        </text>
        <text
          x={18}
          y={plot.y + plot.height / 2}
          fill="#8b919f"
          fontSize="12"
          transform={`rotate(-90 18 ${plot.y + plot.height / 2})`}
        >
          {metric}
        </text>
        {curves.map((series) => {
          const coords = series.samples
            .map((sample, i) => ({ sample, value: series.mean[i] }))
            .filter((point) => Number.isFinite(point.value));
          if (!coords.length) {
            return null;
          }
          const path = coords
            .map((point, i) => `${i === 0 ? "M" : "L"} ${mapX(point.sample)} ${mapY(point.value)}`)
            .join(" ");
          return (
            <path
              key={series.letter}
              d={path}
              fill="none"
              stroke={colorFor(series.letter)}
              strokeWidth="2.75"
            />
          );
        })}
        {question.detail.choices.map((choice, i) => (
          <g key={choice.letter} transform={`translate(${80 + i * 72} 24)`}>
            <circle cx="0" cy="0" r="5" fill={choice.color} />
            <text x="10" y="4" fill="#c5c9d4" fontSize="13" fontWeight="700">
              {choice.letter}
            </text>
          </g>
        ))}
      </svg>
    </div>
  );
}

function formatNumber(value: number, digits = 2): string {
  if (!Number.isFinite(value)) return "—";
  const abs = Math.abs(value);
  if (abs !== 0 && abs < 10 ** -digits) {
    return value.toExponential(Math.max(0, digits - 1)).replace(/\.?0+e/, "e");
  }
  return value.toFixed(digits).replace(/\.?0+$/, "");
}

function formatParam(value: unknown): string {
  if (value == null) return "—";
  if (typeof value === "number") return formatNumber(value);
  if (typeof value === "boolean" || typeof value === "string") return String(value);
  if (Array.isArray(value)) return `[${value.map(formatParam).join(", ")}]`;
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

function formatFieldValue(label: string, value: string): string {
  const trimmed = value.trim();
  if (!trimmed || trimmed === "—") return value;
  const numeric = Number(trimmed);
  if (!Number.isFinite(numeric) || !/^[+-]?(?:\d+\.?\d*|\.\d+)(?:e[+-]?\d+)?$/i.test(trimmed)) {
    return value;
  }
  const lower = label.toLowerCase();
  if (Number.isInteger(numeric) || /count|steps|samples|size|width|depth|layers|heads|grid|order/.test(lower)) {
    return Number.isInteger(numeric) ? numeric.toLocaleString() : formatNumber(numeric);
  }
  return formatNumber(numeric);
}

function humanFamily(family?: string) {
  if (!family) return "Dataset";
  return family.replace(/_/g, " ");
}

function humanMetric(metric?: string) {
  if (!metric) return "selection metric";
  if (metric === "test_mse") return "test MSE";
  if (metric === "test_ce") return "test cross-entropy";
  return metric.replace(/_/g, " ");
}

function humanMetricByFamily(family?: string, metric?: string) {
  if (metric) return humanMetric(metric);
  if (family === "bigram_lm") return "test CE";
  return "test MSE";
}

function humanType(type?: string) {
  if (!type) return "mixed";
  return type.replace(/_/g, " ");
}

function titleCase(text: string) {
  return text.replace(/\b\w/g, (char) => char.toUpperCase());
}

function formatMetric(mean: number | null, std: number | null, metric: string) {
  if (mean == null || !Number.isFinite(mean)) {
    return "unavailable";
  }
  const unit = humanMetric(metric);
  if (std == null || !Number.isFinite(std)) {
    return `${formatNumber(mean)} (${unit})`;
  }
  return `${formatNumber(mean)} ± ${formatNumber(std)}`;
}

function pointDomain(points: Point[]) {
  const xs = points.map((p) => p.x).filter(Number.isFinite);
  const ys = points.map((p) => p.y).filter(Number.isFinite);
  const xMin = Math.min(...xs, 0);
  const xMax = Math.max(...xs, 1);
  const yMin = Math.min(...ys, 0);
  const yMax = Math.max(...ys, 1);
  const xPad = Math.max((xMax - xMin) * 0.08, 0.1);
  const yPad = Math.max((yMax - yMin) * 0.12, 0.1);
  return { xMin: xMin - xPad, xMax: xMax + xPad, yMin: yMin - yPad, yMax: yMax + yPad };
}

function makeTicks(min: number, max: number, count: number) {
  if (count <= 1) return [min];
  return Array.from({ length: count }, (_, i) => min + ((max - min) * i) / (count - 1));
}

function formatTick(value: number) {
  const abs = Math.abs(value);
  if (abs >= 100 || abs === 0) return value.toFixed(0);
  if (abs >= 10) return value.toFixed(1);
  return value.toFixed(2).replace(/\.?0+$/, "");
}

createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
