import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

type Field = {
  label: string;
  value: string;
};

type Point = {
  x: number;
  y: number;
};

type QuestionSummary = {
  id: string;
  index: number;
  type: string;
  datasetId: string;
  budget: number;
  choices: number;
};

type Choice = {
  letter: string;
  candidateId: string;
  variant: Field[];
};

type Question = {
  id: string;
  title: string;
  family: string;
  datasetId: string;
  type: string;
  profile: string;
  budget: Record<string, unknown>;
  metric: string;
  evaluation: Record<string, unknown>;
  invariantAxes: string[];
  varyingAxes: string[];
  prompt: string;
  dataset: {
    train: Point[];
    test: Point[];
  };
  shared: Field[];
  choices: Choice[];
};

type RankedRow = {
  letter: string;
  candidateId: string;
  metric: string;
  mean: number | null;
  std: number | null;
  label: string;
};

type AnswerResult = {
  picked: string;
  correctLetter: string;
  correct: boolean;
  ranked: RankedRow[];
};

type EvidenceTab = "matrix" | "protocol" | "prompt";

const choiceColors = ["#734cff", "#f26e4f", "#20a87e", "#2f7de1"];

function App() {
  const [started, setStarted] = useState(false);
  const [questions, setQuestions] = useState<QuestionSummary[]>([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [question, setQuestion] = useState<Question | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [answer, setAnswer] = useState<AnswerResult | null>(null);
  const [score, setScore] = useState({ correct: 0, total: 0 });
  const [tab, setTab] = useState<EvidenceTab>("matrix");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch("/api/questions")
      .then((response) => response.json())
      .then((data) => {
        setQuestions(data.questions ?? []);
      })
      .catch((err) => setError(err instanceof Error ? err.message : String(err)));
  }, []);

  useEffect(() => {
    if (!started || questions.length === 0) {
      return;
    }
    const next = questions[currentIndex];
    fetch(`/api/questions/${next.id}`)
      .then((response) => {
        if (!response.ok) {
          throw new Error(`Failed to load ${next.id}`);
        }
        return response.json();
      })
      .then((data) => {
        setQuestion(data);
        setSelected(null);
        setAnswer(null);
        setTab("matrix");
      })
      .catch((err) => setError(err instanceof Error ? err.message : String(err)));
  }, [started, questions, currentIndex]);

  const progress = questions.length ? Math.round(((currentIndex + 1) / questions.length) * 100) : 0;

  async function lockAnswer() {
    if (!question || !selected || answer) {
      return;
    }
    const response = await fetch(`/api/questions/${question.id}/answer`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ letter: selected })
    });
    if (!response.ok) {
      setError("Could not submit answer.");
      return;
    }
    const result = (await response.json()) as AnswerResult;
    setAnswer(result);
    setScore((prev) => ({
      correct: prev.correct + (result.correct ? 1 : 0),
      total: prev.total + 1
    }));
  }

  function nextQuestion() {
    setCurrentIndex((index) => (questions.length ? (index + 1) % questions.length : 0));
  }

  function randomQuestion() {
    if (questions.length <= 1) {
      return;
    }
    let next = currentIndex;
    while (next === currentIndex) {
      next = Math.floor(Math.random() * questions.length);
    }
    setCurrentIndex(next);
  }

  if (!started) {
    return <Welcome onStart={() => setStarted(true)} />;
  }

  return (
    <main className="app-shell">
      <TopBar
        progress={progress}
        current={questions.length ? currentIndex + 1 : 0}
        total={questions.length}
        score={score}
        onNext={nextQuestion}
        onRandom={randomQuestion}
      />

      {error ? <div className="error-banner">{error}</div> : null}
      {!question ? <div className="loading-panel">Loading ArchitectureIQ questions...</div> : null}

      {question ? (
        <>
          <section className="question-hero">
            <p>Observe the evidence</p>
            <h1>{question.title}</h1>
            <div className="chips">
              <span>Question {currentIndex + 1} / {questions.length}</span>
              <span>ID {question.id}</span>
              <span>{question.family?.replace(/_/g, " ")}</span>
              <span>Dataset {question.datasetId}</span>
              <span>Budget {String(question.budget?.total_samples_seen ?? "-")} samples</span>
              <span>{question.choices.length} choices</span>
              <span>{question.metric}</span>
            </div>
          </section>

          <section className="quiz-grid">
            <EvidencePanel question={question} tab={tab} onTab={setTab} reveal={Boolean(answer)} />
            <SharedPanel question={question} />
          </section>

          <section className="bottom-rail" style={{ "--choice-count": question.choices.length } as React.CSSProperties}>
            {question.choices.map((choice, index) => (
              <CandidateCard
                key={choice.letter}
                choice={choice}
                color={choiceColors[index % choiceColors.length]}
                selected={selected === choice.letter}
                answer={answer}
                onSelect={() => {
                  if (!answer) {
                    setSelected(choice.letter);
                  }
                }}
              />
            ))}
            <CommitPanel
              choices={question.choices}
              selected={selected}
              answer={answer}
              onPick={(letter) => {
                if (!answer) {
                  setSelected(letter);
                }
              }}
              onLock={lockAnswer}
            />
          </section>

          {answer ? <ResultPanel answer={answer} /> : null}
        </>
      ) : null}
    </main>
  );
}

function Brand() {
  return (
    <div className="brand">
      <div className="logo" aria-hidden="true">
        <span />
      </div>
      <div>
        <strong>Architecture IQ</strong>
        <small>Read the setup - predict the winner</small>
      </div>
    </div>
  );
}

function Welcome({ onStart }: { onStart: () => void }) {
  return (
    <main className="welcome-page">
      <header className="top-pill">
        <Brand />
        <div className="icon-actions" aria-hidden="true">
          <span>...</span>
          <span className="grid-icon">::</span>
        </div>
      </header>
      <section className="welcome-copy">
        <h1>Test your Architecture IQ</h1>
        <p>Wanna know how wise you are for picking the right setup for training tasks :)?</p>
        <button className="primary-cta" onClick={onStart}>
          Test my architecture IQ <span>→</span>
        </button>
      </section>
      <div className="science-lines" aria-hidden="true" />
    </main>
  );
}

function TopBar({
  progress,
  current,
  total,
  score,
  onNext,
  onRandom
}: {
  progress: number;
  current: number;
  total: number;
  score: { correct: number; total: number };
  onNext: () => void;
  onRandom: () => void;
}) {
  return (
    <header className="top-pill app-top">
      <Brand />
      <div className="progress-block">
        <strong>
          Q {current} / {total}
        </strong>
        <span className="progress-track">
          <span style={{ width: `${progress}%` }} />
        </span>
        <small>{progress}%</small>
      </div>
      <div className="top-actions">
        <button onClick={onNext}>Next question</button>
        <button onClick={onRandom}>Random</button>
        <span className="score-pill">
          Score {score.correct} / {score.total}
        </span>
      </div>
    </header>
  );
}

function EvidencePanel({
  question,
  tab,
  onTab,
  reveal
}: {
  question: Question;
  tab: EvidenceTab;
  onTab: (tab: EvidenceTab) => void;
  reveal: boolean;
}) {
  return (
    <div className="panel evidence-panel">
      <div className="panel-head">
        <div>
          <span className="dot" />
          <strong>Evidence 01</strong>
          <small>Question artifacts</small>
        </div>
        <nav className="tabs" aria-label="Evidence tabs">
          {(["matrix", "protocol", "prompt"] as EvidenceTab[]).map((item) => (
            <button key={item} className={tab === item ? "active" : ""} onClick={() => onTab(item)}>
              {item === "matrix" ? "Matrix" : item[0].toUpperCase() + item.slice(1)}
            </button>
          ))}
        </nav>
      </div>
      {tab === "matrix" ? <ScatterPlot question={question} /> : null}
      {tab === "protocol" ? <Protocol question={question} /> : null}
      {tab === "prompt" ? <pre className="prompt-box">{question.prompt}</pre> : null}
      {!reveal ? <div className="hint-chip">No result data before commitment.</div> : null}
    </div>
  );
}

function ScatterPlot({ question }: { question: Question }) {
  const points = useMemo(() => [...question.dataset.train, ...question.dataset.test], [question]);
  const bounds = useMemo(() => {
    const xs = points.map((point) => point.x);
    const ys = points.map((point) => point.y);
    return {
      minX: Math.min(...xs),
      maxX: Math.max(...xs),
      minY: Math.min(...ys),
      maxY: Math.max(...ys)
    };
  }, [points]);

  const mapX = (x: number) => 50 + ((x - bounds.minX) / Math.max(0.0001, bounds.maxX - bounds.minX)) * 800;
  const mapY = (y: number) => 330 - ((y - bounds.minY) / Math.max(0.0001, bounds.maxY - bounds.minY)) * 270;

  return (
    <div className="plot-frame">
      <svg viewBox="0 0 930 390" role="img" aria-label="Dataset train and test evidence scatter plot">
        <rect x="32" y="24" width="840" height="320" rx="12" fill="#fffefa" stroke="#24262c" strokeWidth="2" />
        {Array.from({ length: 9 }).map((_, index) => (
          <line key={`v-${index}`} x1={70 + index * 90} x2={70 + index * 90} y1="42" y2="326" stroke="#d9d2e9" />
        ))}
        {Array.from({ length: 5 }).map((_, index) => (
          <line key={`h-${index}`} x1="54" x2="848" y1={74 + index * 56} y2={74 + index * 56} stroke="#d9d2e9" />
        ))}
        {question.dataset.train.map((point, index) => (
          <circle key={`train-${index}`} cx={mapX(point.x)} cy={mapY(point.y)} r="4" fill="#734cff" opacity="0.72" />
        ))}
        {question.dataset.test.map((point, index) => (
          <circle key={`test-${index}`} cx={mapX(point.x)} cy={mapY(point.y)} r="4" fill="#20a87e" opacity="0.72" />
        ))}
        <text x="54" y="368" fill="#6e7078" fontSize="14">
          x
        </text>
        <text x="14" y="40" fill="#6e7078" fontSize="14">
          y
        </text>
        <g transform="translate(778 54)">
          <circle cx="0" cy="0" r="5" fill="#734cff" />
          <text x="14" y="5" fill="#4f5158" fontSize="14">
            train
          </text>
          <circle cx="0" cy="24" r="5" fill="#20a87e" />
          <text x="14" y="29" fill="#4f5158" fontSize="14">
            test
          </text>
        </g>
      </svg>
    </div>
  );
}

function Protocol({ question }: { question: Question }) {
  return (
    <div className="protocol-grid">
      <InfoRow label="Question type" value={question.type} />
      <InfoRow label="Profile" value={question.profile} />
      <InfoRow label="Varying axes" value={question.varyingAxes.join(", ") || "-"} />
      <InfoRow label="Invariant axes" value={question.invariantAxes.join(", ") || "-"} />
      <InfoRow label="Evaluation" value={`${String(question.evaluation.n_seeds ?? "-")} seeds`} />
      <InfoRow label="Metric" value={question.metric} />
    </div>
  );
}

function SharedPanel({ question }: { question: Question }) {
  const preferred = ["training steps", "batch size", "total samples seen", "loss", "optimizer", "learning rate"];
  const rows = preferred
    .map((label) => question.shared.find((field) => field.label === label))
    .filter((field): field is Field => Boolean(field));
  if (question.evaluation.n_seeds) {
    rows.push({ label: "evaluation", value: `${question.evaluation.n_seeds} seeds` });
  }

  return (
    <aside className="panel shared-panel">
      <h2>Shared training setup</h2>
      <p>These constraints are identical for every choice. Compare only architecture and optimizer differences.</p>
      <div className="rule" />
      <div className="shared-list">
        {rows.map((row) => (
          <InfoRow key={row.label} label={titleCase(row.label)} value={row.value} />
        ))}
      </div>
      <div className="notice">
        <strong>What should you notice?</strong>
        <span>Reason about capacity, optimizer dynamics, and whether each setup can use the fixed budget efficiently.</span>
      </div>
    </aside>
  );
}

function CandidateCard({
  choice,
  color,
  selected,
  answer,
  onSelect
}: {
  choice: Choice;
  color: string;
  selected: boolean;
  answer: AnswerResult | null;
  onSelect: () => void;
}) {
  const isCorrect = answer?.correctLetter === choice.letter;
  const isWrongPick = answer?.picked === choice.letter && !answer.correct;
  const border = isCorrect ? "#20a87e" : isWrongPick ? "#d94b45" : selected ? color : "#24262c";
  const shown = displayVariant(choice.variant);

  return (
    <button className="candidate-card" style={{ "--accent": color, borderColor: border } as React.CSSProperties} onClick={onSelect}>
      <div className="candidate-top">
        <div>
          <span className="letter">{choice.letter}</span>
          <small>{choice.candidateId}</small>
        </div>
        <span className={`radio-dot ${selected ? "selected" : ""}`} />
      </div>
      <div className="mini-network" aria-hidden="true" />
      <div className="candidate-fields">
        {shown.map((field) => (
          <InfoRow key={field.label} label={titleCase(field.label)} value={field.value} />
        ))}
      </div>
    </button>
  );
}

function CommitPanel({
  choices,
  selected,
  answer,
  onPick,
  onLock
}: {
  choices: Choice[];
  selected: string | null;
  answer: AnswerResult | null;
  onPick: (letter: string) => void;
  onLock: () => void;
}) {
  return (
    <aside className="commit-panel">
      <h3>Choose one setup</h3>
      <p>
        {answer
          ? answer.correct
            ? `Locked ${answer.picked}. Correct.`
            : `Locked ${answer.picked}. Correct answer: ${answer.correctLetter}.`
          : "No result data is shown before commitment."}
      </p>
      <div className="commit-buttons">
        {choices.map((choice) => (
          <button key={choice.letter} className={selected === choice.letter ? "active" : ""} onClick={() => onPick(choice.letter)}>
            {choice.letter}
          </button>
        ))}
      </div>
      <button className="lock-button" disabled={!selected || Boolean(answer)} onClick={onLock}>
        {answer ? "Answer locked" : "Lock answer →"}
      </button>
    </aside>
  );
}

function ResultPanel({ answer }: { answer: AnswerResult }) {
  return (
    <section className={`result-panel ${answer.correct ? "correct" : "wrong"}`}>
      <h2>{answer.correct ? "Correct" : "Not this time"}</h2>
      <p>
        You picked {answer.picked}. The correct answer is {answer.correctLetter}.
      </p>
      <div className="result-table">
        {answer.ranked.map((row) => (
          <div key={row.letter}>
            <strong>{row.letter}</strong>
            <span>{row.candidateId}</span>
            <span>{row.label}</span>
          </div>
        ))}
      </div>
    </section>
  );
}

function InfoRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="info-row">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function titleCase(text: string) {
  return text.replace(/\b\w/g, (char) => char.toUpperCase());
}

function displayVariant(fields: Field[]) {
  const priority = ["layers", "width", "activations", "residual", "layer norm", "optimizer", "learning rate", "loss", "lambda"];
  const ordered = priority
    .map((label) => fields.find((field) => field.label === label))
    .filter((field): field is Field => Boolean(field));
  for (const field of fields) {
    if (!priority.includes(field.label)) {
      ordered.push(field);
    }
  }
  return ordered.slice(0, 5);
}

createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
