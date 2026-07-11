import React, { useEffect, useState } from "react";
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
            <EvidencePanel
              question={question}
              tab={tab}
              onTab={setTab}
              selected={selected}
              answer={answer}
              onSelect={(letter) => {
                if (!answer) {
                  setSelected(letter);
                }
              }}
              onLock={lockAnswer}
            />
            <SharedPanel question={question} />
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
  selected,
  answer,
  onSelect,
  onLock
}: {
  question: Question;
  tab: EvidenceTab;
  onTab: (tab: EvidenceTab) => void;
  selected: string | null;
  answer: AnswerResult | null;
  onSelect: (letter: string) => void;
  onLock: () => void;
}) {
  return (
    <div className="panel evidence-panel">
      <div className="panel-head">
        <div>
          <span className="dot" />
          <strong>Evidence · {question.id}</strong>
          <small>{question.choices.map((choice) => `${choice.letter} ${choice.candidateId}`).join("  ·  ")}</small>
        </div>
        <nav className="tabs" aria-label="Evidence tabs">
          {(["matrix", "protocol", "prompt"] as EvidenceTab[]).map((item) => (
            <button key={item} className={tab === item ? "active" : ""} onClick={() => onTab(item)}>
              {item === "matrix" ? "Matrix" : item[0].toUpperCase() + item.slice(1)}
            </button>
          ))}
        </nav>
      </div>
      <div className="matchup-strip" aria-label="Current question matchup">
        {question.choices.map((choice, index) => {
          const isSelected = selected === choice.letter;
          const isCorrect = answer?.correctLetter === choice.letter;
          const isWrongPick = answer?.picked === choice.letter && !answer.correct;
          return (
            <button
              key={choice.letter}
              className={[
                "matchup-choice",
                isSelected ? "selected" : "",
                isCorrect ? "correct" : "",
                isWrongPick ? "wrong" : ""
              ]
                .filter(Boolean)
                .join(" ")}
              disabled={Boolean(answer)}
              onClick={() => onSelect(choice.letter)}
              style={{ "--accent": choiceColors[index % choiceColors.length] } as React.CSSProperties}
            >
              <span>{choice.letter}</span>
              <strong>{choice.candidateId}</strong>
              <small>{displayVariant(choice.variant).map((field) => `${titleCase(field.label)} ${field.value}`).join(" · ")}</small>
            </button>
          );
        })}
      </div>
      {tab === "matrix" ? <DatasetPlot question={question} answer={answer} /> : null}
      {tab === "protocol" ? <Protocol question={question} /> : null}
      {tab === "prompt" ? <pre className="prompt-box">{question.prompt}</pre> : null}
      <div className="evidence-actions">
        <div className="hint-chip">
          {answer
            ? answer.correct
              ? `Locked ${answer.picked}. Correct.`
              : `Locked ${answer.picked}. Winner: ${answer.correctLetter}.`
            : selected
              ? `Selected ${selected}.`
              : "No result data before commitment."}
        </div>
        <button className="lock-button evidence-lock" disabled={!selected || Boolean(answer)} onClick={onLock}>
          {answer ? "Answer locked" : "Lock answer →"}
        </button>
      </div>
    </div>
  );
}

function DatasetPlot({ question, answer }: { question: Question; answer: AnswerResult | null }) {
  const train = question.dataset.train ?? [];
  const test = question.dataset.test ?? [];
  const allPoints = [...train, ...test];
  const width = 930;
  const height = 430;
  const plot = { x: 72, y: 54, width: 770, height: 300 };
  const domain = pointDomain(allPoints);
  const xTicks = makeTicks(domain.xMin, domain.xMax, 7);
  const yTicks = makeTicks(domain.yMin, domain.yMax, 5);

  const pointPosition = (point: Point) => ({
    x: plot.x + ((point.x - domain.xMin) / (domain.xMax - domain.xMin || 1)) * plot.width,
    y: plot.y + plot.height - ((point.y - domain.yMin) / (domain.yMax - domain.yMin || 1)) * plot.height
  });

  return (
    <div className="plot-frame">
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`Dataset plot for ${question.id}`}>
        <rect x="28" y="24" width="874" height="372" rx="18" fill="#fffefa" stroke="#24262c" strokeWidth="2" />
        <text x="54" y="42" fill="#6d7080" fontSize="13" fontWeight="800">
          Dataset {question.datasetId}
        </text>
        <text x="844" y="42" textAnchor="end" fill="#6d7080" fontSize="13" fontWeight="800">
          train {train.length} · test {test.length}
        </text>
        <rect x={plot.x} y={plot.y} width={plot.width} height={plot.height} rx="10" fill="#f4f0ff" stroke="#24262c" strokeWidth="1.5" />
        {xTicks.map((tick) => {
          const x = plot.x + ((tick - domain.xMin) / (domain.xMax - domain.xMin || 1)) * plot.width;
          return (
            <g key={`x-${tick}`}>
              <line x1={x} x2={x} y1={plot.y} y2={plot.y + plot.height} stroke="#d8d1e9" />
              <text x={x} y={plot.y + plot.height + 26} textAnchor="middle" fill="#6e7078" fontSize="11" fontWeight="700">
                {formatTick(tick)}
              </text>
            </g>
          );
        })}
        {yTicks.map((tick) => {
          const y = plot.y + plot.height - ((tick - domain.yMin) / (domain.yMax - domain.yMin || 1)) * plot.height;
          return (
            <g key={`y-${tick}`}>
              <line x1={plot.x} x2={plot.x + plot.width} y1={y} y2={y} stroke="#d8d1e9" />
              <text x={plot.x - 14} y={y + 4} textAnchor="end" fill="#6e7078" fontSize="11" fontWeight="700">
                {formatTick(tick)}
              </text>
            </g>
          );
        })}
        <line x1={plot.x} x2={plot.x + plot.width} y1={plot.y + plot.height} y2={plot.y + plot.height} stroke="#24262c" strokeWidth="1.5" />
        <line x1={plot.x} x2={plot.x} y1={plot.y} y2={plot.y + plot.height} stroke="#24262c" strokeWidth="1.5" />
        <text x={plot.x} y={plot.y + plot.height + 52} fill="#26364c" fontSize="13" fontWeight="800">
          x
        </text>
        <text x={plot.x - 24} y={plot.y + 4} fill="#26364c" fontSize="13" fontWeight="800">
          y
        </text>
        {train.map((point, index) => {
          const pos = pointPosition(point);
          return <circle key={`train-${index}`} cx={pos.x} cy={pos.y} r="4.1" fill="#734cff" opacity="0.78" />;
        })}
        {test.map((point, index) => {
          const pos = pointPosition(point);
          return <circle key={`test-${index}`} cx={pos.x} cy={pos.y} r="4.1" fill="#20a87e" opacity="0.78" />;
        })}
        <g transform="translate(664 76)">
          <rect x="0" y="0" width="146" height="66" rx="14" fill="#fffefa" stroke="#d9d2e9" />
          <circle cx="20" cy="22" r="5" fill="#734cff" />
          <text x="34" y="27" fill="#26364c" fontSize="13" fontWeight="800">
            train
          </text>
          <circle cx="20" cy="46" r="5" fill="#20a87e" />
          <text x="34" y="51" fill="#26364c" fontSize="13" fontWeight="800">
            test
          </text>
        </g>
        {answer ? (
          <g transform="translate(54 366)">
            <rect x="0" y="0" width="242" height="36" rx="18" fill={answer.correct ? "#20a87e" : "#d94b45"} />
            <text x="121" y="23" textAnchor="middle" fill="#fff" fontSize="13" fontWeight="850">
              {answer.correct ? "Correct" : `Winner: ${answer.correctLetter}`}
            </text>
          </g>
        ) : null}
      </svg>
      {answer ? <InlineComparison answer={answer} /> : null}
    </div>
  );
}

function pointDomain(points: Point[]) {
  const xValues = points.map((point) => point.x).filter(Number.isFinite);
  const yValues = points.map((point) => point.y).filter(Number.isFinite);
  const xMin = Math.min(...xValues, 0);
  const xMax = Math.max(...xValues, 1);
  const yMin = Math.min(...yValues, 0);
  const yMax = Math.max(...yValues, 1);
  const xPad = Math.max((xMax - xMin) * 0.08, 0.1);
  const yPad = Math.max((yMax - yMin) * 0.12, 0.1);
  return {
    xMin: xMin - xPad,
    xMax: xMax + xPad,
    yMin: yMin - yPad,
    yMax: yMax + yPad
  };
}

function makeTicks(min: number, max: number, count: number) {
  if (count <= 1) {
    return [min];
  }
  return Array.from({ length: count }, (_, index) => min + ((max - min) * index) / (count - 1));
}

function formatTick(value: number) {
  const abs = Math.abs(value);
  if (abs >= 100 || abs === 0) {
    return value.toFixed(0);
  }
  if (abs >= 10) {
    return value.toFixed(1);
  }
  return value.toFixed(2).replace(/\.?0+$/, "");
}

function InlineComparison({ answer }: { answer: AnswerResult }) {
  return (
    <div className="inline-comparison">
      <div>
        <strong>{answer.correct ? "Correct" : "Not this time"}</strong>
        <span>
          You picked {answer.picked}. Winner: {answer.correctLetter}.
        </span>
      </div>
      <div className="comparison-list">
        {answer.ranked.map((row, index) => (
          <div key={row.letter}>
            <span>#{index + 1}</span>
            <strong>{row.letter}</strong>
            <em>{row.candidateId}</em>
            <small>{row.label}</small>
          </div>
        ))}
      </div>
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
