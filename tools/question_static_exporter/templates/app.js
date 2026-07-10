const state = {
  index: 0,
  committed: {},
  focusLetter: null,
  fileScope: "prompt",
  fileName: null,
};

const data = window.ARCHITECTURE_IQ_DATA || { title: "ArchitectureIQ Quiz", questions: [] };

function storageKey() {
  return `architecture-iq-static:${data.generated_at || "local"}`;
}

function loadState() {
  try {
    const saved = JSON.parse(localStorage.getItem(storageKey()) || "{}");
    if (saved && typeof saved === "object") {
      state.committed = saved.committed || {};
      state.index = Number.isInteger(saved.index) ? saved.index : 0;
    }
  } catch {
    state.committed = {};
  }
}

function saveState() {
  localStorage.setItem(
    storageKey(),
    JSON.stringify({ committed: state.committed, index: state.index }),
  );
}

function el(tag, attrs = {}, children = []) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value === false || value === null || value === undefined) continue;
    if (key === "class") node.className = value;
    else if (key === "text") node.textContent = String(value);
    else if (key === "html") node.innerHTML = value;
    else if (key.startsWith("on") && typeof value === "function") node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, String(value));
  }
  for (const child of children) {
    if (child === null || child === undefined) continue;
    node.append(child.nodeType ? child : document.createTextNode(String(child)));
  }
  return node;
}

function fmt(value, digits = 6) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  return Number(value).toFixed(digits);
}

function currentQuestion() {
  return data.questions[state.index] || null;
}

function score() {
  let total = 0;
  let correct = 0;
  for (const q of data.questions) {
    const picked = state.committed[q.id];
    if (!picked) continue;
    total += 1;
    if (picked === q.correct_letter) correct += 1;
  }
  return { total, correct };
}

function setQuestion(index) {
  state.index = (index + data.questions.length) % data.questions.length;
  state.focusLetter = null;
  state.fileScope = "prompt";
  state.fileName = null;
  saveState();
  render();
}

function selectAnswer(letter) {
  const q = currentQuestion();
  if (!q || state.committed[q.id]) return;
  state.committed[q.id] = letter;
  state.focusLetter = letter;
  saveState();
  render();
}

function resetScore() {
  state.committed = {};
  state.focusLetter = null;
  saveState();
  render();
}

function questionSelect() {
  const select = el("select", {
    onchange: (event) => setQuestion(Number(event.target.value)),
    "aria-label": "Question",
  });
  data.questions.forEach((q, index) => {
    select.append(el("option", { value: index, text: q.label }));
  });
  select.value = String(state.index);
  return select;
}

function renderSidebar() {
  const s = score();
  return el("aside", { class: "sidebar" }, [
    el("div", { class: "brand" }, [
      el("h1", { text: data.title || "ArchitectureIQ Quiz" }),
      el("p", { text: `${data.questions.length} questions · offline static build` }),
    ]),
    el("div", { class: "control-panel" }, [
      el("label", { class: "fact-label", text: "Question" }),
      questionSelect(),
      el("div", { class: "sidebar-actions", style: "margin-top: .7rem" }, [
        el("button", { text: "Next", onclick: () => setQuestion(state.index + 1) }),
        el("button", {
          text: "Random",
          onclick: () => setQuestion(Math.floor(Math.random() * data.questions.length)),
        }),
      ]),
    ]),
    el("div", { class: "score-panel" }, [
      el("div", { class: "score" }, [
        el("span", { class: "muted", text: "Score" }),
        el("strong", { text: `${s.correct} / ${s.total}` }),
      ]),
      el("button", { class: "ghost", text: "Reset score", onclick: resetScore }),
    ]),
    el("div", { class: "score-panel muted" }, [
      el("div", { text: `Generated: ${data.generated_at || "-"}` }),
      el("div", { text: "Answers are embedded locally; use this as practice/demo material." }),
    ]),
  ]);
}

function renderDataset(q) {
  const info = q.dataset.info || {};
  const facts = [
    el("div", {}, [el("div", { class: "fact-label", text: "Dataset ID" }), el("div", { text: q.dataset_id })]),
    el("div", {}, [el("div", { class: "fact-label", text: "Family" }), el("div", { text: q.family })]),
    ...((info.summary_lines || []).map((line) => el("div", { text: line }))),
  ];
  if (info.latex_expression) {
    facts.push(el("div", {}, [el("div", { class: "fact-label", text: "LaTeX" }), el("div", { class: "latex", text: info.latex_expression })]));
  }
  return el("section", { class: "section" }, [
    el("div", { class: "section-header" }, [el("h2", { text: "Dataset" })]),
    el("div", { class: "section-body dataset-grid" }, [
      el("div", { class: "facts" }, facts),
      q.dataset.plot
        ? el("img", { class: "plot", src: q.dataset.plot, alt: "Dataset plot" })
        : el("div", { class: "muted", text: "Dataset plot unavailable." }),
    ]),
  ]);
}

function specBlock(block) {
  return el("div", { class: "spec-block" }, [
    el("div", { class: "spec-label", text: block.label }),
    ...(block.lines || []).map((line) => el("div", { class: "spec-line", text: line })),
  ]);
}

function choiceClass(q, choice, picked) {
  const classes = ["choice-card"];
  if (state.focusLetter === choice.letter) classes.push("focused");
  if (picked) {
    if (choice.letter === q.correct_letter) classes.push("correct");
    if (choice.letter === picked && choice.letter !== q.correct_letter) classes.push("incorrect");
  }
  return classes.join(" ");
}

function renderChoice(q, choice) {
  const picked = state.committed[q.id];
  const committed = Boolean(picked);
  const buttonText = committed ? (picked === choice.letter ? "Your pick" : "View") : "Select";
  const disabled = committed && picked === choice.letter;
  const children = [
    el("div", { class: "choice-head" }, [
      el("div", {}, [
        el("div", { class: "letter", text: choice.letter }),
        el("div", { class: "candidate-id", text: choice.candidate_id }),
      ]),
      el("button", {
        class: "info-button",
        text: "i",
        title: "View candidate files",
        onclick: () => {
          state.focusLetter = choice.letter;
          state.fileScope = `choice:${choice.letter}`;
          state.fileName = null;
          render();
        },
      }),
    ]),
    ...(choice.spec_blocks || []).map(specBlock),
  ];
  if (committed) children.push(el("span", { class: "metric-pill", text: choice.metric.display || "Metrics unavailable" }));
  children.push(
    el("button", {
      class: !committed || state.focusLetter === choice.letter ? "primary" : "",
      text: buttonText,
      disabled,
      onclick: () => {
        if (!committed) selectAnswer(choice.letter);
        else {
          state.focusLetter = choice.letter;
          state.fileScope = `choice:${choice.letter}`;
          state.fileName = null;
          render();
        }
      },
    }),
  );
  return el("article", { class: choiceClass(q, choice, picked) }, children);
}

function renderChoices(q) {
  return el("section", { class: "section" }, [
    el("div", { class: "section-header" }, [el("h2", { text: "Choices" })]),
    el("div", { class: "section-body choices" }, q.choices.map((choice) => renderChoice(q, choice))),
  ]);
}

function renderAnswer(q) {
  const picked = state.committed[q.id];
  if (!picked) return null;
  const correct = picked === q.correct_letter;
  return el("div", { class: `answer-banner ${correct ? "correct" : "incorrect"}` }, [
    correct
      ? `Correct: ${picked} achieves the best ${q.metric_display}.`
      : `Incorrect: you picked ${picked}; correct answer is ${q.correct_letter}.`,
  ]);
}

function renderResults(q) {
  if (!state.committed[q.id]) return null;
  return el("section", { class: "section" }, [
    el("div", { class: "section-header" }, [el("h2", { text: "Results" })]),
    el("div", { class: "section-body" }, [
      q.curves_plot
        ? el("img", { class: "plot", src: q.curves_plot, alt: "Learning curves" })
        : el("p", { class: "muted", text: "Learning curves unavailable." }),
      el(
        "div",
        { class: "ranked" },
        (q.ranked_results || []).map((row, index) =>
          el("div", { class: "rank-row" }, [
            el("span", {
              text: `${index + 1}. ${row.letter} · ${row.candidate_id}${row.correct ? " · best" : ""}`,
            }),
            el("span", { text: `${fmt(row.mean)} ± ${fmt(row.std)}` }),
          ]),
        ),
      ),
    ]),
  ]);
}

function availableFiles(q) {
  if (state.fileScope === "dataset") return q.dataset.files || {};
  if (state.fileScope.startsWith("choice:")) {
    const letter = state.fileScope.split(":")[1];
    const choice = q.choices.find((item) => item.letter === letter);
    return choice ? choice.files || {} : {};
  }
  return { "prompt.txt": q.prompt_text, "question.json": JSON.stringify(q.question, null, 2) };
}

function renderFiles(q) {
  const scopes = [
    ["prompt", "Prompt"],
    ["dataset", "Dataset files"],
    ...q.choices.map((choice) => [`choice:${choice.letter}`, `Choice ${choice.letter}`]),
  ];
  const files = availableFiles(q);
  const names = Object.keys(files);
  const activeName = state.fileName && files[state.fileName] !== undefined ? state.fileName : names[0];
  state.fileName = activeName || null;
  return el("section", { class: "section" }, [
    el("div", { class: "section-header" }, [el("h2", { text: "Inspect" })]),
    el("div", { class: "section-body" }, [
      el(
        "div",
        { class: "tabs" },
        scopes.map(([key, label]) =>
          el("button", {
            class: state.fileScope === key ? "active" : "",
            text: label,
            onclick: () => {
              state.fileScope = key;
              state.fileName = null;
              render();
            },
          }),
        ),
      ),
      el("div", { class: "file-grid" }, [
        el(
          "div",
          { class: "file-list" },
          names.map((name) =>
            el("button", {
              class: activeName === name ? "active" : "",
              text: name,
              onclick: () => {
                state.fileName = name;
                render();
              },
            }),
          ),
        ),
        el("pre", { text: activeName ? files[activeName] : "No files available." }),
      ]),
    ]),
  ]);
}

function renderMain() {
  const q = currentQuestion();
  if (!q) {
    return el("main", { class: "main" }, [el("p", { text: "No questions found in this export." })]);
  }
  return el("main", { class: "main" }, [
    el("div", { class: "content" }, [
      el("div", { class: "topline" }, [
        el("div", {}, [
          el("h1", { class: "qid", text: q.id }),
          el("p", {
            class: "meta",
            text: `Type: ${q.type || "-"} · Budget: ${q.budget_total_samples || "-"} samples · Metric: ${q.metric || "-"} · Choices: ${q.num_choices}`,
          }),
        ]),
      ]),
      renderAnswer(q),
      renderDataset(q),
      renderChoices(q),
      renderResults(q),
      renderFiles(q),
    ]),
  ]);
}

function render() {
  const app = document.getElementById("app");
  app.replaceChildren(el("div", { class: "layout" }, [renderSidebar(), renderMain()]));
}

loadState();
render();
