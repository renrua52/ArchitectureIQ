# Frontend Concept Implementation Playbook

This guide turns a conceptual design image into a working ArchitectureIQ frontend quickly. It is based on the vanilla frontend implementation process used in this repository.

## Goal

Given a concept image, build a frontend that preserves the visual direction while remaining faithful to the backend data model.

For ArchitectureIQ, the core interaction is:

1. Welcome the user.
2. Show a quiz question.
3. Display the dataset evidence from `dataset.train` and `dataset.test`.
4. Let the user pick a candidate setup.
5. Lock the answer.
6. Reveal the comparison/result after commitment.

## Current Architecture

Frontend:

- `frontend/vanilla/src/main.tsx`
- `frontend/vanilla/src/styles.css`
- Vite + React + TypeScript

Backend:

- `backend/app.py`
- `GET /api/questions`
- `GET /api/questions/{question_id}`
- `POST /api/questions/{question_id}/answer`

Question generation:

- `src/architecture_iq/datasets.py`
- `src/architecture_iq/questions/generator.py`
- `src/architecture_iq/cli.py`

Important backend fact: the backend sends structured JSON, not images. The frontend must render figures from the data.

## Backend Payload To Design Against

A full question response includes:

- `id`
- `title`
- `family`
- `datasetId`
- `budget`
- `metric`
- `evaluation`
- `prompt`
- `dataset.train`
- `dataset.test`
- `shared`
- `choices`

The dataset field looks like:

```json
{
  "dataset": {
    "train": [{ "x": 0.12, "y": -1.4 }],
    "test": [{ "x": 0.18, "y": -1.2 }]
  }
}
```

The frontend should render this as a central train/test plot.

## Implementation Workflow

### 1. Read The Concept Image

Extract design intent from the image:

- Layout hierarchy
- Visual density
- Rounded panel style
- Button placement
- Color accents
- Typography scale
- Where the user should look first
- Which panels are primary vs secondary

Do not copy decorative details blindly. Convert them into a usable UI.

### 2. Inspect The Backend Before Designing

Run:

```bash
curl http://127.0.0.1:8000/api/questions
```

Then inspect one question:

```bash
curl http://127.0.0.1:8000/api/questions/<question_id>
```

Confirm:

- Whether the backend sends images. It currently does not.
- Whether each question has train/test data. It should.
- Whether datasets vary by question. In the current 60-question run, all questions use the same dataset payload.

### 3. Build The First Screen

The welcome page should be simple:

- Title: `Test your Architecture IQ`
- Subtitle: `Wanna know how wise you are for picking the right setup for training tasks :)?`
- One CTA: `Test my architecture IQ`

No marketing page, no extra onboarding.

### 4. Build The Quiz Screen Around The Real Task

The main quiz screen should include:

- Top bar with brand, progress, next/random controls, and score.
- Question hero with title and metadata chips.
- Evidence panel as the primary work area.
- Shared training setup panel as supporting context.

The evidence panel should own the answer flow:

- Candidate matchup tiles are clickable answer buttons.
- The dataset train/test plot is centered below the matchup tiles.
- `Lock answer` sits inside the evidence panel.
- Result comparison appears only after commitment.

Avoid duplicating answer controls elsewhere.

### 5. Render Dataset Evidence Client-Side

Use `dataset.train` and `dataset.test` to render an SVG scatterplot:

- Compute the x/y domain from both splits.
- Add padding around the domain.
- Draw a chart frame, grid, axes, tick labels, and legend.
- Use distinct colors for train/test.
- Keep the plot responsive by using an SVG `viewBox`.

This is better than asking the backend for images because the frontend remains lightweight and interactive.

### 6. Preserve Commitment Integrity

Before the user locks an answer:

- Do not show result data.
- Do not show ranking.
- Only show candidate setup metadata and dataset evidence.

After the user locks:

- Disable candidate selection.
- Mark the correct choice.
- Mark the wrong picked choice if applicable.
- Show the ranked comparison from the answer endpoint.

### 7. Keep Styling In The Concept's Spirit

For the vanilla concept:

- Use warm off-white panel backgrounds.
- Use strong black outlines.
- Use purple as the primary accent.
- Use compact rounded chips.
- Keep the quiz dense but readable.
- Use panels with purpose, not decorative nesting.

Avoid:

- A separate bottom answer rail if candidate tiles already exist.
- Static placeholder figures.
- Frontend visuals that do not derive from backend data.
- Large explanatory text blocks inside the app.

### 8. Verify

Run:

```bash
cd frontend/vanilla
npm run build
```

If browser testing is available, verify:

- Welcome button enters quiz.
- Next question updates question metadata.
- Candidate tiles select answers.
- Lock answer calls the backend.
- Result comparison appears after lock.
- Dataset plot renders train/test points.

If browser automation cannot access localhost, state that clearly and rely on build/API verification.

## Common Pitfalls

- Assuming the backend returns images. It returns JSON.
- Rendering a static figure instead of using `dataset.train` and `dataset.test`.
- Making duplicate answer controls.
- Showing comparison/result data before the answer is locked.
- Forgetting that current generated questions may share the same dataset.
- Changing unrelated backend or generated data while only implementing the frontend.

## Reusable Developer Prompt

Use this prompt when handing a concept image to a new developer:

```text
You are implementing a React/Vite frontend for ArchitectureIQ from the attached concept design image.

First inspect the existing codebase and backend API. Confirm the shape of:
- GET /api/questions
- GET /api/questions/{question_id}
- POST /api/questions/{question_id}/answer

Important: the backend does not send images for questions. It sends structured JSON. Render the central evidence figure from `question.dataset.train` and `question.dataset.test`.

Use the concept image as visual direction, not as a static mock. Build the actual quiz experience:
1. A simple welcome page:
   - Title: "Test your Architecture IQ"
   - Subtitle: "Wanna know how wise you are for picking the right setup for training tasks :)?"
   - One button: "Test my architecture IQ"
2. A quiz page:
   - Top progress/score bar.
   - Question title and metadata chips.
   - Evidence panel as the primary area.
   - Shared training setup panel as secondary context.
   - Candidate matchup tiles inside the evidence panel.
   - Make the matchup tiles clickable answer buttons.
   - Put the Lock answer button inside the evidence panel.
   - Remove any duplicate bottom answer panel.
   - Show no result data before commitment.
   - After locking, mark correct/wrong choices and show ranked comparison.

Render the dataset plot as responsive SVG:
- Compute domain from train and test points.
- Draw axes, grid, ticks, train/test legend.
- Use the concept's warm off-white panel style, black outlines, compact chips, and purple accent.

Keep implementation scoped to the frontend unless API issues are discovered. Follow existing project style. Run `npm run build` before finishing. If browser automation cannot access localhost, report that limitation and include the build/API checks you ran.
```

## Suggested File Checklist

Implementation usually touches:

- `frontend/vanilla/src/main.tsx`
- `frontend/vanilla/src/styles.css`

Optional docs:

- `AGENT.html` for a visual implementation brief.
- This Markdown file for repeatable process documentation.
