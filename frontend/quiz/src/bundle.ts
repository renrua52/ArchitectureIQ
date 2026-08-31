/**
 * Assemble the downloadable "reproduce the ground truth" bundle for one question.
 *
 * Everything the bundle needs already lives in the BakeFile: `detail.dataset.files`,
 * `detail.choices[].files`, and (after reveal) `reveal.files[letter]`. So this is
 * pure repackaging — no extra fetch, no bake or schema change.
 *
 * `tools/export_repro_bundle.py` builds the identical layout from the same bake,
 * and `tests/test_repro_bundle.py` cross-checks the two. Keep them in step.
 */

import type { BakedQuestion } from "./types";
import reproduceSource from "../repro/reproduce.py?raw";
import bundleReadme from "../repro/README.md?raw";
import { createZip, downloadBlob, type ZipEntry } from "./zip";

export const BUNDLE_VERSION = 1;

/** Bake `files` values are either raw source text or already-parsed JSON. */
function renderFile(value: unknown): string {
  if (typeof value === "string") return value;
  return `${JSON.stringify(value, null, 2)}\n`;
}

function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" ? (value as Record<string, unknown>) : {};
}

function numberOrUndefined(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

/** True when the bake carries the source files a bundle needs (older bakes may not). */
export function canBuildBundle(question: BakedQuestion | undefined | null): boolean {
  if (!question) return false;
  const dataset = asRecord(question.detail?.dataset?.files);
  if (typeof dataset["synthesize.py"] !== "string") return false;
  const choices = question.detail?.choices ?? [];
  if (choices.length === 0) return false;
  return choices.every((choice) => typeof asRecord(choice.files)["train.py"] === "string");
}

export function bundleFilename(questionId: string): string {
  return `architectureiq_${questionId}.zip`;
}

/**
 * Build the bundle entries.
 *
 * `answered` gates exactly two things — the reference metrics and the answer key —
 * mirroring what the file viewer already reveals (see `InfoModal` in main.tsx).
 */
export function buildBundleEntries(question: BakedQuestion, answered: boolean): ZipEntry[] {
  const root = question.id;
  const evaluation = asRecord(question.evaluation);
  const datasetFiles = asRecord(question.detail.dataset.files);
  const datasetSpec = asRecord(datasetFiles["dataset_spec.json"]);
  const significance = asRecord(datasetSpec["significance"]);
  const metric =
    (question.detail.dataset.selectionMetric as string | undefined) ??
    (evaluation["selection_metric"] as string | undefined) ??
    question.metric ??
    "test_mse";

  const revealFiles = asRecord(question.reveal?.files) as Record<string, Record<string, unknown>>;

  // The thread count the ground truth ran with lets reproduce.py match bit for bit.
  // It is an environment detail, not part of the answer, so it is not gated.
  let threads: number | undefined;
  for (const letter of Object.keys(revealFiles)) {
    const summary = asRecord(revealFiles[letter]?.["summary.json"]);
    const candidate = numberOrUndefined(asRecord(summary["environment"])["torch_threads_per_seed"]);
    if (candidate !== undefined) {
      threads = candidate;
      break;
    }
  }

  const meta: Record<string, unknown> = {
    bundle_version: BUNDLE_VERSION,
    question_id: question.id,
    family: question.family,
    dataset_id: question.datasetId,
    type: question.type,
    profile: question.profile ?? null,
    selection_metric: metric,
    n_seeds: numberOrUndefined(evaluation["n_seeds"]) ?? 10,
    base_seed: numberOrUndefined(evaluation["base_seed"]) ?? 0,
    device: (evaluation["device"] as string | undefined) ?? "cpu",
    fail_threshold: numberOrUndefined(significance["fail_threshold"]) ?? null,
    varying_axes: question.varyingAxes ?? [],
    invariant_axes: question.invariantAxes ?? [],
    answered,
    choices: question.detail.choices.map((choice) => {
      const spec = asRecord(asRecord(choice.files)["candidate_spec.json"]);
      const budget = asRecord(spec["budget"]);
      return {
        letter: choice.letter,
        candidate_id: choice.candidateId,
        training_steps: numberOrUndefined(budget["training_steps"]) ?? null,
        batch_size: numberOrUndefined(budget["batch_size"]) ?? null,
        total_samples_seen: numberOrUndefined(budget["total_samples_seen"]) ?? null,
      };
    }),
  };
  if (threads !== undefined) meta.torch_threads_per_seed = threads;
  if (answered) {
    meta.correct_letter = question.reveal.correctLetter;
    meta.ranked = (question.reveal.ranked ?? []).map((entry) => entry.letter);
  }

  const entries: ZipEntry[] = [
    { path: `${root}/README.md`, content: bundleReadme },
    { path: `${root}/reproduce.py`, content: reproduceSource },
    { path: `${root}/question.json`, content: `${JSON.stringify(meta, null, 2)}\n` },
    { path: `${root}/prompt.txt`, content: question.detail.prompt ?? "" },
  ];

  for (const name of Object.keys(datasetFiles).sort()) {
    entries.push({ path: `${root}/dataset/${name}`, content: renderFile(datasetFiles[name]) });
  }

  for (const choice of question.detail.choices) {
    const files = asRecord(choice.files);
    for (const name of Object.keys(files).sort()) {
      entries.push({
        path: `${root}/choices/${choice.letter}/${name}`,
        content: renderFile(files[name]),
      });
    }
    if (!answered) continue;
    const reference = asRecord(revealFiles[choice.letter]);
    for (const name of Object.keys(reference).sort()) {
      entries.push({
        path: `${root}/choices/${choice.letter}/reference/${name}`,
        content: renderFile(reference[name]),
      });
    }
  }

  return entries;
}

export function downloadBundle(question: BakedQuestion, answered: boolean): void {
  const blob = createZip(buildBundleEntries(question, answered));
  downloadBlob(blob, bundleFilename(question.id));
}
