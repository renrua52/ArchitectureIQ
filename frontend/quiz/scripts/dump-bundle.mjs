/**
 * Dump the repro-bundle entries the browser would produce, as JSON on stdout.
 *
 * `src/bundle.ts` is the shipped builder; `tools/export_repro_bundle.py` is its
 * Python twin. This script is how `tests/test_repro_bundle.py` compares the two
 * without a browser: esbuild bundles bundle.ts (resolving Vite's `?raw` imports
 * the way Vite does), then we call buildBundleEntries() under plain Node.
 *
 *   node scripts/dump-bundle.mjs --bake <bake.json> --question q_502033 [--answered]
 */

import { readFile, writeFile, mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, dirname, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import * as esbuild from "esbuild";

const HERE = dirname(fileURLToPath(import.meta.url));
const QUIZ_ROOT = resolve(HERE, "..");

function arg(name, fallback = undefined) {
  const index = process.argv.indexOf(`--${name}`);
  return index === -1 ? fallback : process.argv[index + 1];
}

/** Mirror of Vite's `?raw` suffix: import the file's text as a default export. */
const rawPlugin = {
  name: "vite-raw",
  setup(build) {
    build.onResolve({ filter: /\?raw$/ }, (args) => ({
      path: resolve(args.resolveDir, args.path.replace(/\?raw$/, "")),
      namespace: "raw",
    }));
    build.onLoad({ filter: /.*/, namespace: "raw" }, async (args) => ({
      contents: `export default ${JSON.stringify(await readFile(args.path, "utf8"))};`,
      loader: "js",
    }));
  },
};

const bakePath = arg("bake", join(QUIZ_ROOT, "public", "data", "questions.json"));
const questionId = arg("question");
const answered = process.argv.includes("--answered");
if (!questionId) {
  console.error("usage: node scripts/dump-bundle.mjs --question <id> [--bake <path>] [--answered]");
  process.exit(2);
}

const work = await mkdtemp(join(tmpdir(), "aiq-bundle-"));
try {
  const outfile = join(work, "bundle.mjs");
  await esbuild.build({
    entryPoints: [join(QUIZ_ROOT, "src", "bundle.ts")],
    outfile,
    bundle: true,
    format: "esm",
    platform: "node",
    plugins: [rawPlugin],
    logLevel: "silent",
  });
  const { buildBundleEntries } = await import(pathToFileURL(outfile).href);
  const bake = JSON.parse(await readFile(bakePath, "utf8"));
  const question = bake.byId?.[questionId];
  if (!question) {
    console.error(`${questionId} not in ${bakePath}`);
    process.exit(1);
  }
  const entries = buildBundleEntries(question, answered);
  process.stdout.write(JSON.stringify(Object.fromEntries(entries.map((e) => [e.path, e.content]))));
} finally {
  await rm(work, { recursive: true, force: true });
}
