#!/usr/bin/env node
/**
 * Split monolithic BakeFile for Cloudflare Pages (25 MiB file limit).
 *
 * Reads  <dir>/questions.json
 * Writes <dir>/index.json + <dir>/by-id/<id>.json
 * Deletes <dir>/questions.json
 */
import fs from "node:fs";
import path from "node:path";

const MAX_BYTES = 25 * 1024 * 1024;
const dirArgIdx = process.argv.indexOf("--dir");
const dir = path.resolve(dirArgIdx >= 0 ? process.argv[dirArgIdx + 1] : "dist/data");
const sourcePath = path.join(dir, "questions.json");

if (!fs.existsSync(sourcePath)) {
  console.error(`split-bake: missing ${sourcePath}`);
  process.exit(1);
}

const bake = JSON.parse(fs.readFileSync(sourcePath, "utf8"));
if (!bake || typeof bake !== "object" || !bake.byId || !Array.isArray(bake.questions)) {
  console.error("split-bake: questions.json is not a BakeFile");
  process.exit(1);
}

const byIdDir = path.join(dir, "by-id");
fs.mkdirSync(byIdDir, { recursive: true });

let maxFile = { path: "", bytes: 0 };
for (const [id, question] of Object.entries(bake.byId)) {
  const safe = String(id).replace(/[^a-zA-Z0-9._-]/g, "_");
  const outPath = path.join(byIdDir, `${safe}.json`);
  const body = JSON.stringify(question);
  fs.writeFileSync(outPath, body);
  if (body.length > maxFile.bytes) {
    maxFile = { path: outPath, bytes: body.length };
  }
}

const index = {
  schema_version: bake.schema_version,
  ordered: bake.ordered,
  collection: bake.collection ?? null,
  questions: bake.questions,
  split: true
};
const indexBody = JSON.stringify(index);
const indexPath = path.join(dir, "index.json");
fs.writeFileSync(indexPath, indexBody);
if (indexBody.length > maxFile.bytes) {
  maxFile = { path: indexPath, bytes: indexBody.length };
}

fs.unlinkSync(sourcePath);

const oversized = [];
for (const name of fs.readdirSync(byIdDir)) {
  const full = path.join(byIdDir, name);
  const bytes = fs.statSync(full).size;
  if (bytes > MAX_BYTES) {
    oversized.push({ full, bytes });
  }
}
if (indexBody.length > MAX_BYTES) {
  oversized.push({ full: indexPath, bytes: indexBody.length });
}

console.log(
  `split-bake: ${Object.keys(bake.byId).length} questions → ${path.relative(process.cwd(), dir)} ` +
    `(index ${(indexBody.length / (1024 * 1024)).toFixed(2)} MiB, ` +
    `largest ${(maxFile.bytes / (1024 * 1024)).toFixed(2)} MiB)`
);

if (oversized.length) {
  console.error("split-bake: files still exceed Cloudflare 25 MiB limit:");
  for (const item of oversized) {
    console.error(`  ${item.full} (${(item.bytes / (1024 * 1024)).toFixed(2)} MiB)`);
  }
  process.exit(1);
}
