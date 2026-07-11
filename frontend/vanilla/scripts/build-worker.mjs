import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";

const root = resolve(import.meta.dirname, "..");
const data = JSON.parse(await readFile(resolve(root, "src/generated/questions.json"), "utf8"));
const indexHtml = await readFile(resolve(root, "dist/index.html"), "utf8");
const out = resolve(root, "dist/server/index.js");

const worker = `const DATA = ${JSON.stringify(data)};
const INDEX_HTML = ${JSON.stringify(indexHtml)};

function json(value, init = {}) {
  return new Response(JSON.stringify(value), {
    ...init,
    headers: {
      "content-type": "application/json; charset=utf-8",
      ...(init.headers || {}),
    },
  });
}

function notFound(message = "Not found") {
  return json({ detail: message }, { status: 404 });
}

function questionIdFromPath(pathname) {
  const parts = pathname.split("/").filter(Boolean);
  return parts[2] || null;
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    if (url.pathname === "/api/health") {
      return json({ status: "ok" });
    }

    if (url.pathname === "/api/questions") {
      return json({ questions: DATA.questions });
    }

    if (url.pathname.startsWith("/api/questions/")) {
      const qid = questionIdFromPath(url.pathname);
      if (!qid) return notFound();

      if (url.pathname.endsWith("/answer") && request.method === "POST") {
        const payload = await request.json().catch(() => ({}));
        const letter = String(payload.letter || "").toUpperCase();
        const answer = DATA.answers[qid];
        if (!answer) return notFound("Question not found");
        const valid = new Set((DATA.details[qid]?.choices || []).map((choice) => choice.letter));
        if (!valid.has(letter)) return json({ detail: "Invalid answer letter" }, { status: 400 });
        return json({
          picked: letter,
          correctLetter: answer.correctLetter,
          correct: letter === answer.correctLetter,
          ranked: answer.ranked,
        });
      }

      const detail = DATA.details[qid];
      if (!detail) return notFound("Question not found");
      return json(detail);
    }

    if (env && env.ASSETS) {
      const assetResponse = await env.ASSETS.fetch(request);
      if (assetResponse.status !== 404) return assetResponse;
    }

    return new Response(INDEX_HTML, {
      headers: { "content-type": "text/html; charset=utf-8" },
    });
  },
};
`;

await mkdir(dirname(out), { recursive: true });
await writeFile(out, worker, "utf8");
console.log(`wrote ${out}`);
