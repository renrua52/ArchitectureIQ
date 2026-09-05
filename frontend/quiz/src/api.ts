/**
 * Backend client for the quiz user system.
 *
 * The browser talks to Supabase PostgREST with the anon key, which is
 * locked down by RLS: it can only call three SECURITY DEFINER RPCs
 * (quiz_register / quiz_ingest_chunk / quiz_upsert_session). The per-user
 * token returned by quiz_register authorizes uploads at the RPC level.
 *
 * Failed uploads are persisted in localStorage and retried, so a flaky
 * network never loses recorded data. When no backend is configured every
 * call degrades to a silent no-op (recording still works locally).
 */

import type { RecEvent } from "./recorder";

const SUPA_URL = (import.meta.env.VITE_SUPABASE_URL as string | undefined)?.replace(/\/$/, "");
const SUPA_ANON = import.meta.env.VITE_SUPABASE_ANON_KEY as string | undefined;

const LS_AUTH = "aiq.auth.v1";
const LS_QUEUE = "aiq.uploadQueue.v1";
const QUEUE_MAX = 400;

export type Auth = {
  user_id: string;
  username: string;
  token: string;
};

type QueuedChunk = {
  session_id: string;
  seq: number;
  events: RecEvent[];
};

export function apiConfigured(): boolean {
  return Boolean(SUPA_URL && SUPA_ANON);
}

export function loadAuth(): Auth | null {
  try {
    const raw = localStorage.getItem(LS_AUTH);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as Auth;
    if (!parsed.user_id || !parsed.token || !parsed.username) return null;
    return parsed;
  } catch {
    return null;
  }
}

export function saveAuth(auth: Auth): void {
  localStorage.setItem(LS_AUTH, JSON.stringify(auth));
}

export function clearAuth(): void {
  localStorage.removeItem(LS_AUTH);
}

async function rpc<T>(fn: string, body: Record<string, unknown>, keepalive = false): Promise<T> {
  if (!SUPA_URL || !SUPA_ANON) throw new Error("backend_not_configured");
  const response = await fetch(`${SUPA_URL}/rest/v1/rpc/${fn}`, {
    method: "POST",
    headers: {
      apikey: SUPA_ANON,
      Authorization: `Bearer ${SUPA_ANON}`,
      "Content-Type": "application/json"
    },
    body: JSON.stringify(body),
    keepalive
  });
  if (!response.ok) {
    let message = `rpc_${response.status}`;
    try {
      const data = (await response.json()) as { message?: string };
      if (data.message) message = data.message;
    } catch {
      /* keep default message */
    }
    throw new Error(message);
  }
  return (await response.json()) as T;
}

/** Register (or log into) a username with the shared password. */
export async function registerUser(username: string, password: string): Promise<Auth> {
  const result = await rpc<Auth & { existed: boolean }>("quiz_register", {
    p_username: username,
    p_password: password
  });
  const auth: Auth = {
    user_id: result.user_id,
    username: result.username,
    token: result.token
  };
  saveAuth(auth);
  return auth;
}

export type SessionUpsert = {
  session_id: string;
  pack?: string;
  score_correct: number;
  score_total: number;
  meta?: Record<string, unknown>;
};

export async function upsertSession(auth: Auth, session: SessionUpsert, keepalive = false): Promise<void> {
  await rpc("quiz_upsert_session", {
    p_token: auth.token,
    p_session_id: session.session_id,
    p_pack: session.pack ?? null,
    p_score_correct: session.score_correct,
    p_score_total: session.score_total,
    p_meta: session.meta ?? {}
  }, keepalive);
}

// ------------------------------------------------------------ upload queue

function loadQueue(): QueuedChunk[] {
  try {
    return (JSON.parse(localStorage.getItem(LS_QUEUE) ?? "[]") as QueuedChunk[]) ?? [];
  } catch {
    return [];
  }
}

function saveQueue(queue: QueuedChunk[]): void {
  try {
    localStorage.setItem(LS_QUEUE, JSON.stringify(queue.slice(-QUEUE_MAX)));
  } catch {
    /* storage full: drop oldest until it fits */
    try {
      localStorage.setItem(LS_QUEUE, JSON.stringify(queue.slice(-Math.floor(QUEUE_MAX / 2))));
    } catch {
      /* give up persisting; in-memory retry still applies */
    }
  }
}

async function sendChunk(auth: Auth, chunk: QueuedChunk, keepalive = false): Promise<void> {
  await rpc("quiz_ingest_chunk", {
    p_token: auth.token,
    p_session_id: chunk.session_id,
    p_seq: chunk.seq,
    p_events: chunk.events
  }, keepalive);
}

/** Upload one chunk; on failure persist it for later retry. */
export async function uploadChunk(auth: Auth, chunk: QueuedChunk, keepalive = false): Promise<void> {
  if (!apiConfigured()) return;
  try {
    await sendChunk(auth, chunk, keepalive);
  } catch (err) {
    // A rejected token will never succeed later: drop instead of queueing.
    if (err instanceof Error && err.message === "invalid_token") return;
    const queue = loadQueue();
    queue.push(chunk);
    saveQueue(queue);
  }
}

/** Drain persisted chunks (call on startup and after each successful flush). */
export async function drainQueue(auth: Auth): Promise<void> {
  if (!apiConfigured()) return;
  const queue = loadQueue();
  if (!queue.length) return;
  const remaining: QueuedChunk[] = [];
  for (const chunk of queue) {
    try {
      await sendChunk(auth, chunk);
    } catch {
      remaining.push(chunk);
    }
  }
  saveQueue(remaining);
}
