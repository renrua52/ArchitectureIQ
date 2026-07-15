/**
 * ArchitectureIQ quiz telemetry ingest (Supabase Edge Function).
 *
 * POST /functions/v1/telemetry
 * Auth: Authorization: Bearer <TELEMETRY_API_KEY>
 * Body: one event object, or { "events": [ ... ] } (max 50)
 *
 * Deploy with JWT verification OFF (we use our own bearer secret).
 */
import { createClient } from "https://esm.sh/@supabase/supabase-js@2.49.8";

const ALLOWED_EVENTS = new Set([
  "session_start",
  "question_view",
  "stage_change",
  "answer_submit",
  "question_leave",
]);

type TelemetryEvent = {
  session_id: string;
  event_type: string;
  question_id?: string | null;
  duration_ms?: number | null;
  payload?: Record<string, unknown>;
};

function corsHeaders(req: Request): HeadersInit {
  const allowed = (Deno.env.get("CORS_ORIGINS") ??
    "http://127.0.0.1:5173,http://localhost:5173")
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean);
  const origin = req.headers.get("Origin") ?? "";
  const headers: Record<string, string> = {
    "Access-Control-Allow-Headers": "authorization, content-type, x-client-info, apikey",
    "Access-Control-Allow-Methods": "POST, GET, OPTIONS",
  };
  if (origin && allowed.includes(origin)) {
    headers["Access-Control-Allow-Origin"] = origin;
    headers["Vary"] = "Origin";
  }
  return headers;
}

function json(status: number, body: unknown, req: Request): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: {
      ...corsHeaders(req),
      "Content-Type": "application/json",
    },
  });
}

function authorize(req: Request): Response | null {
  const key = (Deno.env.get("TELEMETRY_API_KEY") ?? "").trim();
  if (!key) {
    return json(503, { detail: "TELEMETRY_API_KEY is not configured" }, req);
  }
  const auth = req.headers.get("Authorization") ?? "";
  if (!auth.startsWith("Bearer ")) {
    return json(401, { detail: "Missing bearer token" }, req);
  }
  if (auth.slice("Bearer ".length).trim() !== key) {
    return json(401, { detail: "Invalid token" }, req);
  }
  return null;
}

function normalize(raw: unknown): TelemetryEvent {
  if (!raw || typeof raw !== "object") {
    throw new Error("Invalid event");
  }
  const e = raw as Record<string, unknown>;
  const session_id = String(e.session_id ?? "");
  const event_type = String(e.event_type ?? "");
  if (session_id.length < 8 || session_id.length > 128) {
    throw new Error("session_id must be 8–128 chars");
  }
  if (!ALLOWED_EVENTS.has(event_type)) {
    throw new Error(`Unsupported event_type: ${event_type}`);
  }
  const duration_ms =
    e.duration_ms === undefined || e.duration_ms === null
      ? null
      : Number(e.duration_ms);
  if (duration_ms !== null && (Number.isNaN(duration_ms) || duration_ms < 0)) {
    throw new Error("duration_ms must be non-negative");
  }
  const payload =
    e.payload && typeof e.payload === "object" && !Array.isArray(e.payload)
      ? { schema_version: 1, ...(e.payload as Record<string, unknown>) }
      : { schema_version: 1 };
  return {
    session_id,
    event_type,
    question_id: e.question_id == null ? null : String(e.question_id),
    duration_ms,
    payload,
  };
}

Deno.serve(async (req) => {
  if (req.method === "OPTIONS") {
    return new Response("ok", { headers: corsHeaders(req) });
  }

  if (req.method === "GET") {
    const ready =
      Boolean(Deno.env.get("TELEMETRY_API_KEY")?.trim()) &&
      Boolean(Deno.env.get("SUPABASE_URL")) &&
      Boolean(Deno.env.get("SUPABASE_SERVICE_ROLE_KEY"));
    return json(200, { status: ready ? "ok" : "misconfigured" }, req);
  }

  if (req.method !== "POST") {
    return json(405, { detail: "Method not allowed" }, req);
  }

  const denied = authorize(req);
  if (denied) return denied;

  let body: unknown;
  try {
    body = await req.json();
  } catch {
    return json(400, { detail: "Invalid JSON" }, req);
  }

  let eventsRaw: unknown[];
  if (body && typeof body === "object" && Array.isArray((body as { events?: unknown }).events)) {
    eventsRaw = (body as { events: unknown[] }).events;
  } else {
    eventsRaw = [body];
  }
  if (eventsRaw.length < 1 || eventsRaw.length > 50) {
    return json(400, { detail: "Provide 1–50 events" }, req);
  }

  let events: TelemetryEvent[];
  try {
    events = eventsRaw.map(normalize);
  } catch (err) {
    return json(400, { detail: err instanceof Error ? err.message : "Bad event" }, req);
  }

  const supabaseUrl = Deno.env.get("SUPABASE_URL") ?? "";
  const serviceKey = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY") ?? "";
  if (!supabaseUrl || !serviceKey) {
    return json(503, { detail: "Supabase service credentials missing" }, req);
  }

  const supabase = createClient(supabaseUrl, serviceKey, {
    auth: { persistSession: false, autoRefreshToken: false },
  });

  const rows = events.map((e) => ({
    session_id: e.session_id,
    event_type: e.event_type,
    question_id: e.question_id,
    payload: e.payload ?? { schema_version: 1 },
    duration_ms: e.duration_ms,
  }));

  const { error } = await supabase.from("quiz_events").insert(rows);
  if (error) {
    return json(500, { detail: `Database error: ${error.message}` }, req);
  }

  return json(202, { accepted: rows.length }, req);
});
