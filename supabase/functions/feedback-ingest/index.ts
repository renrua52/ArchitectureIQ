// ArchitectureIQ feedback collector.
//
// Deploy this function with Supabase JWT verification disabled: it authenticates
// the existing Python client with FEEDBACK_INGEST_TOKEN instead.  The service
// role key is read only inside the Edge Function and is never returned.

const TRACE_SCHEMA_VERSION = "1.0";
const EVENT_SCHEMA_VERSION = "1.0";
const MAX_REQUEST_BYTES = 1_048_576; // 1 MiB, measured after the request is read.
const MAX_EVENTS_PER_REQUEST = 500;
const MAX_IDENTIFIER_LENGTH = 200;
const MAX_COMMENT_LENGTH = 2_000;
const MAX_EVENT_SEQUENCE = 2_147_483_647; // PostgreSQL integer upper bound.
const MAX_SAFE_JSON_INTEGER = Number.MAX_SAFE_INTEGER;
const EVENT_WRITE_TIMEOUT_MS = 8_000;
const OUTCOME_WRITE_TIMEOUT_MS = 2_000;
const OBSERVER_REVISION = "obs2";

const EVENT_TYPES = new Set([
  "answer_submitted",
  "custom_setting_proposed",
  "custom_setting_rejected",
  "custom_run_completed",
  "custom_run_failed",
  "comment_submitted",
  "question_presented",
  "question_reaction_submitted",
]);

const COMMENT_CATEGORIES = new Set([
  "question_quality",
  "answer_or_result",
  "custom_setting",
  "bug",
  "suggestion",
  "other",
]);

const ENVELOPE_KEYS = new Set([
  "schema_version",
  "envelope_type",
  "trace_id",
  "session_id",
  "created_at",
  "event_count",
  "events",
]);

const EVENT_KEYS = new Set([
  "schema_version",
  "event_id",
  "event_type",
  "occurred_at",
  "session_id",
  "question_id",
  "question_version",
  "payload",
  "sequence",
]);

const RFC3339_PATTERN =
  /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,9})?(?:Z|[+-]\d{2}:\d{2})$/;

type JsonObject = Record<string, unknown>;

interface FeedbackEvent {
  schema_version: string;
  event_id: string;
  event_type: string;
  occurred_at: string;
  session_id: string;
  question_id: string;
  question_version: string;
  payload: JsonObject;
  sequence: number;
}

interface SessionTraceEnvelope {
  schema_version: string;
  envelope_type: "session_trace";
  trace_id: string;
  session_id: string;
  created_at: string;
  event_count: number;
  events: FeedbackEvent[];
}

type OutcomeClass =
  | "success"
  | "client_rejection"
  | "service_failure";
type SubmissionKind = "session_trace" | "single_comment" | "unknown";
type StorageState = "confirmed" | "not_attempted" | "not_committed" | "unknown";

interface OutcomeCounts {
  requested: number | null;
  accepted: number | null;
  duplicate: number | null;
  conflicting: number | null;
  rejected: number | null;
}

interface IngestResult {
  requested: number;
  newEvents: number;
  accepted: number;
  duplicate: number;
  conflicting: number;
  rejected: number;
  committed: boolean;
}

interface OutcomeDetails {
  outcomeClass: OutcomeClass;
  outcomeCode: string;
  httpStatus: number;
  submissionKind: SubmissionKind;
  counts: OutcomeCounts;
  storageState: StorageState;
  retryable: boolean;
}

interface OutcomeContext {
  requestId: string;
  startedAt: string;
  startedMonotonic: number;
  supabaseUrl: string;
  serviceRoleKey: string;
}

class ValidationError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "ValidationError";
  }
}

class StorageError extends Error {
  storageState: StorageState;

  constructor(storageState: StorageState) {
    super("feedback storage failed");
    this.name = "StorageError";
    this.storageState = storageState;
  }
}

function isObject(value: unknown): value is JsonObject {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function unicodeCodePointLength(value: string): number {
  return Array.from(value).length;
}

function hasLoneSurrogate(value: string): boolean {
  for (let index = 0; index < value.length; index += 1) {
    const codeUnit = value.charCodeAt(index);
    if (codeUnit >= 0xD800 && codeUnit <= 0xDBFF) {
      const next = value.charCodeAt(index + 1);
      if (!(next >= 0xDC00 && next <= 0xDFFF)) {
        return true;
      }
      index += 1;
    } else if (codeUnit >= 0xDC00 && codeUnit <= 0xDFFF) {
      return true;
    }
  }
  return false;
}

function validateJsonInteroperability(value: unknown): void {
  const pending = [value];
  while (pending.length > 0) {
    const item = pending.pop();
    if (typeof item === "string") {
      if (hasLoneSurrogate(item)) {
        throw new ValidationError(
          "request body cannot contain unpaired Unicode surrogates",
        );
      }
    } else if (typeof item === "number") {
      if (!Number.isFinite(item)) {
        throw new ValidationError(
          "request body must contain only finite JSON numbers",
        );
      }
      if (Number.isInteger(item) && Math.abs(item) > MAX_SAFE_JSON_INTEGER) {
        throw new ValidationError(
          "request body contains an integer-valued JSON number outside the " +
            "interoperable safe-integer range",
        );
      }
    } else if (Array.isArray(item)) {
      for (const child of item) {
        pending.push(child);
      }
    } else if (isObject(item)) {
      for (const [key, child] of Object.entries(item)) {
        pending.push(key, child);
      }
    }
  }
}

function hasOwn(value: JsonObject, key: string): boolean {
  return Object.prototype.hasOwnProperty.call(value, key);
}

function requireExactKeys(
  value: JsonObject,
  allowed: Set<string>,
  path: string,
): void {
  for (const key of allowed) {
    if (!hasOwn(value, key)) {
      throw new ValidationError(`${path} is missing ${key}`);
    }
  }
  for (const key of Object.keys(value)) {
    if (!allowed.has(key)) {
      throw new ValidationError(`${path} contains an unsupported field`);
    }
  }
}

function requireIdentifier(value: unknown, path: string): string {
  if (typeof value !== "string") {
    throw new ValidationError(`${path} must be a string`);
  }
  const length = unicodeCodePointLength(value);
  if (
    length < 1 ||
    length > MAX_IDENTIFIER_LENGTH ||
    value !== value.trim() ||
    value.includes("\r") ||
    value.includes("\n")
  ) {
    throw new ValidationError(`${path} is not a valid identifier`);
  }
  return value;
}

function requireRfc3339(value: unknown, path: string): string {
  if (
    typeof value !== "string" ||
    !RFC3339_PATTERN.test(value) ||
    !Number.isFinite(Date.parse(value))
  ) {
    throw new ValidationError(`${path} must be an RFC 3339 timestamp`);
  }
  return value;
}

function requireInteger(value: unknown, path: string): number {
  if (typeof value !== "number" || !Number.isSafeInteger(value)) {
    throw new ValidationError(`${path} must be an integer`);
  }
  return value;
}

function requireOptionalIdentifier(payload: JsonObject, key: string): void {
  if (hasOwn(payload, key) && payload[key] !== null) {
    requireIdentifier(payload[key], `event.payload.${key}`);
  }
}

function validateSharedPayloadContext(payload: JsonObject): void {
  for (const key of [
    "attempt_id",
    "release_id",
    "family",
    "dataset_id",
    "question_type",
    "selection_metric",
  ]) {
    requireOptionalIdentifier(payload, key);
  }
  if (
    hasOwn(payload, "budget") &&
    payload.budget !== null &&
    !isObject(payload.budget)
  ) {
    throw new ValidationError("event.payload.budget must be an object or null");
  }
}

function validatePayload(eventType: string, value: unknown): JsonObject {
  if (!isObject(value)) {
    throw new ValidationError("event.payload must be an object");
  }
  validateSharedPayloadContext(value);

  switch (eventType) {
    case "answer_submitted": {
      requireIdentifier(value.selected_letter, "event.payload.selected_letter");
      requireOptionalIdentifier(value, "selected_candidate_id");
      if (hasOwn(value, "is_correct") && typeof value.is_correct !== "boolean") {
        throw new ValidationError("event.payload.is_correct must be a boolean");
      }
      break;
    }
    case "custom_setting_proposed":
    case "custom_setting_rejected": {
      if (!isObject(value.setting)) {
        throw new ValidationError("event.payload.setting must be an object");
      }
      if (
        hasOwn(value, "label") &&
        value.label !== null &&
        typeof value.label !== "string"
      ) {
        throw new ValidationError("event.payload.label must be a string or null");
      }
      for (const key of ["n_seeds", "base_seed"]) {
        if (hasOwn(value, key) && value[key] !== null) {
          requireInteger(value[key], `event.payload.${key}`);
        }
      }
      if (
        hasOwn(value, "inherited_from") &&
        value.inherited_from !== null &&
        !isObject(value.inherited_from)
      ) {
        throw new ValidationError(
          "event.payload.inherited_from must be an object or null",
        );
      }
      break;
    }
    case "custom_run_completed":
    case "custom_run_failed": {
      if (!isObject(value.run)) {
        throw new ValidationError("event.payload.run must be an object");
      }
      const expectedStatus = eventType === "custom_run_completed"
        ? "completed"
        : "failed";
      if (value.run.status !== expectedStatus) {
        throw new ValidationError(
          "event.payload.run.status does not match event.event_type",
        );
      }
      break;
    }
    case "comment_submitted": {
      if (
        typeof value.category !== "string" ||
        !COMMENT_CATEGORIES.has(value.category)
      ) {
        throw new ValidationError(
          "event.payload.category is not a supported category",
        );
      }
      if (typeof value.text !== "string") {
        throw new ValidationError("event.payload.text is not a valid comment");
      }
      const commentLength = unicodeCodePointLength(value.text);
      if (
        value.text !== value.text.trim() ||
        commentLength < 1 ||
        commentLength > MAX_COMMENT_LENGTH
      ) {
        throw new ValidationError("event.payload.text is not a valid comment");
      }
      break;
    }
    case "question_reaction_submitted": {
      if (value.reaction !== "surprise") {
        throw new ValidationError(
          "event.payload.reaction must be surprise",
        );
      }
      if (typeof value.value !== "boolean") {
        throw new ValidationError("event.payload.value must be a boolean");
      }
      if (value.timing !== "after_reveal") {
        throw new ValidationError(
          "event.payload.timing must be after_reveal",
        );
      }
      requireIdentifier(value.attempt_id, "event.payload.attempt_id");
      if (hasOwn(value, "release_id")) {
        requireIdentifier(value.release_id, "event.payload.release_id");
      }
      break;
    }
    case "question_presented": {
      for (const key of [
        "attempt_id",
        "release_id",
        "decision_id",
        "policy_version",
        "source",
      ]) {
        requireIdentifier(value[key], `event.payload.${key}`);
      }
      if (!["exploit", "explore", "fallback", "manual"].includes(
        String(value.mode),
      )) {
        throw new ValidationError("event.payload.mode is not supported");
      }
      if (!["initial", "next", "random", "picker"].includes(
        String(value.source),
      )) {
        throw new ValidationError("event.payload.source is not supported");
      }
      if (
        typeof value.propensity !== "number" ||
        !Number.isFinite(value.propensity) ||
        value.propensity <= 0 ||
        value.propensity > 1
      ) {
        throw new ValidationError(
          "event.payload.propensity must be in (0, 1]",
        );
      }
      const position = requireInteger(
        value.position,
        "event.payload.position",
      );
      if (position <= 0) {
        throw new ValidationError("event.payload.position must be positive");
      }
      break;
    }
    default:
      throw new ValidationError("event.event_type is not supported");
  }
  return value;
}

function validateEvent(
  value: unknown,
  envelopeSessionId: string,
): FeedbackEvent {
  if (!isObject(value)) {
    throw new ValidationError("event must be an object");
  }
  requireExactKeys(value, EVENT_KEYS, "event");

  if (value.schema_version !== EVENT_SCHEMA_VERSION) {
    throw new ValidationError("event.schema_version is not supported");
  }
  if (typeof value.event_type !== "string" || !EVENT_TYPES.has(value.event_type)) {
    throw new ValidationError("event.event_type is not supported");
  }

  const sessionId = requireIdentifier(value.session_id, "event.session_id");
  if (sessionId !== envelopeSessionId) {
    throw new ValidationError("event.session_id does not match the envelope");
  }

  const sequence = requireInteger(value.sequence, "event.sequence");
  if (sequence <= 0 || sequence > MAX_EVENT_SEQUENCE) {
    throw new ValidationError(
      `event.sequence must be between 1 and ${MAX_EVENT_SEQUENCE}`,
    );
  }

  return {
    schema_version: EVENT_SCHEMA_VERSION,
    event_id: requireIdentifier(value.event_id, "event.event_id"),
    event_type: value.event_type,
    occurred_at: requireRfc3339(value.occurred_at, "event.occurred_at"),
    session_id: sessionId,
    question_id: requireIdentifier(value.question_id, "event.question_id"),
    question_version: requireIdentifier(
      value.question_version,
      "event.question_version",
    ),
    payload: validatePayload(value.event_type, value.payload),
    sequence,
  };
}

function validateEnvelope(value: unknown): SessionTraceEnvelope {
  if (!isObject(value)) {
    throw new ValidationError("request body must be a session trace object");
  }
  requireExactKeys(value, ENVELOPE_KEYS, "envelope");

  if (value.schema_version !== TRACE_SCHEMA_VERSION) {
    throw new ValidationError("envelope.schema_version is not supported");
  }
  if (value.envelope_type !== "session_trace") {
    throw new ValidationError("envelope.envelope_type is not supported");
  }

  const traceId = requireIdentifier(value.trace_id, "envelope.trace_id");
  const sessionId = requireIdentifier(value.session_id, "envelope.session_id");
  const createdAt = requireRfc3339(value.created_at, "envelope.created_at");
  const eventCount = requireInteger(value.event_count, "envelope.event_count");

  if (!Array.isArray(value.events)) {
    throw new ValidationError("envelope.events must be an array");
  }
  if (
    value.events.length < 1 ||
    value.events.length > MAX_EVENTS_PER_REQUEST
  ) {
    throw new ValidationError("envelope.events has an unsupported batch size");
  }
  if (eventCount !== value.events.length) {
    throw new ValidationError("envelope.event_count does not match events");
  }

  const seenEventIds = new Set<string>();
  let previousSequence = 0;
  const events = value.events.map((event) => {
    const validated = validateEvent(event, sessionId);
    if (seenEventIds.has(validated.event_id)) {
      throw new ValidationError("envelope.events contains duplicate event_id values");
    }
    if (validated.sequence <= previousSequence) {
      throw new ValidationError(
        "event.sequence values must be strictly increasing",
      );
    }
    seenEventIds.add(validated.event_id);
    previousSequence = validated.sequence;
    return validated;
  });

  return {
    schema_version: TRACE_SCHEMA_VERSION,
    envelope_type: "session_trace",
    trace_id: traceId,
    session_id: sessionId,
    created_at: createdAt,
    event_count: eventCount,
    events,
  };
}

function potentialEventCount(value: unknown): number | null {
  if (isObject(value) && Array.isArray(value.events)) {
    return value.events.length;
  }
  return null;
}

function submissionKind(envelope: SessionTraceEnvelope): SubmissionKind {
  return envelope.event_count === 1 &&
      envelope.events[0].event_type === "comment_submitted"
    ? "single_comment"
    : "session_trace";
}

function successfulOutcomeCode(accepted: number, duplicate: number): string {
  if (accepted > 0 && duplicate === 0) {
    return "accepted_only";
  }
  if (accepted === 0 && duplicate > 0) {
    return "duplicate_only";
  }
  return "mixed_success";
}

function safeOutcomeLog(
  requestId: string,
  outcomeCode: string,
  httpStatus: number,
  includedInRate: boolean,
): void {
  console.error(
    JSON.stringify({
      request_id: requestId,
      outcome_code: outcomeCode,
      http_status: httpStatus,
      included_in_rate: includedInRate,
    }),
  );
}

function responseHeaders(requestId: string): Headers {
  return new Headers({
    "Cache-Control": "no-store",
    "Content-Type": "application/json; charset=utf-8",
    "X-Content-Type-Options": "nosniff",
    "X-Request-ID": requestId,
  });
}

function receiptResponse(
  status: number,
  requestId: string,
  accepted: number,
  duplicate: number,
  rejected: number,
  error?: { code: string; message: string },
  conflict = 0,
): Response {
  const body: JsonObject = {
    accepted,
    duplicate,
    conflict,
    rejected,
    request_id: requestId,
  };
  if (error) {
    body.error = error;
  }
  return new Response(JSON.stringify(body), {
    status,
    headers: responseHeaders(requestId),
  });
}

async function tokenMatches(received: string, expected: string): Promise<boolean> {
  const encoder = new TextEncoder();
  const [receivedHash, expectedHash] = await Promise.all([
    crypto.subtle.digest("SHA-256", encoder.encode(received)),
    crypto.subtle.digest("SHA-256", encoder.encode(expected)),
  ]);
  const left = new Uint8Array(receivedHash);
  const right = new Uint8Array(expectedHash);
  let difference = left.length ^ right.length;
  for (let index = 0; index < left.length; index += 1) {
    difference |= left[index] ^ right[index];
  }
  return difference === 0;
}

async function authorized(request: Request, expectedToken: string): Promise<boolean> {
  const authorization = request.headers.get("Authorization") ?? "";
  const match = /^Bearer ([^\s]+)$/i.exec(authorization);
  if (!match) {
    return false;
  }
  return await tokenMatches(match[1], expectedToken);
}

async function readJsonBody(request: Request): Promise<unknown> {
  const contentType = request.headers.get("Content-Type") ?? "";
  if (!/^application\/json(?:\s*;|$)/i.test(contentType)) {
    throw new ValidationError("Content-Type must be application/json");
  }

  const contentLength = request.headers.get("Content-Length");
  if (contentLength !== null) {
    if (!/^\d+$/.test(contentLength)) {
      throw new ValidationError("Content-Length is invalid");
    }
    if (Number(contentLength) > MAX_REQUEST_BYTES) {
      throw new RangeError("request body is too large");
    }
  }

  const bytes = new Uint8Array(await request.arrayBuffer());
  if (bytes.byteLength > MAX_REQUEST_BYTES) {
    throw new RangeError("request body is too large");
  }

  let text: string;
  try {
    text = new TextDecoder("utf-8", { fatal: true }).decode(bytes);
  } catch {
    throw new ValidationError("request body is not valid UTF-8");
  }
  try {
    const value: unknown = JSON.parse(text);
    validateJsonInteroperability(value);
    return value;
  } catch (error) {
    if (error instanceof ValidationError) {
      throw error;
    }
    throw new ValidationError("request body is not valid interoperable JSON");
  }
}

async function recordOutcome(
  context: OutcomeContext,
  details: OutcomeDetails,
): Promise<void> {
  let endpoint: URL;
  try {
    endpoint = new URL(
      "/rest/v1/feedback_ingest_request_outcomes",
      context.supabaseUrl,
    );
  } catch {
    safeOutcomeLog(
      context.requestId,
      "outcome_record_failed",
      details.httpStatus,
      details.outcomeClass !== "success" || details.httpStatus === 200,
    );
    return;
  }
  endpoint.searchParams.set("on_conflict", "request_id");

  const finishedAt = new Date().toISOString();
  const durationMs = Math.max(
    0,
    Math.round(performance.now() - context.startedMonotonic),
  );
  const row = {
    request_id: context.requestId,
    schema_version: "1.1",
    started_at: context.startedAt,
    finished_at: finishedAt,
    duration_ms: durationMs,
    method: "POST",
    authenticated: true,
    included_in_rate: true,
    outcome_class: details.outcomeClass,
    outcome_code: details.outcomeCode,
    http_status: details.httpStatus,
    submission_kind: details.submissionKind,
    requested_event_count: details.counts.requested,
    accepted_event_count: details.counts.accepted,
    duplicate_event_count: details.counts.duplicate,
    conflicting_event_count: details.counts.conflicting,
    rejected_event_count: details.counts.rejected,
    storage_state: details.storageState,
    retryable: details.retryable,
    observer_revision: OBSERVER_REVISION,
  };

  try {
    const response = await fetch(endpoint, {
      method: "POST",
      headers: {
        apikey: context.serviceRoleKey,
        Authorization: `Bearer ${context.serviceRoleKey}`,
        "Content-Type": "application/json",
        Prefer: "resolution=ignore-duplicates,return=minimal",
      },
      body: JSON.stringify(row),
      signal: AbortSignal.timeout(OUTCOME_WRITE_TIMEOUT_MS),
    });
    if (!response.ok) {
      safeOutcomeLog(
        context.requestId,
        "outcome_record_failed",
        details.httpStatus,
        true,
      );
    }
  } catch {
    safeOutcomeLog(
      context.requestId,
      "outcome_record_failed",
      details.httpStatus,
      true,
    );
  }
}

function scheduleOutcome(
  context: OutcomeContext,
  details: OutcomeDetails,
): void {
  const task = recordOutcome(context, details);
  const runtime = (
    globalThis as unknown as {
      EdgeRuntime?: { waitUntil(promise: Promise<unknown>): void };
    }
  ).EdgeRuntime;
  if (runtime) {
    runtime.waitUntil(task);
  } else {
    // Supabase provides EdgeRuntime.waitUntil. This fallback keeps local
    // harnesses fail-open without introducing an unhandled rejection.
    void task;
  }
}

async function ingestEvents(
  envelope: SessionTraceEnvelope,
  requestId: string,
  supabaseUrl: string,
  serviceRoleKey: string,
): Promise<IngestResult> {
  let endpoint: URL;
  try {
    endpoint = new URL("/rest/v1/rpc/feedback_ingest_events", supabaseUrl);
  } catch {
    throw new StorageError("not_attempted");
  }
  let response: Response;
  try {
    response = await fetch(endpoint, {
      method: "POST",
      headers: {
        apikey: serviceRoleKey,
        Authorization: `Bearer ${serviceRoleKey}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        p_request_id: requestId,
        p_trace_id: envelope.trace_id,
        p_trace_created_at: envelope.created_at,
        p_events: envelope.events,
      }),
      signal: AbortSignal.timeout(EVENT_WRITE_TIMEOUT_MS),
    });
  } catch {
    throw new StorageError("unknown");
  }

  if (!response.ok) {
    // Never include the PostgREST body in logs or responses: configuration and
    // database diagnostics belong in the Supabase dashboard, not client output.
    console.error(
      JSON.stringify({
        request_id: requestId,
        code: "postgrest_ingest_rpc_failed",
        status: response.status,
      }),
    );
    throw new StorageError(
      response.status >= 400 && response.status < 500
        ? "not_committed"
        : "unknown",
    );
  }

  let value: unknown;
  try {
    value = await response.json();
  } catch {
    throw new StorageError("unknown");
  }
  if (!Array.isArray(value) || value.length !== 1 || !isObject(value[0])) {
    throw new StorageError("unknown");
  }
  const row = value[0];
  const expectedKeys = new Set([
    "requested_event_count",
    "new_event_count",
    "accepted_event_count",
    "duplicate_event_count",
    "conflicting_event_count",
    "rejected_event_count",
    "committed",
  ]);
  if (
    Object.keys(row).length !== expectedKeys.size ||
    Object.keys(row).some((key) => !expectedKeys.has(key))
  ) {
    throw new StorageError("unknown");
  }
  const countNames = [
    "requested_event_count",
    "new_event_count",
    "accepted_event_count",
    "duplicate_event_count",
    "conflicting_event_count",
    "rejected_event_count",
  ] as const;
  for (const name of countNames) {
    const count = row[name];
    if (
      typeof count !== "number" ||
      !Number.isSafeInteger(count) ||
      count < 0
    ) {
      throw new StorageError("unknown");
    }
  }
  if (typeof row.committed !== "boolean") {
    throw new StorageError("unknown");
  }

  const result: IngestResult = {
    requested: row.requested_event_count as number,
    newEvents: row.new_event_count as number,
    accepted: row.accepted_event_count as number,
    duplicate: row.duplicate_event_count as number,
    conflicting: row.conflicting_event_count as number,
    rejected: row.rejected_event_count as number,
    committed: row.committed as boolean,
  };
  const classificationTotal = result.newEvents + result.duplicate +
    result.conflicting;
  const receiptTotal = result.accepted + result.duplicate + result.rejected;
  const successIsValid = result.committed &&
    result.conflicting === 0 && result.rejected === 0 &&
    result.accepted === result.newEvents && receiptTotal === result.requested;
  const conflictIsValid = !result.committed &&
    result.conflicting > 0 && result.accepted === 0 &&
    result.rejected === result.newEvents + result.conflicting &&
    receiptTotal === result.requested;
  if (
    result.requested !== envelope.event_count ||
    classificationTotal !== result.requested ||
    (!successIsValid && !conflictIsValid)
  ) {
    throw new StorageError("unknown");
  }
  return result;
}

Deno.serve(async (request: Request): Promise<Response> => {
  const requestId = crypto.randomUUID();
  const startedAt = new Date().toISOString();
  const startedMonotonic = performance.now();

  if (request.method !== "POST") {
    safeOutcomeLog(requestId, "method_not_allowed", 405, false);
    const response = receiptResponse(405, requestId, 0, 0, 0, {
      code: "METHOD_NOT_ALLOWED",
      message: "Only POST is supported",
    });
    response.headers.set("Allow", "POST");
    return response;
  }

  const ingestToken = Deno.env.get("FEEDBACK_INGEST_TOKEN")?.trim() ?? "";
  const supabaseUrl = Deno.env.get("SUPABASE_URL")?.trim() ?? "";
  const serviceRoleKey =
    Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")?.trim() ?? "";

  if (!ingestToken || !supabaseUrl || !serviceRoleKey) {
    safeOutcomeLog(requestId, "service_unavailable", 503, false);
    return receiptResponse(503, requestId, 0, 0, 0, {
      code: "SERVICE_UNAVAILABLE",
      message: "Feedback ingestion is temporarily unavailable",
    });
  }

  if (!(await authorized(request, ingestToken))) {
    safeOutcomeLog(requestId, "unauthorized", 401, false);
    const response = receiptResponse(401, requestId, 0, 0, 0, {
      code: "UNAUTHORIZED",
      message: "A valid bearer token is required",
    });
    response.headers.set("WWW-Authenticate", "Bearer");
    return response;
  }

  const outcomeContext: OutcomeContext = {
    requestId,
    startedAt,
    startedMonotonic,
    supabaseUrl,
    serviceRoleKey,
  };

  let rawEnvelope: unknown;
  try {
    rawEnvelope = await readJsonBody(request);
  } catch (error) {
    if (error instanceof RangeError) {
      scheduleOutcome(outcomeContext, {
        outcomeClass: "client_rejection",
        outcomeCode: "request_too_large",
        httpStatus: 413,
        submissionKind: "unknown",
        counts: {
          requested: null,
          accepted: 0,
          duplicate: 0,
          conflicting: null,
          rejected: 0,
        },
        storageState: "not_attempted",
        retryable: false,
      });
      return receiptResponse(413, requestId, 0, 0, 0, {
        code: "REQUEST_TOO_LARGE",
        message: `Request body must not exceed ${MAX_REQUEST_BYTES} bytes`,
      });
    }
    scheduleOutcome(outcomeContext, {
      outcomeClass: "client_rejection",
      outcomeCode: "invalid_request",
      httpStatus: 400,
      submissionKind: "unknown",
      counts: {
        requested: null,
        accepted: 0,
        duplicate: 0,
        conflicting: null,
        rejected: 0,
      },
      storageState: "not_attempted",
      retryable: false,
    });
    return receiptResponse(400, requestId, 0, 0, 0, {
      code: "INVALID_REQUEST",
      message: error instanceof ValidationError
        ? error.message
        : "Request body could not be read",
    });
  }

  const potentialCount = potentialEventCount(rawEnvelope);
  const rejected = potentialCount ?? 0;
  let envelope: SessionTraceEnvelope;
  try {
    envelope = validateEnvelope(rawEnvelope);
  } catch (error) {
    scheduleOutcome(outcomeContext, {
      outcomeClass: "client_rejection",
      outcomeCode: "invalid_envelope",
      httpStatus: 400,
      submissionKind: "unknown",
      counts: potentialCount === null
        ? {
          requested: null,
          accepted: 0,
          duplicate: 0,
          conflicting: null,
          rejected: 0,
        }
        : {
          requested: potentialCount,
          accepted: 0,
          duplicate: 0,
          conflicting: null,
          rejected: potentialCount,
        },
      storageState: "not_attempted",
      retryable: false,
    });
    return receiptResponse(400, requestId, 0, 0, rejected, {
      code: "INVALID_ENVELOPE",
      message: error instanceof ValidationError
        ? error.message
        : "Session trace is invalid",
    });
  }

  try {
    const ingest = await ingestEvents(
      envelope,
      requestId,
      supabaseUrl,
      serviceRoleKey,
    );
    if (ingest.conflicting > 0) {
      scheduleOutcome(outcomeContext, {
        outcomeClass: "client_rejection",
        outcomeCode: "event_id_conflict",
        httpStatus: 409,
        submissionKind: submissionKind(envelope),
        counts: {
          requested: ingest.requested,
          accepted: 0,
          duplicate: ingest.duplicate,
          conflicting: ingest.conflicting,
          rejected: ingest.rejected,
        },
        storageState: "confirmed",
        retryable: false,
      });
      return receiptResponse(
        409,
        requestId,
        0,
        ingest.duplicate,
        ingest.rejected,
        {
          code: "EVENT_ID_CONFLICT",
          message:
            "One or more event IDs already store different logical content; " +
            "the batch was not inserted",
        },
        ingest.conflicting,
      );
    }
    scheduleOutcome(outcomeContext, {
      outcomeClass: "success",
      outcomeCode: successfulOutcomeCode(ingest.accepted, ingest.duplicate),
      httpStatus: 200,
      submissionKind: submissionKind(envelope),
      counts: {
        requested: ingest.requested,
        accepted: ingest.accepted,
        duplicate: ingest.duplicate,
        conflicting: 0,
        rejected: 0,
      },
      storageState: "confirmed",
      retryable: false,
    });
    return receiptResponse(
      200,
      requestId,
      ingest.accepted,
      ingest.duplicate,
      0,
    );
  } catch (error) {
    const storageState = error instanceof StorageError
      ? error.storageState
      : "unknown";
    scheduleOutcome(outcomeContext, {
      outcomeClass: "service_failure",
      outcomeCode: "storage_unavailable",
      httpStatus: 502,
      submissionKind: submissionKind(envelope),
      counts: {
        requested: envelope.event_count,
        accepted: null,
        duplicate: null,
        conflicting: null,
        rejected: null,
      },
      storageState,
      retryable: true,
    });
    safeOutcomeLog(requestId, "storage_unavailable", 502, true);
    return receiptResponse(502, requestId, 0, 0, 0, {
      code: "STORAGE_UNAVAILABLE",
      message: "Feedback storage result is unknown; retry is safe",
    });
  }
});
