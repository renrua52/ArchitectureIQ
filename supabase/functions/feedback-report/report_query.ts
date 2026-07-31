/** Pure parsing and RPC-parameter construction for feedback-report. */

export const REPORT_VIEWS = [
  "feedback_report_summary",
  "feedback_report_ingestion_summary",
  "feedback_report_authority_status",
  "feedback_report_business_snapshot",
  "feedback_report_registry_quality",
  "feedback_report_surprise_questions",
  "feedback_report_surprise_quality",
  "feedback_report_event_resolution",
  "feedback_report_sessions",
  "feedback_report_questions",
  "feedback_report_answers",
  "feedback_report_proposals",
  "feedback_report_comments",
] as const;

export type ReportView = (typeof REPORT_VIEWS)[number];

const REPORT_VIEW_SET = new Set<string>(REPORT_VIEWS);
const QUERY_KEYS = new Set([
  "view",
  "release_id",
  "family",
  "question_type",
  "question_id",
  "session_id",
  "attempt_id",
  "from",
  "to",
  "request_id",
  "event_id",
  "category",
  "limit",
  "offset",
]);
const COMMENT_CATEGORIES = new Set([
  "question_quality",
  "answer_or_result",
  "custom_setting",
  "bug",
  "suggestion",
  "other",
]);
const RFC3339_PATTERN =
  /^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})(?:\.(\d{1,9}))?(?:Z|([+-])(\d{2}):(\d{2}))$/;
const UUID_PATTERN =
  /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
const MAX_IDENTIFIER_LENGTH = 200;
export const DEFAULT_REPORT_LIMIT = 200;
export const MAX_REPORT_LIMIT = 1_000;
export const MAX_REPORT_OFFSET = 1_000_000;
const SINGLE_ROW_VIEWS = new Set<ReportView>([
  "feedback_report_summary",
  "feedback_report_ingestion_summary",
  "feedback_report_authority_status",
  "feedback_report_business_snapshot",
  "feedback_report_registry_quality",
  "feedback_report_surprise_quality",
  "feedback_report_event_resolution",
]);

export class ReportQueryError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "ReportQueryError";
  }
}

export interface ReportQuery {
  view: ReportView;
  releaseId: string | null;
  family: string | null;
  questionType: string | null;
  questionId: string | null;
  sessionId: string | null;
  attemptId: string | null;
  from: string | null;
  to: string | null;
  requestId: string | null;
  eventId: string | null;
  category: string | null;
  limit: number;
  offset: number;
}

function requireOneValue(params: URLSearchParams, key: string): string | null {
  const values = params.getAll(key);
  if (values.length > 1) {
    throw new ReportQueryError(
      `query parameter ${key} must appear at most once`,
    );
  }
  return values.length === 1 ? values[0] : null;
}

function optionalIdentifier(
  params: URLSearchParams,
  key: string,
): string | null {
  const value = requireOneValue(params, key);
  if (value === null) {
    return null;
  }
  const codePointLength = Array.from(value).length;
  if (
    codePointLength < 1 ||
    codePointLength > MAX_IDENTIFIER_LENGTH ||
    value !== value.trim() ||
    value.includes("\r") ||
    value.includes("\n")
  ) {
    throw new ReportQueryError(
      `query parameter ${key} is not a valid identifier`,
    );
  }
  return value;
}

function optionalTimestamp(
  params: URLSearchParams,
  key: "from" | "to",
): string | null {
  const value = requireOneValue(params, key);
  if (value === null) {
    return null;
  }
  try {
    timestampInstant(value);
  } catch {
    throw new ReportQueryError(`query parameter ${key} must be RFC 3339`);
  }
  return value;
}

function optionalUuid(
  params: URLSearchParams,
  key: "request_id",
): string | null {
  const value = requireOneValue(params, key);
  if (value === null) {
    return null;
  }
  if (!UUID_PATTERN.test(value)) {
    throw new ReportQueryError(`query parameter ${key} must be a UUID`);
  }
  return value;
}

function timestampInstant(value: string): bigint {
  const match = RFC3339_PATTERN.exec(value);
  if (match === null) {
    throw new Error("invalid timestamp");
  }
  const year = Number(match[1]);
  const month = Number(match[2]);
  const day = Number(match[3]);
  const hour = Number(match[4]);
  const minute = Number(match[5]);
  const second = Number(match[6]);
  const leapYear = year % 4 === 0 && (year % 100 !== 0 || year % 400 === 0);
  const monthLengths = [
    31,
    leapYear ? 29 : 28,
    31,
    30,
    31,
    30,
    31,
    31,
    30,
    31,
    30,
    31,
  ];
  const daysInMonth = month >= 1 && month <= 12 ? monthLengths[month - 1] : 0;
  const offsetHour = match[9] === undefined ? 0 : Number(match[9]);
  const offsetMinute = match[10] === undefined ? 0 : Number(match[10]);
  if (
    day < 1 ||
    day > daysInMonth ||
    hour > 23 ||
    minute > 59 ||
    second > 59 ||
    offsetHour > 23 ||
    offsetMinute > 59
  ) {
    throw new Error("invalid timestamp");
  }

  const local = new Date(0);
  local.setUTCFullYear(year, month - 1, day);
  local.setUTCHours(hour, minute, second, 0);
  const offsetDirection = match[8] === "+" ? 1 : match[8] === "-" ? -1 : 0;
  const offsetMilliseconds =
    offsetDirection * (offsetHour * 60 + offsetMinute) * 60_000;
  const fractionalNanoseconds = BigInt((match[7] ?? "").padEnd(9, "0") || "0");
  return (
    BigInt(local.getTime() - offsetMilliseconds) * 1_000_000n +
    fractionalNanoseconds
  );
}

function integerParameter(
  params: URLSearchParams,
  key: "limit" | "offset",
  fallback: number,
  minimum: number,
  maximum: number,
): number {
  const value = requireOneValue(params, key);
  if (value === null) {
    return fallback;
  }
  if (!/^\d+$/.test(value)) {
    throw new ReportQueryError(`query parameter ${key} must be an integer`);
  }
  const parsed = Number(value);
  if (!Number.isSafeInteger(parsed) || parsed < minimum || parsed > maximum) {
    throw new ReportQueryError(
      `query parameter ${key} must be between ${minimum} and ${maximum}`,
    );
  }
  return parsed;
}

export function parseReportQuery(input: URL | string): ReportQuery {
  const url = typeof input === "string" ? new URL(input) : input;
  for (const key of url.searchParams.keys()) {
    if (!QUERY_KEYS.has(key)) {
      throw new ReportQueryError("query contains an unsupported parameter");
    }
  }

  const rawView = requireOneValue(url.searchParams, "view");
  if (rawView === null || !REPORT_VIEW_SET.has(rawView)) {
    throw new ReportQueryError("query parameter view is not supported");
  }
  const category = optionalIdentifier(url.searchParams, "category");
  if (category !== null && !COMMENT_CATEGORIES.has(category)) {
    throw new ReportQueryError("query parameter category is not supported");
  }
  if (category !== null && rawView !== "feedback_report_comments") {
    throw new ReportQueryError(
      "query parameter category is supported only for feedback_report_comments",
    );
  }
  const from = optionalTimestamp(url.searchParams, "from");
  const to = optionalTimestamp(url.searchParams, "to");
  if (
    from !== null &&
    to !== null &&
    timestampInstant(from) >= timestampInstant(to)
  ) {
    throw new ReportQueryError("query parameter from must be earlier than to");
  }

  const releaseId = optionalIdentifier(url.searchParams, "release_id");
  const family = optionalIdentifier(url.searchParams, "family");
  const questionType = optionalIdentifier(url.searchParams, "question_type");
  const questionId = optionalIdentifier(url.searchParams, "question_id");
  const sessionId = optionalIdentifier(url.searchParams, "session_id");
  const attemptId = optionalIdentifier(url.searchParams, "attempt_id");
  const requestId = optionalUuid(url.searchParams, "request_id");
  const eventId = optionalIdentifier(url.searchParams, "event_id");
  if (requestId !== null && rawView !== "feedback_report_ingestion_summary") {
    throw new ReportQueryError(
      "query parameter request_id is supported only for feedback_report_ingestion_summary",
    );
  }
  if (eventId !== null && rawView !== "feedback_report_event_resolution") {
    throw new ReportQueryError(
      "query parameter event_id is supported only for feedback_report_event_resolution",
    );
  }
  if (rawView === "feedback_report_event_resolution" && eventId === null) {
    throw new ReportQueryError(
      "event resolution requires query parameter event_id",
    );
  }
  if (
    rawView === "feedback_report_authority_status" &&
    [
      releaseId,
      family,
      questionType,
      questionId,
      sessionId,
      attemptId,
      from,
      to,
    ].some((value) => value !== null)
  ) {
    throw new ReportQueryError("authority status does not support filters");
  }
  if (
    rawView === "feedback_report_ingestion_summary" &&
    [releaseId, family, questionType, questionId, sessionId, attemptId].some(
      (value) => value !== null,
    )
  ) {
    throw new ReportQueryError(
      "ingestion summary supports only server-time from/to and request_id filters",
    );
  }
  if (
    rawView === "feedback_report_registry_quality" &&
    [releaseId, family, questionType, questionId, sessionId, attemptId].some(
      (value) => value !== null,
    )
  ) {
    throw new ReportQueryError(
      "registry quality supports only event-time from/to filters",
    );
  }
  if (
    rawView === "feedback_report_event_resolution" &&
    [
      releaseId,
      family,
      questionType,
      questionId,
      sessionId,
      attemptId,
      from,
      to,
      requestId,
    ].some((value) => value !== null)
  ) {
    throw new ReportQueryError(
      "event resolution supports only the exact event_id filter",
    );
  }
  const offset = integerParameter(
    url.searchParams,
    "offset",
    0,
    0,
    MAX_REPORT_OFFSET,
  );
  if (SINGLE_ROW_VIEWS.has(rawView as ReportView) && offset !== 0) {
    throw new ReportQueryError("single-row report offset must be zero");
  }

  return {
    view: rawView as ReportView,
    releaseId,
    family,
    questionType,
    questionId,
    sessionId,
    attemptId,
    from,
    to,
    requestId,
    eventId,
    category,
    limit: integerParameter(
      url.searchParams,
      "limit",
      DEFAULT_REPORT_LIMIT,
      1,
      MAX_REPORT_LIMIT,
    ),
    offset,
  };
}

export function reportRpcParameters(
  query: ReportQuery,
): Record<string, string | number | null> {
  if (query.view === "feedback_report_authority_status") {
    return {};
  }
  if (query.view === "feedback_report_business_snapshot") {
    return {
      p_release_id: query.releaseId,
      p_family: query.family,
      p_question_type: query.questionType,
      p_question_id: query.questionId,
      p_from: query.from,
      p_to: query.to,
      p_limit: query.limit,
      p_session_id: query.sessionId,
      p_attempt_id: query.attemptId,
    };
  }
  if (query.view === "feedback_report_ingestion_summary") {
    return {
      p_from: query.from,
      p_to: query.to,
      p_request_id: query.requestId,
    };
  }
  if (query.view === "feedback_report_registry_quality") {
    return {
      p_from: query.from,
      p_to: query.to,
    };
  }
  if (query.view === "feedback_report_event_resolution") {
    return { p_event_id: query.eventId };
  }
  const parameters: Record<string, string | number | null> = {
    p_release_id: query.releaseId,
    p_family: query.family,
    p_question_type: query.questionType,
    p_question_id: query.questionId,
    p_from: query.from,
    p_to: query.to,
    p_session_id: query.sessionId,
    p_attempt_id: query.attemptId,
  };
  if (query.view === "feedback_report_comments") {
    parameters.p_category = query.category;
  }
  return parameters;
}
