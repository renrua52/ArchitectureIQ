// Protected, read-only ArchitectureIQ feedback reporting endpoint.
//
// Deploy with Supabase JWT verification disabled. This function authenticates
// a dedicated FEEDBACK_REPORT_TOKEN and keeps the service-role credential only
// in its hosted environment. The ingestion token is deliberately not accepted.

import {
  parseReportQuery,
  ReportQueryError,
  reportRpcParameters,
} from "./report_query.ts";
import type { ReportQuery } from "./report_query.ts";

type JsonObject = Record<string, unknown>;

interface ReportResult {
  rows: JsonObject[];
  total: number;
  // Business and surprise reports contain PostgreSQL bigint counts. Keep the
  // validated PostgREST array bytes so those integers never round-trip through
  // an IEEE-754 JavaScript Number before reaching the strict client.
  rawRowsJson?: string;
}

const REPORT_ORDER = {
  feedback_report_sessions:
    "last_event_at.desc,session_id.asc,attempt_id.asc.nullsfirst",
  feedback_report_questions:
    "answer_count.desc,question_id.asc,question_version.asc," +
    "release_id.asc.nullsfirst,family.asc.nullsfirst," +
    "dataset_id.asc.nullsfirst,question_type.asc.nullsfirst",
  feedback_report_answers: "occurred_at.desc,event_id.asc",
  feedback_report_proposals: "occurred_at.desc,event_id.asc",
  feedback_report_comments: "occurred_at.desc,event_id.asc",
  feedback_report_surprise_questions:
    "posterior_mean.desc,rating_count.desc,release_id.asc," +
    "question_id.asc,question_version.asc",
} as const;
const SINGLE_ROW_REPORT_VIEWS = new Set([
  "feedback_report_summary",
  "feedback_report_ingestion_summary",
  "feedback_report_authority_status",
  "feedback_report_business_snapshot",
  "feedback_report_registry_quality",
  "feedback_report_surprise_quality",
  "feedback_report_event_resolution",
]);

function isObject(value: unknown): value is JsonObject {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function responseHeaders(requestId: string): Headers {
  return new Headers({
    "Cache-Control": "no-store",
    "Content-Type": "application/json; charset=utf-8",
    Vary: "Authorization",
    "X-Content-Type-Options": "nosniff",
    "X-Request-ID": requestId,
  });
}

function jsonResponse(
  status: number,
  requestId: string,
  body: JsonObject,
): Response {
  return new Response(JSON.stringify({ ...body, request_id: requestId }), {
    status,
    headers: responseHeaders(requestId),
  });
}

function errorResponse(
  status: number,
  requestId: string,
  code: string,
  message: string,
): Response {
  return jsonResponse(status, requestId, {
    error: { code, message },
  });
}

function reportResponse(
  requestId: string,
  query: ReportQuery,
  report: ReportResult,
): Response {
  const rowsJson = report.rawRowsJson ?? JSON.stringify(report.rows);
  const body =
    `{"view":${JSON.stringify(query.view)},"rows":${rowsJson},` +
    `"total":${report.total},"limit":${query.limit},` +
    `"offset":${query.offset},"request_id":${JSON.stringify(requestId)}}`;
  return new Response(body, {
    status: 200,
    headers: responseHeaders(requestId),
  });
}

async function tokenMatches(
  received: string,
  expected: string,
): Promise<boolean> {
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

async function authorized(
  request: Request,
  expectedToken: string,
): Promise<boolean> {
  const authorization = request.headers.get("Authorization") ?? "";
  const match = /^Bearer ([^\s]+)$/i.exec(authorization);
  return match !== null && (await tokenMatches(match[1], expectedToken));
}

function exactContentRangeTotal(response: Response): number {
  const contentRange = response.headers.get("Content-Range") ?? "";
  const match = /^(?:\d+-\d+|\*)\/(\d+)$/.exec(contentRange);
  if (match === null) {
    throw new Error("report response did not include an exact total");
  }
  const total = Number(match[1]);
  if (!Number.isSafeInteger(total) || total < 0) {
    throw new Error("report response total was invalid");
  }
  return total;
}

async function queryReport(
  query: ReportQuery,
  requestId: string,
  supabaseUrl: string,
  serviceRoleKey: string,
): Promise<ReportResult> {
  let endpoint: URL;
  try {
    endpoint = new URL(`/rest/v1/rpc/${query.view}`, supabaseUrl);
  } catch {
    throw new Error("report storage configuration is invalid");
  }
  const paginated = !SINGLE_ROW_REPORT_VIEWS.has(query.view);
  if (paginated) {
    endpoint.searchParams.set("limit", String(query.limit));
    endpoint.searchParams.set("offset", String(query.offset));
    endpoint.searchParams.set(
      "order",
      REPORT_ORDER[query.view as keyof typeof REPORT_ORDER],
    );
  }

  const response = await fetch(endpoint, {
    method: "POST",
    headers: {
      apikey: serviceRoleKey,
      Authorization: `Bearer ${serviceRoleKey}`,
      "Content-Type": "application/json",
      ...(paginated ? { Prefer: "count=exact" } : {}),
    },
    body: JSON.stringify(reportRpcParameters(query)),
  });
  if (paginated && response.status === 416) {
    // PostgREST may represent a valid page beyond the end as 416 with an exact
    // `Content-Range: */N`. Treat that as an empty page without reading or
    // exposing its diagnostic body.
    return { rows: [], total: exactContentRangeTotal(response) };
  }
  if (!response.ok) {
    // Never log or return the PostgREST response body. It can contain database
    // diagnostics; the request id and status are sufficient for correlation.
    console.error(
      JSON.stringify({
        request_id: requestId,
        code: "postgrest_report_failed",
        view: query.view,
        status: response.status,
      }),
    );
    throw new Error("report query failed");
  }

  let value: unknown;
  let rawRowsJson: string | undefined;
  try {
    if (
      query.view === "feedback_report_business_snapshot" ||
      query.view === "feedback_report_surprise_questions" ||
      query.view === "feedback_report_surprise_quality"
    ) {
      rawRowsJson = await response.text();
      value = JSON.parse(rawRowsJson);
    } else {
      value = await response.json();
    }
  } catch {
    throw new Error("report response was invalid");
  }
  if (!Array.isArray(value) || !value.every(isObject)) {
    throw new Error("report response was invalid");
  }
  const rows = value as JsonObject[];
  if (!paginated) {
    if (rows.length !== 1) {
      throw new Error(
        "single-row report response must contain exactly one row",
      );
    }
    return { rows, total: 1, rawRowsJson };
  }
  return { rows, total: exactContentRangeTotal(response), rawRowsJson };
}

Deno.serve(async (request: Request): Promise<Response> => {
  const requestId = crypto.randomUUID();

  if (request.method !== "GET") {
    const response = errorResponse(
      405,
      requestId,
      "METHOD_NOT_ALLOWED",
      "Only GET is supported",
    );
    response.headers.set("Allow", "GET");
    return response;
  }

  const reportToken = Deno.env.get("FEEDBACK_REPORT_TOKEN")?.trim() ?? "";
  const supabaseUrl = Deno.env.get("SUPABASE_URL")?.trim() ?? "";
  const serviceRoleKey =
    Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")?.trim() ?? "";
  if (!reportToken || !supabaseUrl || !serviceRoleKey) {
    console.error(
      JSON.stringify({
        request_id: requestId,
        code: "missing_report_configuration",
      }),
    );
    return errorResponse(
      503,
      requestId,
      "SERVICE_UNAVAILABLE",
      "Feedback reporting is temporarily unavailable",
    );
  }

  if (!(await authorized(request, reportToken))) {
    const response = errorResponse(
      401,
      requestId,
      "UNAUTHORIZED",
      "A valid report bearer token is required",
    );
    response.headers.set("WWW-Authenticate", "Bearer");
    return response;
  }

  let query: ReportQuery;
  try {
    query = parseReportQuery(request.url);
  } catch (error) {
    return errorResponse(
      400,
      requestId,
      "INVALID_QUERY",
      error instanceof ReportQueryError ? error.message : "Query is invalid",
    );
  }

  try {
    const report = await queryReport(
      query,
      requestId,
      supabaseUrl,
      serviceRoleKey,
    );
    return reportResponse(requestId, query, report);
  } catch (error) {
    console.error(
      JSON.stringify({
        request_id: requestId,
        code: "report_unavailable",
        view: query.view,
        error_type: error instanceof Error ? error.name : "unknown",
      }),
    );
    return errorResponse(
      502,
      requestId,
      "REPORT_UNAVAILABLE",
      "Feedback report could not be loaded",
    );
  }
});
