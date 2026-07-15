const URL = import.meta.env.VITE_TELEMETRY_URL as string | undefined;
const KEY = import.meta.env.VITE_TELEMETRY_KEY as string | undefined;

export type TelemetryEvent = {
  session_id: string;
  event_type: string;
  question_id?: string;
  duration_ms?: number;
  payload?: Record<string, unknown>;
};

export function newSessionId(): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
    return crypto.randomUUID();
  }
  return `sess_${Math.random().toString(36).slice(2)}${Date.now().toString(36)}`;
}

/** Fire-and-forget; never throws to callers. */
export function track(event: TelemetryEvent): void {
  if (!URL || !KEY) {
    return;
  }
  const body = {
    ...event,
    payload: { schema_version: 1, ...(event.payload ?? {}) }
  };
  void fetch(`${URL.replace(/\/$/, "")}/api/telemetry/events`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${KEY}`
    },
    body: JSON.stringify(body),
    keepalive: true
  }).catch(() => {
    /* quiz must not depend on telemetry */
  });
}
