/**
 * Session recorder: captures pointer trajectory, clicks, scrolls,
 * tab-visibility switches and quiz semantic events into one compact,
 * replayable timeline. Coordinates are stored as permille (0-1000) of the
 * viewport so a recording replays correctly on any screen size.
 *
 * Event wire format (arrays keep the payload small):
 *   [dt, "m", x, y]            pointer move (throttled ~60ms, >=2px delta)
 *   [dt, "c", x, y, button]    pointer down (0=left 1=middle 2=right)
 *   [dt, "s", scrollY]         window scroll (throttled ~150ms), px
 *   [dt, "v", 0|1]             tab hidden (0) / visible again (1)
 *   [dt, "f", 0|1]             window blurred (0) / focused (1)
 *   [dt, "o", 0|1]             pointer left (0) / re-entered (1) the page
 *   [dt, "q", questionId]      question view
 *   [dt, "g", stage]           stage change (observe|compare|reveal)
 *   [dt, "a", letter, 0|1]     answer submitted (1 = correct)
 *   [dt, "r", w, h]            viewport resize, px
 * dt = milliseconds since recording start.
 */

export type RecScalar = number | string;
export type RecEvent = [number, ...RecScalar[]];

export type RecordingMeta = {
  session_id: string;
  username?: string;
  pack?: string;
  started_at: string;
  user_agent: string;
  viewport: { w: number; h: number };
  app_version: string;
};

export type Recording = {
  schema_version: 1;
  meta: RecordingMeta;
  events: RecEvent[];
};

const MOVE_MIN_MS = 60;
const MOVE_MIN_PX = 2;
const SCROLL_MIN_MS = 150;
const FLUSH_INTERVAL_MS = 5_000;
const FLUSH_MAX_EVENTS = 300;

function permille(value: number, max: number): number {
  if (max <= 0) return 0;
  return Math.max(0, Math.min(1000, Math.round((value / max) * 1000)));
}

export class SessionRecorder {
  readonly sessionId: string;
  meta: RecordingMeta;
  events: RecEvent[] = [];
  /** Auth user this recorder belongs to; lets the app detect stale recorders
   * after a user switch instead of blindly nulling the current one. */
  userId: string | null = null;

  private t0 = 0;
  private seq = 0;
  private running = false;
  private buffered = 0; // events not yet handed to the uploader
  private flushTimer: number | null = null;
  private lastMoveAt = 0;
  private lastX = -1;
  private lastY = -1;
  private lastScrollAt = 0;
  private pointerIn = true;

  /** Called with (seq, chunk, final) for every batch; seq is 0-based and dense.
   * final=true means the page is unloading — the upload must use keepalive. */
  onFlush: ((seq: number, events: RecEvent[], final: boolean) => void) | null = null;

  constructor(sessionId: string) {
    this.sessionId = sessionId;
    this.meta = {
      session_id: sessionId,
      started_at: "",
      user_agent: "",
      viewport: { w: 0, h: 0 },
      app_version: "quiz-0.4"
    };
  }

  get isRunning(): boolean {
    return this.running;
  }

  get nextSeq(): number {
    return this.seq;
  }

  start(meta: Partial<RecordingMeta>): void {
    if (this.running || typeof window === "undefined") return;
    this.running = true;
    this.t0 = Date.now();
    this.meta = {
      ...this.meta,
      ...meta,
      started_at: new Date().toISOString(),
      user_agent: navigator.userAgent,
      viewport: { w: window.innerWidth, h: window.innerHeight }
    };
    window.addEventListener("pointermove", this.onPointerMove, { passive: true });
    window.addEventListener("pointerdown", this.onPointerDown, { passive: true });
    window.addEventListener("scroll", this.onScroll, { passive: true, capture: true });
    document.addEventListener("visibilitychange", this.onVisibility);
    window.addEventListener("blur", this.onBlur);
    window.addEventListener("focus", this.onFocus);
    window.addEventListener("resize", this.onResize);
    window.addEventListener("pagehide", this.onPageHide);
    document.addEventListener("mouseleave", this.onPointerLeave);
    document.addEventListener("mouseenter", this.onPointerEnter);
    this.flushTimer = window.setInterval(() => this.flush(false), FLUSH_INTERVAL_MS);
  }

  stop(): void {
    if (!this.running) return;
    this.running = false;
    window.removeEventListener("pointermove", this.onPointerMove);
    window.removeEventListener("pointerdown", this.onPointerDown);
    window.removeEventListener("scroll", this.onScroll, { capture: true });
    document.removeEventListener("visibilitychange", this.onVisibility);
    window.removeEventListener("blur", this.onBlur);
    window.removeEventListener("focus", this.onFocus);
    window.removeEventListener("resize", this.onResize);
    window.removeEventListener("pagehide", this.onPageHide);
    document.removeEventListener("mouseleave", this.onPointerLeave);
    document.removeEventListener("mouseenter", this.onPointerEnter);
    if (this.flushTimer !== null) {
      window.clearInterval(this.flushTimer);
      this.flushTimer = null;
    }
    this.flush(true);
  }

  /** Record a semantic quiz event ("q" | "g" | "a"). */
  mark(type: string, ...args: RecScalar[]): void {
    if (!this.running) return;
    this.push([this.dt(), type, ...args]);
  }

  snapshot(): Recording {
    return {
      schema_version: 1,
      meta: this.meta,
      events: this.events
    };
  }

  private dt(): number {
    return Date.now() - this.t0;
  }

  private push(event: RecEvent): void {
    this.events.push(event);
    this.buffered += 1;
    if (this.buffered >= FLUSH_MAX_EVENTS) this.flush(false);
  }

  private flushedCount = 0; // events already handed to the uploader

  private flush(final: boolean): void {
    if (!this.onFlush) return;
    const chunk = this.events.slice(this.flushedCount);
    if (!chunk.length) return;
    const seq = this.seq;
    this.seq += 1;
    this.flushedCount = this.events.length;
    this.buffered = 0;
    this.onFlush(seq, chunk, final);
  }

  private onPointerMove = (ev: PointerEvent): void => {
    const now = Date.now();
    if (now - this.lastMoveAt < MOVE_MIN_MS) return;
    const dx = ev.clientX - this.lastX;
    const dy = ev.clientY - this.lastY;
    if (this.lastX >= 0 && dx * dx + dy * dy < MOVE_MIN_PX * MOVE_MIN_PX) return;
    this.lastMoveAt = now;
    this.lastX = ev.clientX;
    this.lastY = ev.clientY;
    this.push([
      this.dt(),
      "m",
      permille(ev.clientX, window.innerWidth),
      permille(ev.clientY, window.innerHeight)
    ]);
  };

  private onPointerDown = (ev: PointerEvent): void => {
    this.push([
      this.dt(),
      "c",
      permille(ev.clientX, window.innerWidth),
      permille(ev.clientY, window.innerHeight),
      ev.button
    ]);
  };

  private onScroll = (): void => {
    const now = Date.now();
    if (now - this.lastScrollAt < SCROLL_MIN_MS) return;
    this.lastScrollAt = now;
    this.push([this.dt(), "s", Math.round(window.scrollY)]);
  };

  private onVisibility = (): void => {
    this.push([this.dt(), "v", document.hidden ? 0 : 1]);
    // The page is still alive while hidden: this is the most reliable
    // moment to ship everything pending (tab switch, app switch, close).
    if (document.hidden) this.flush(false);
  };

  private onBlur = (): void => {
    this.push([this.dt(), "f", 0]);
  };

  private onFocus = (): void => {
    this.push([this.dt(), "f", 1]);
  };

  private onPointerLeave = (): void => {
    if (!this.pointerIn) return;
    this.pointerIn = false;
    this.push([this.dt(), "o", 0]);
  };

  private onPointerEnter = (): void => {
    if (this.pointerIn) return;
    this.pointerIn = true;
    this.push([this.dt(), "o", 1]);
  };

  private onResize = (): void => {
    this.push([this.dt(), "r", window.innerWidth, window.innerHeight]);
  };

  private onPageHide = (): void => {
    this.flush(true);
  };
}

export function parseRecording(raw: unknown): Recording | null {
  if (!raw || typeof raw !== "object") return null;
  const rec = raw as Partial<Recording> & { recording?: Partial<Recording> };
  // exportSession wraps the recording under a "recording" key.
  const inner = (rec.recording?.schema_version === 1 ? rec.recording : rec) as Partial<Recording>;
  if (inner.schema_version !== 1 || !inner.meta || !Array.isArray(inner.events)) {
    return null;
  }
  return { schema_version: 1, meta: inner.meta, events: inner.events };
}
