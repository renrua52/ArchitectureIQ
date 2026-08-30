/**
 * Exact decision geometry for the classification families.
 *
 * The baked `plot.probability` grid is a *smoothed empirical* estimate of
 * P(class 1) from 1024 rows spread over 12x12 bins, so most cells are empty and
 * the picture reads as noise even when the rule is as simple as "the sign of
 * x_a * x_b". Every one of these families, though, publishes the rule itself in
 * `dataset.params` — so when the plotted feature pair carries the whole rule we
 * can evaluate it directly and shade the true regions instead.
 *
 * `decisionField` returns null whenever the projection would *not* be faithful
 * (a rule that also depends on a coordinate the plot does not show); the caller
 * then falls back to the empirical grid rather than drawing something false.
 */

export type ArmCurve = { label: number; points: Array<[number, number]> };

export type DecisionField = {
  /** The class the rule assigns to a point of the plotted plane. */
  labelAt: (x: number, y: number) => number;
  /** Generating curves worth drawing over the shading (the spiral arms). */
  curves: ArmCurve[];
  /** Whether the origin axes are part of the boundary and worth marking. */
  originAxes: boolean;
  /** One-line caption describing the shading. */
  caption: string;
};

function num(value: unknown, fallback = 0): number {
  const parsed = typeof value === "number" ? value : Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function intList(value: unknown): number[] {
  return Array.isArray(value) ? value.map((item) => num(item, -1)) : [];
}

function numList(value: unknown): number[] {
  return Array.isArray(value) ? value.map((item) => num(item)) : [];
}

/** Reader for "the value of feature `index` at this point of the plot", or null. */
function projector(featurePair: [number, number], index: number) {
  if (index === featurePair[0]) {
    return (x: number, _y: number) => x;
  }
  if (index === featurePair[1]) {
    return (_x: number, y: number) => y;
  }
  return null;
}

function angularDistance(a: number, b: number): number {
  const raw = Math.abs((a - b) % (2 * Math.PI));
  return Math.min(raw, 2 * Math.PI - raw);
}

function spiralField(params: Record<string, unknown>): DecisionField {
  // Generation: t ~ U[0.5, 0.5 + 2*pi*turns], r = t, point = (r cos(t + phase),
  // r sin(t + phase)) with phase 0 for arm 0 and pi for arm 1. So a point sits
  // on arm k exactly when its polar angle equals its radius plus k*pi, and the
  // regions between the arms follow from whichever arm is nearer in angle.
  const turns = num(params.spiral_turns, 1);
  const tMin = 0.5;
  const tMax = 0.5 + 2 * Math.PI * turns;
  const samples = 240;
  const curves: ArmCurve[] = [0, 1].map((label) => {
    const phase = label * Math.PI;
    const points: Array<[number, number]> = [];
    for (let i = 0; i <= samples; i += 1) {
      const t = tMin + ((tMax - tMin) * i) / samples;
      points.push([t * Math.cos(t + phase), t * Math.sin(t + phase)]);
    }
    return { label, points };
  });
  return {
    labelAt: (x, y) => {
      const radius = Math.hypot(x, y);
      const angle = Math.atan2(y, x);
      return angularDistance(angle, radius) <= angularDistance(angle, radius + Math.PI) ? 0 : 1;
    },
    curves,
    originAxes: false,
    caption: "shading: the arm each point of the plane belongs to · curves: the two generating arms"
  };
}

export function decisionField(
  params: Record<string, unknown>,
  featurePair: [number, number]
): DecisionField | null {
  const ruleFamily = String(params.rule_family ?? "");
  if (ruleFamily === "spiral") {
    return spiralField(params);
  }

  const threshold = num(params.decision_threshold);
  const weights = numList(params.rule_weights);

  if (ruleFamily === "xor" || ruleFamily === "sparse_interaction") {
    const pairs = (Array.isArray(params.interaction_pairs) ? params.interaction_pairs : [])
      .map((pair) => intList(pair))
      .filter((pair) => pair.length === 2);
    if (!pairs.length || pairs.length !== weights.length) {
      return null;
    }
    const terms = pairs.map((pair, index) => {
      const left = projector(featurePair, pair[0]);
      const right = projector(featurePair, pair[1]);
      return left && right ? { weight: weights[index], left, right } : null;
    });
    // Any interaction reaching outside the plotted pair makes the projection a
    // shadow of the rule rather than the rule; the empirical grid is honest there.
    if (terms.some((term) => term == null)) {
      return null;
    }
    const exact = terms as Array<{
      weight: number;
      left: (x: number, y: number) => number;
      right: (x: number, y: number) => number;
    }>;
    return {
      labelAt: (x, y) =>
        exact.reduce((sum, term) => sum + term.weight * term.left(x, y) * term.right(x, y), 0) >
        threshold
          ? 1
          : 0,
      curves: [],
      originAxes: true,
      caption: "shading: the exact label regions of the dataset rule"
    };
  }

  const active = intList(params.active_features);

  if (ruleFamily === "piecewise_boundary" && active.length >= 2 && weights.length >= 3) {
    const primary = projector(featurePair, active[0]);
    const secondary = projector(featurePair, active[1]);
    if (!primary || !secondary || active.length > 2) {
      return null;
    }
    const [below, above, offset] = weights;
    const breakpoint = num(params.piecewise_breakpoint);
    return {
      labelAt: (x, y) => {
        const p = primary(x, y);
        const s = secondary(x, y);
        const score = (p > breakpoint ? above : below) * s + offset * p;
        return score > threshold ? 1 : 0;
      },
      curves: [],
      originAxes: false,
      caption: "shading: the exact label regions of the dataset rule"
    };
  }

  if (ruleFamily === "smooth_additive" && active.length && active.length === weights.length) {
    const terms = active.map((feature, index) => {
      const read = projector(featurePair, feature);
      return read ? { weight: weights[index], read } : null;
    });
    if (terms.some((term) => term == null)) {
      return null;
    }
    const exact = terms as Array<{ weight: number; read: (x: number, y: number) => number }>;
    return {
      labelAt: (x, y) =>
        exact.reduce((sum, term) => {
          const value = term.read(x, y);
          return sum + term.weight * (Math.sin(value) + 0.25 * value * value);
        }, 0) > threshold
          ? 1
          : 0,
      curves: [],
      originAxes: false,
      caption: "shading: the exact label regions of the dataset rule"
    };
  }

  return null;
}

export type RegionBand = { label: number; x0: number; x1: number; y0: number; y1: number };

/**
 * Sample `field` over the plotted rectangle and merge equal-label neighbours
 * along each row, so a smooth boundary costs a few dozen rectangles instead of
 * one per pixel.
 */
export function regionBands(
  field: DecisionField,
  bounds: { xMin: number; xMax: number; yMin: number; yMax: number },
  columns = 200,
  rows = 120
): RegionBand[] {
  const bands: RegionBand[] = [];
  const dx = (bounds.xMax - bounds.xMin) / columns;
  const dy = (bounds.yMax - bounds.yMin) / rows;
  if (!Number.isFinite(dx) || !Number.isFinite(dy) || dx <= 0 || dy <= 0) {
    return bands;
  }
  for (let row = 0; row < rows; row += 1) {
    const y0 = bounds.yMin + row * dy;
    const yMid = y0 + dy / 2;
    let runStart = 0;
    let runLabel = field.labelAt(bounds.xMin + dx / 2, yMid);
    for (let column = 1; column <= columns; column += 1) {
      const label =
        column === columns ? Number.NaN : field.labelAt(bounds.xMin + (column + 0.5) * dx, yMid);
      if (label !== runLabel) {
        bands.push({
          label: runLabel,
          x0: bounds.xMin + runStart * dx,
          x1: bounds.xMin + column * dx,
          y0,
          y1: y0 + dy
        });
        runStart = column;
        runLabel = label;
      }
    }
  }
  return bands;
}
