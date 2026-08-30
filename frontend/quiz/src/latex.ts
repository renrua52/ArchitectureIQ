/** LaTeX for the two kinds of formula a question states: sampled regression
 *  targets and classification label rules.
 *
 *  The BakeFile carries formulas the way the generator writes them -- infix
 *  source like `x1**3 + cos(2*pi*x3)*sin(2*pi*x0) + 5` for a target, and the
 *  rule parameters themselves for a classifier. Both become LaTeX here, by
 *  parsing rather than by string substitution: a regex pass gets precedence and
 *  unary minus wrong, and a formula the reader cannot trust is worse than the
 *  plain source text -- so a parse failure falls back to that source.
 *
 *  The grammar mirrors `architecture_iq.families.symbolic_expr.render_infix`:
 *  variables `x` / `x<i>`, the constant `pi`, calls `sin` `cos` `tanh` `abs`,
 *  integer `**` powers, and `+ - * /`.
 */

type Node =
  | { t: "num"; text: string }
  | { t: "var"; index: number | null }
  | { t: "pi" }
  | { t: "call"; name: string; arg: Node }
  | { t: "pow"; base: Node; exponent: string }
  | { t: "neg"; arg: Node }
  | { t: "bin"; op: "+" | "-" | "*" | "/"; left: Node; right: Node };

type Token = { kind: "num" | "ident" | "op" | "(" | ")"; text: string };

const CALL_LATEX: Record<string, (arg: string) => string> = {
  sin: (arg) => `\\sin\\!\\left(${arg}\\right)`,
  cos: (arg) => `\\cos\\!\\left(${arg}\\right)`,
  tanh: (arg) => `\\tanh\\!\\left(${arg}\\right)`,
  exp: (arg) => `\\exp\\!\\left(${arg}\\right)`,
  log: (arg) => `\\log\\!\\left(${arg}\\right)`,
  abs: (arg) => `\\left|${arg}\\right|`,
  sqrt: (arg) => `\\sqrt{${arg}}`
};

function tokenize(source: string): Token[] {
  const tokens: Token[] = [];
  let i = 0;
  while (i < source.length) {
    const ch = source[i];
    if (ch === " " || ch === "\t" || ch === "\n") {
      i += 1;
      continue;
    }
    if (ch === "(" || ch === ")") {
      tokens.push({ kind: ch, text: ch });
      i += 1;
      continue;
    }
    if (source.startsWith("**", i)) {
      tokens.push({ kind: "op", text: "**" });
      i += 2;
      continue;
    }
    if (ch && "+-*/".includes(ch)) {
      tokens.push({ kind: "op", text: ch });
      i += 1;
      continue;
    }
    const rest = source.slice(i);
    const number = /^\d+(?:\.\d*)?(?:[eE][+-]?\d+)?/.exec(rest);
    if (number) {
      tokens.push({ kind: "num", text: number[0] });
      i += number[0].length;
      continue;
    }
    const ident = /^[A-Za-z_][A-Za-z_0-9]*/.exec(rest);
    if (ident) {
      tokens.push({ kind: "ident", text: ident[0] });
      i += ident[0].length;
      continue;
    }
    throw new Error(`unexpected character "${ch}" at offset ${i}`);
  }
  return tokens;
}

class Parser {
  private position = 0;

  constructor(private readonly tokens: Token[]) {}

  parse(): Node {
    const node = this.parseSum();
    if (this.position !== this.tokens.length) {
      throw new Error(`trailing input at token ${this.position}`);
    }
    return node;
  }

  private peek(): Token | undefined {
    return this.tokens[this.position];
  }

  private take(): Token {
    const token = this.tokens[this.position];
    if (!token) {
      throw new Error("unexpected end of expression");
    }
    this.position += 1;
    return token;
  }

  private expect(kind: Token["kind"], text?: string): Token {
    const token = this.take();
    if (token.kind !== kind || (text != null && token.text !== text)) {
      throw new Error(`expected ${text ?? kind}, got "${token.text}"`);
    }
    return token;
  }

  private parseSum(): Node {
    let left = this.parseProduct();
    for (;;) {
      const token = this.peek();
      if (!token || token.kind !== "op" || (token.text !== "+" && token.text !== "-")) {
        return left;
      }
      this.take();
      left = { t: "bin", op: token.text as "+" | "-", left, right: this.parseProduct() };
    }
  }

  private parseProduct(): Node {
    let left = this.parseUnary();
    for (;;) {
      const token = this.peek();
      if (!token || token.kind !== "op" || (token.text !== "*" && token.text !== "/")) {
        return left;
      }
      this.take();
      left = { t: "bin", op: token.text as "*" | "/", left, right: this.parseUnary() };
    }
  }

  private parseUnary(): Node {
    const token = this.peek();
    if (token && token.kind === "op" && (token.text === "-" || token.text === "+")) {
      this.take();
      const arg = this.parseUnary();
      return token.text === "-" ? { t: "neg", arg } : arg;
    }
    return this.parsePower();
  }

  private parsePower(): Node {
    const base = this.parseAtom();
    const token = this.peek();
    if (token && token.kind === "op" && token.text === "**") {
      this.take();
      // Only integer exponents are sampled, and `2**-1` never appears.
      const exponent = this.expect("num");
      return { t: "pow", base, exponent: exponent.text };
    }
    return base;
  }

  private parseAtom(): Node {
    const token = this.take();
    if (token.kind === "num") {
      return { t: "num", text: token.text };
    }
    if (token.kind === "(") {
      const inner = this.parseSum();
      this.expect(")");
      return inner;
    }
    if (token.kind === "ident") {
      if (token.text === "pi") {
        return { t: "pi" };
      }
      if (CALL_LATEX[token.text]) {
        this.expect("(");
        const arg = this.parseSum();
        this.expect(")");
        return { t: "call", name: token.text, arg };
      }
      const variable = /^x(\d*)$/.exec(token.text);
      if (variable) {
        return { t: "var", index: variable[1] ? Number(variable[1]) : null };
      }
      throw new Error(`unknown identifier "${token.text}"`);
    }
    throw new Error(`unexpected token "${token.text}"`);
  }
}

/** Binding strength, so the emitter parenthesizes exactly where it must. */
function precedence(node: Node): number {
  if (node.t === "bin") {
    return node.op === "+" || node.op === "-" ? 1 : 2;
  }
  if (node.t === "neg") {
    return 1;
  }
  return 4;
}

function wrap(latex: string, node: Node, minimum: number): string {
  return precedence(node) < minimum ? `\\left(${latex}\\right)` : latex;
}

function emit(node: Node, minimum = 0): string {
  switch (node.t) {
    case "num":
      return node.text;
    case "pi":
      return "\\pi";
    case "var":
      return node.index == null ? "x" : `x_{${node.index}}`;
    case "call":
      return CALL_LATEX[node.name]!(emit(node.arg));
    case "pow": {
      // A product, sum or negation under a power needs its own parentheses;
      // `x_{0}` and `\cos(...)` do not.
      const base = precedence(node.base) < 4 ? `\\left(${emit(node.base)}\\right)` : emit(node.base);
      return wrap(`${base}^{${node.exponent}}`, node, minimum);
    }
    case "neg":
      return wrap(`-${emit(node.arg, 2)}`, node, minimum);
    case "bin": {
      if (node.op === "/") {
        return `\\frac{${emit(node.left)}}{${emit(node.right)}}`;
      }
      if (node.op === "*") {
        // Multiplication is left-associative and shares its level with
        // division, so neither side needs parentheses at level 2: `2*pi*x`
        // stays `2\pi x` rather than becoming `\left(2\pi\right)x`.
        const right = emit(node.right, 2);
        const left = emit(node.left, 2);
        // Juxtaposition reads as multiplication, with two exceptions: in front
        // of a digit `2 \cdot 3` keeps it from becoming the number 23, and
        // after a control word a thin space keeps `2\pi x` from parsing as the
        // undefined command `\pix`.
        const joint = /^[\d.]/.test(right)
          ? " \\cdot "
          : /\\[A-Za-z]+$/.test(left)
            ? "\\,"
            : "";
        return wrap(`${left}${joint}${right}`, node, minimum);
      }
      const operator = node.op === "+" ? " + " : " - ";
      return wrap(`${emit(node.left, 1)}${operator}${emit(node.right, 2)}`, node, minimum);
    }
  }
}

/** LaTeX for one sampled target expression, or null when it does not parse. */
export function expressionToLatex(source: string): string | null {
  try {
    return emit(new Parser(tokenize(source)).parse());
  } catch {
    return null;
  }
}

/** One display line of a multi-line rule: optional lead-in text plus math. */
export type MathLine = { prefix?: string; latex: string };

/** `%.6g`-ish: enough digits for the sampled grids, no trailing zero noise. */
function num(value: number): string {
  if (!Number.isFinite(value)) return String(value);
  const rendered = Number(value.toPrecision(6)).toString();
  return rendered === "-0" ? "0" : rendered;
}

function coefficient(weight: number, term: string): string {
  if (weight === 1) return term;
  if (weight === -1) return `-${term}`;
  return `${num(weight)}${term}`;
}

/** `+ 3x_1` / `- 3x_1`, with the sign folded into the first term. */
function signedSum(terms: { weight: number; term: string }[]): string {
  return terms
    .map(({ weight, term }, index) => {
      if (index === 0) return coefficient(weight, term);
      const magnitude = coefficient(Math.abs(weight), term);
      return `${weight < 0 ? " - " : " + "}${magnitude}`;
    })
    .join("");
}

function variable(index: unknown): string {
  return `x_{${String(index)}}`;
}

function numbers(value: unknown): number[] {
  return Array.isArray(value) ? value.map(Number) : [];
}

/** LaTeX for a classification family's label rule, or null when unavailable.
 *
 *  Mirrors `format_synthetic_tabular_classification_rule` in the prompt
 *  formatters: same score, same threshold, same branch order -- the reader must
 *  be able to check the card against the prompt it accompanies. A shape this
 *  does not recognize returns null and the card simply omits the row.
 */
export function classificationRuleLatex(family: string, params: Record<string, unknown>): MathLine[] | null {
  const ruleFamily = String(params.rule_family ?? "");
  const active = numbers(params.active_features);
  const weights = numbers(params.rule_weights);
  const threshold = Number(params.decision_threshold);

  if (family === "spiral_classification" || ruleFamily === "spiral") {
    const turns = Number(params.spiral_turns);
    if (!Number.isFinite(turns)) return null;
    const span = turns === 1 ? "2\\pi" : `${num(turns)}\\cdot 2\\pi`;
    return [
      { prefix: "arm", latex: `t \\sim \\mathrm{Uniform}\\left[0.5,\\ 0.5 + ${span}\\right],\\quad r = t` },
      { prefix: "point", latex: `(x_{0},\\, x_{1}) = \\left(r\\cos(t + \\varphi),\\ r\\sin(t + \\varphi)\\right),\\quad \\varphi \\in \\{0,\\, \\pi\\}` },
      { prefix: "label", latex: `y = 0 \\text{ for } \\varphi = 0,\\quad y = 1 \\text{ for } \\varphi = \\pi` }
    ];
  }

  if (!Number.isFinite(threshold)) return null;
  const decision = { prefix: "label", latex: `y = 1 \\iff s(x) > ${num(threshold)}` };

  if (ruleFamily === "xor" || ruleFamily === "sparse_interaction") {
    const pairs = Array.isArray(params.interaction_pairs)
      ? (params.interaction_pairs as unknown[]).map((pair) => numbers(pair))
      : [];
    if (!pairs.length || pairs.length !== weights.length) return null;
    const terms = pairs.map((pair, index) => ({
      weight: weights[index]!,
      term: `${variable(pair[0])}${variable(pair[1])}`
    }));
    return [{ latex: `s(x) = ${signedSum(terms)}` }, decision];
  }

  if (ruleFamily === "smooth_additive") {
    if (!active.length || active.length !== weights.length) return null;
    const terms = active.map((feature, index) => ({
      weight: weights[index]!,
      term: `\\left[\\sin(${variable(feature)}) + 0.25\\,${variable(feature)}^{2}\\right]`
    }));
    return [{ latex: `s(x) = ${signedSum(terms)}` }, decision];
  }

  if (ruleFamily === "piecewise_boundary") {
    const breakpoint = Number(params.piecewise_breakpoint);
    if (active.length < 2 || weights.length < 3 || !Number.isFinite(breakpoint)) return null;
    const [primary, secondary] = active;
    const [below, above, offset] = weights as [number, number, number];
    const branch = (slope: number) =>
      signedSum([
        { weight: slope, term: variable(secondary) },
        { weight: offset, term: variable(primary) }
      ]);
    return [
      {
        latex:
          `s(x) = \\begin{cases}` +
          `${branch(above)} & ${variable(primary)} > ${num(breakpoint)} \\\\[2pt]` +
          `${branch(below)} & \\text{otherwise}` +
          `\\end{cases}`
      },
      decision
    ];
  }

  return null;
}
