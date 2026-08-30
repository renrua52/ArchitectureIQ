import katex from "katex";
// Bundled locally rather than pulled from a CDN so the fonts ship with the site.
import "katex/dist/katex.min.css";
import { useMemo } from "react";

/** Render one LaTeX fragment, falling back to `fallback` if KaTeX rejects it.
 *
 *  KaTeX's own error rendering paints the offending source in red inside the
 *  card, which looks like a broken page. A formula that will not typeset is
 *  still perfectly readable as source text, so that is what a failure shows.
 */
export function MathInline({ latex, fallback }: { latex: string; fallback?: string }) {
  const html = useMemo(() => {
    try {
      return katex.renderToString(latex, { throwOnError: true, displayMode: false });
    } catch {
      return null;
    }
  }, [latex]);
  if (html == null) {
    return <span className="mono">{fallback ?? latex}</span>;
  }
  // KaTeX output is generated here from our own converter, never from user input.
  return <span className="math" dangerouslySetInnerHTML={{ __html: html }} />;
}
