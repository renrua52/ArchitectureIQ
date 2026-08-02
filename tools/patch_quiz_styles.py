from pathlib import Path


path = Path(__file__).resolve().parents[1] / "frontend" / "quiz" / "src" / "styles.css"
text = path.read_text(encoding="utf-8")
snippet = """
.provenance {
  display: flex;
  flex-wrap: wrap;
  gap: 0.55rem 1rem;
  margin: 0.35rem 0 0.9rem;
  color: var(--muted);
  font-size: 0.78rem;
  font-family: var(--mono);
}
"""
if ".provenance {" not in text:
    path.write_text(text.rstrip() + "\n" + snippet, encoding="utf-8")
print("patched quiz provenance styles")
