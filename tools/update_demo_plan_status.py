from pathlib import Path


path = Path(__file__).resolve().parents[1] / "docs" / "plans" / "demo" / "DEMO_RELEASE_INTEGRATION_PLAN.md"
text = path.read_text(encoding="utf-8")
old = "状态：已确认方向，待连续实施"
new = "状态：M1–M2 已完成；M3 GPU 受硬件限制；M4–M5 按容量形成 39 题 pilot；M6 发布候选已通过代码/静态/API 验收，待人工审计"
if old in text:
    text = text.replace(old, new, 1)
path.write_text(text, encoding="utf-8")
print("updated DEMO_RELEASE_INTEGRATION_PLAN status")
