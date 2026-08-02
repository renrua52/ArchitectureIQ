from pathlib import Path


path = Path(__file__).resolve().parents[1] / "tools" / "export_quiz_collection.py"
text = path.read_text(encoding="utf-8")
old = '''        q = json.loads((question_dir / "question.json").read_text(encoding="utf-8"))
        qid = str(q.get("question_id", question_dir.name))'''
new = '''        q = json.loads((question_dir / "question.json").read_text(encoding="utf-8"))
        run = json.loads((question_dir.parent / "run.json").read_text(encoding="utf-8"))
        qid = str(q.get("question_id", question_dir.name))'''
if old in text:
    text = text.replace(old, new)
old = '''        provenance = {
            "profile": q.get("profile"),
            "profile_hash": q.get("profile_hash"),
            "track": record.get("track", "default"),'''
new = '''        provenance = {
            "profile": record.get("profile") or q.get("profile") or run.get("profile"),
            "profile_hash": record.get("profile_hash") or q.get("profile_hash") or run.get("profile_hash"),
            "track": record.get("track", "default"),'''
if old in text:
    text = text.replace(old, new)
path.write_text(text, encoding="utf-8")
print("patched frontend adapter provenance fallback")
