# Collection-backed quiz API

The API reads `frontend/quiz/public/data/questions.json` (override with
`QUIZ_BAKE_FILE`). GET question responses omit `reveal`; the answer endpoint
returns the reveal only after an explicit POST.

```powershell
uvicorn services.quiz_api.app:app --host 127.0.0.1 --port 8090
```
