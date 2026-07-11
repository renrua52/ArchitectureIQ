# ArchitectureIQ Vanilla React Frontend

This frontend replaces the Streamlit UI with a Vite + React app backed by the FastAPI service in `backend/app.py`.

From the repository root, start the API:

```bash
python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

From this directory, start the frontend:

```bash
npm install
npm run dev
```

Open:

```text
http://localhost:5173/
```

The Vite dev server proxies `/api/*` to `http://127.0.0.1:8000`.

For deployment, run this from the repository root before building:

```bash
python -m backend.export_static_data
```

Then:

```bash
npm run build
```

The build writes `dist/server/index.js`, which serves the same `/api/*` contract from baked JSON for production hosting.
