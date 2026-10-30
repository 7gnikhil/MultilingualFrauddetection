# Backend on Vercel — FastAPI notes

How the FastAPI backend (`backend/`) is wired for Vercel, using the
`vercel.json` in this component (it belongs at the **repository root**).

## Layout Vercel expects

```
<repo root>/
  vercel.json
  backend/
    requirements.txt        # fastapi, uvicorn not needed at runtime, pydantic, httpx...
    api/
      index.py              # Vercel Python entry point - must expose `app`
    app/                    # (optional) real application code
      main.py               # builds the FastAPI app
      routes/feedback.py
      ...
```

`backend/api/index.py` should be a thin entry point:

```python
from backend.app.main import app  # noqa: F401  (Vercel serves this `app`)
```

## Project settings

- **Root Directory: the repository root** (leave it as `./`). Do *not* set it
  to `backend/` — `vercel.json`, the rewrites and the function path
  `backend/api/index.py` are all written relative to the repo root.
- **Framework Preset:** Other. Vercel detects the Python function from
  `vercel.json`; there is no build command for the API.
- **Install command** (set in `vercel.json`): `pip install -r backend/requirements.txt`.

## Routing

`vercel.json` rewrites:

| Incoming path | Goes to |
| --- | --- |
| `/api/:path*` | `backend/api/index.py` (the FastAPI app) |
| `/health` | `backend/api/index.py` |

Because rewrites preserve the original path, **define FastAPI routes with
their full public paths** — e.g. `@app.get("/health")` and
`@app.post("/api/feedback")` — rather than mounting a sub-application and
stripping prefixes. One app, one path space, no surprises between local
`uvicorn` runs and production.

## Function limits to design around

- `maxDuration: 30s`, `memory: 512 MB` in `vercel.json`. The feedback
  endpoint is deliberately thin: validate → anonymise → append one CSV row
  to the private feedback repo via the GitHub API → return. Anything slow
  (preprocessing, retraining) happens in GitHub Actions, never in the
  request path.
- The backend holds **no model** and does no inference; prediction stays
  on-device in the extension. The backend is a data-collection and
  housekeeping service only.
- Vercel functions are stateless — never store feedback in local files or
  memory between requests. GitHub (private repo) is the store.

## Environment variables (set in Vercel project settings)

| Variable | Purpose |
| --- | --- |
| `FEEDBACK_GITHUB_TOKEN` | Token with Contents: write on the private feedback repo only |
| `FEEDBACK_REPO` | `owner/name` of the private feedback repo |
| `FEEDBACK_BRANCH` | Branch the backend appends to (default `main`) |
| `INGEST_API_KEY` | Shared key the extension sends as `X-Ingest-Key` on feedback posts |
| `RETRAIN_DISPATCH_TOKEN` | Token allowed to fire `repository_dispatch` on the main repo (may be the same PAT if scoped for both repos) |
| `ALLOWED_ORIGINS` | CORS allow-list; `chrome-extension://*` for the extension plus any web dashboard origin |

Locally, copy these into `backend/.env` (git-ignored) and load with
`python-dotenv`; never commit real values.

## Local parity check

```bash
pip install -r backend/requirements.txt
uvicorn backend.api.index:app --reload --port 8000
curl localhost:8000/health
```

CI runs the same import + `backend/tests` on every PR (see
`.github/workflows/ci.yml`), so a broken entry point fails before deploy.
