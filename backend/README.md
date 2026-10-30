# Fraud Detector v4 — Backend

FastAPI backend for the v4 fraud-detector system (Telugu / Hindi / English).
It does two jobs:

1. **Serves the model manifest** — the extension asks `GET /v1/model/latest`
   which model + config it should download and run locally.
2. **Collects explicit user feedback** — when a user corrects a verdict in
   the extension *and has opted in*, the extension sends one anonymized
   feedback event. Events land in an inbox; a GitHub Actions pipeline later
   preprocesses them, retrains a challenger, and only promotes it if it
   beats the current model on held-out real data. Nothing retrains or
   deploys from this API itself.

## Privacy rules (enforced, not just documented)

- **Opt-in only.** Every event must carry `consent: true`; anything else is
  rejected.
- **Anonymized before sending.** The extension replaces personal values
  client-side (`[PHONE]`, `[OTP]`, `[EMAIL]`, `[UPI]`, `[URL]`, `[ACCOUNT]`).
- **Server-side backstop.** Events whose text still looks like it contains a
  raw phone number, OTP code, email, UPI id, URL, or long digit run are
  rejected with `422` and never stored.
- **No public storage.** Local mode writes to a private data directory.
  Production mode commits to a **private** GitHub repository/branch that
  only the project owner and the retraining workflow can read.
- **Tokens stay in env vars.** `GITHUB_FEEDBACK_TOKEN` and `ADMIN_API_KEY`
  are never logged, returned, or committed.

## API

| Method | Path | Description |
| --- | --- | --- |
| GET | `/health` | Liveness + active storage mode |
| GET | `/v1/model/latest` | Model manifest (version, model/config URLs, sha256) |
| POST | `/v1/feedback` | One feedback event → `202 {status: accepted\|duplicate}` |
| POST | `/v1/feedback/batch` | JSON list, `{"rows": [...]}`, raw CSV body, or CSV file upload (`file` field) |
| GET | `/v1/feedback/export.csv` | All stored events as CSV; needs `X-Admin-API-Key` when `ADMIN_API_KEY` is set |

### Feedback event

```json
{
  "event_id": "evt-2026-0001",
  "text_anonymized": "Your account will be blocked. Share the [OTP] and pay via [UPI].",
  "language": "en",
  "predicted_label": "safe",
  "predicted_probability": 0.12,
  "corrected_label": "fraud",
  "reward": -1.0,
  "model_version": "4.0.0",
  "extension_version": "2.0.0",
  "client_created_at": "2026-10-30T10:00:00+05:30",
  "consent": true
}
```

Validation: `language` ∈ en/hi/te (full names accepted, stored normalized);
labels accept fraud/safe, yes/no, 1/0 (stored as fraud/safe);
`predicted_probability` ∈ [0, 1]; `reward` ∈ [-1, 1]; duplicate `event_id`
values are idempotent (reported as `duplicate`, stored once). Per-IP rate
limiting, a request-body size cap, a per-event text cap, and a batch row
cap are all configurable (see `.env.example`).

Export CSV columns (ready for ML preprocessing):
`event_id, text_anonymized, language, predicted_label,
predicted_probability, corrected_label, reward, model_version,
extension_version, client_created_at, received_at, consent`

## Run locally

```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # edit as needed; local mode needs nothing
uvicorn api.index:app --reload --port 8000
# or: uvicorn app.main:app --reload --port 8000
```

Without GitHub env vars the API runs in local mode and appends events to
`$DATA_DIR/feedback.jsonl` (default `./data/feedback.jsonl`).

Run the tests:

```bash
python -m pytest
```

## Storage modes

`FEEDBACK_STORAGE=auto` (default): if `GITHUB_FEEDBACK_TOKEN` and
`FEEDBACK_REPO` are set, events are committed — one JSON file per event
under `feedback/inbox/<date>/<event_id>.json` — to `FEEDBACK_BRANCH`
(default `feedback-inbox`) of that **private** repo via the GitHub
Contents API. Otherwise local JSONL is used. Force a mode with
`FEEDBACK_STORAGE=local` or `=github`.

> **Vercel note:** serverless filesystems are ephemeral, so local JSONL on
> Vercel is only a smoke-test mode. For a real Vercel deployment, set the
> GitHub inbox env vars so feedback is durable.

## Deploy on Vercel

1. Push this `backend/` directory to GitHub and import it as a Vercel
   project (root directory = `backend`). `api/index.py` is the entry
   point; `vercel.json` routes all paths to it.
2. In **Project → Settings → Environment Variables**, set:

   | Variable | Value |
   | --- | --- |
   | `FEEDBACK_STORAGE` | `auto` |
   | `GITHUB_FEEDBACK_TOKEN` | Fine-grained PAT, Contents read/write on the private feedback repo only |
   | `FEEDBACK_REPO` | `owner/private-feedback-repo` |
   | `FEEDBACK_BRANCH` | `feedback-inbox` |
   | `ADMIN_API_KEY` | Long random string (protects the CSV export) |
   | `MODEL_MANIFEST_URL` | *(optional)* URL of the manifest JSON published by the CI/CD pipeline |
   | `MODEL_VERSION` / `MODEL_URL` / `MODEL_SHA256` / `MODEL_CONFIG_URL` / `MODEL_CONFIG_SHA256` | *(optional)* Used when no manifest file/URL is provided |

3. Deploy, then verify: `GET https://<your-app>.vercel.app/health` should
   return `{"status": "ok", "storage": "github", ...}`.

## How this fits the retraining loop

```
extension ──POST /v1/feedback──▶ this API ──commit──▶ private repo (feedback-inbox branch)
                                                              │
                                        GitHub Actions (scheduled / manual)
                                                              │ preprocess → dedupe → merge with dataset
                                                              │ train challenger → validate on held-out real data
                                                              ▼
                                        promote only if better → publish model + manifest
                                                              │
extension ◀──GET /v1/model/latest── this API ◀───────────────┘
```

The model files themselves (ONNX encoder + head config) are far too large
for this API — the manifest only points at their release URLs and sha256
hashes so the extension can download and verify them.
