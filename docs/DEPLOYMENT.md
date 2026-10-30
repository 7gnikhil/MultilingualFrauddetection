# Deployment Guide — Fraud Detector v4

Everything in this system is deployed from the GitHub repo
`7gnikhil/MultilingualFrauddetection`. There are four moving parts:

```
┌────────────────┐   corrections (CSV rows)   ┌─────────────────────┐
│ Chrome/Edge    │ ──────── POST /api ──────▶ │ Backend (FastAPI)   │
│ extension      │                             │ on Vercel           │
│ (on-device     │ ◀──── model config URL ──── │ - validate/anonymise│
│  ONNX model)   │        (GitHub Release)     │ - append to GitHub  │
└────────────────┘                             └──────────┬──────────┘
                                                          │ commits CSV
                                                          ▼
                                               ┌─────────────────────┐
                                               │ PRIVATE feedback    │
                                               │ repo (CSV exports)  │
                                               └──────────┬──────────┘
                                  repository_dispatch / weekly schedule
                                                          ▼
                                               ┌─────────────────────┐
│ GitHub Releases                              │ GitHub Actions      │
│ - extension zips      ◀── release-extension  │ retrain.yml: merge  │
│ - model assets        ◀── promotion merge    │ feedback, train     │
│   (model.onnx, config)                       │ challenger, gates,  │
└────────────────┘                             │ open PR (no merge)  │
                                               └─────────────────────┘
```

## 1. Backend → Vercel

1. On [vercel.com](https://vercel.com), **Add New → Project** and import
   `7gnikhil/MultilingualFrauddetection`.
2. **Root Directory: `./`** (repository root — `vercel.json` lives there).
   Framework Preset: **Other**.
3. Add the environment variables from
   [BACKEND_VERCEL.md](BACKEND_VERCEL.md#environment-variables-set-in-vercel-project-settings).
4. Deploy. Verify: `https://<project>.vercel.app/health` returns the
   backend's health JSON.
5. Every push to `main` redeploys automatically; PRs get preview URLs.

Routing, entry-point layout and function limits are documented in
[BACKEND_VERCEL.md](BACKEND_VERCEL.md).

## 2. Model artifacts → GitHub Releases

The ONNX encoder is ~239 MB — far over GitHub's 100 MB per-file git limit —
so model files are **Release assets, never git commits**.

- Release tag convention: `model-v<version>` (e.g. `model-v1.1.0`).
- Assets: `model.onnx`, `config.json` (heads, scalers, per-language
  thresholds, model-config version), `SHA256SUMS.txt`, `evaluation.json`.
- A model Release is created when a promotion PR merges (see
  [MODEL_PROMOTION_POLICY.md](MODEL_PROMOTION_POLICY.md)). The previous
  releases are kept — they are the rollback path.
- The extension checks the latest model Release for a newer `config.json`,
  verifies its SHA-256, and silently keeps its bundled config on any
  failure. The encoder binary itself ships inside the extension package;
  config-only updates (new heads/thresholds) do not need a store update.

## 3. Extension → GitHub Releases

Pushing a tag like `v1.2.0` runs `.github/workflows/release-extension.yml`,
which validates the manifest, fetches `model.onnx` from the latest model
Release, builds the zips with `scripts/package_extension.py`, and publishes
a GitHub Release with the zips + `SHA256SUMS.txt`.

Users install by unzipping and using **Load unpacked** (Chrome/Edge →
Extensions → Developer mode). This is the free distribution channel; the
Edge store (next section) is the one-click channel.

## 4. Microsoft Edge Add-ons (manual submission)

The release workflow **prepares** the store package
(`fraud-detector-edge-v<version>.zip`, manifest validated with
`scripts/validate_manifest.py --edge`) but cannot submit it: submission
requires the publisher's own Microsoft Partner Center account and passes
Microsoft's certification review.

Manual steps (per release):

1. Sign in to [Microsoft Partner Center](https://partner.microsoft.com/dashboard/microsofthardware)
   → **Microsoft Edge program** (one-time developer registration is free).
2. Open the **Fraud Detector** product (create it on first submission).
3. **Create new submission** → **Packages**: upload
   `fraud-detector-edge-v<version>.zip` from the GitHub Release.
4. **Availability**: public, all markets (or as intended).
5. **Properties**: category (Productivity / Security), privacy policy URL
   (required — the policy must state that corrections the user explicitly
   flags are sent to the backend; nothing else leaves the device).
6. **Store listings**: name, summary, description, logo/screenshots
   (reuse the GitHub repo assets).
7. **Notes for certification**: explain the permissions (floating button on
   pages, offscreen document for the on-device model, storage for learned
   corrections) and that inference is 100% on-device.
8. **Publish** and wait for certification (typically a few days). If it
   fails, the reviewer notes land in Partner Center + email — fix and
   resubmit a new package; do not reuse the failed submission's package.
9. Optional future automation: Partner Center offers a submissions API;
   with `EDGE_PRODUCT_ID`, `EDGE_API_CLIENT_ID`, `EDGE_API_CLIENT_SECRET`
   secrets, a publish step can be added to `release-extension.yml`. Review
   itself can never be automated away.

Chrome Web Store is the analogous channel for Chrome users (one-time
US$5 developer fee, same zip, submission via the Chrome Web Store
dashboard). It is optional and not wired into the workflows.

## 5. The feedback → retrain loop

1. In the extension, the user flags a wrong verdict ("It's fraud" /
   "It's safe"). Only flagged messages are sent — never unflagged ones.
2. The extension POSTs the correction to the backend
   (`POST /api/feedback`, header `X-Ingest-Key: <INGEST_API_KEY>`).
3. The backend validates, anonymises (OTP/phone/account → placeholders)
   and appends the row to a CSV in the **private feedback repo** via the
   GitHub API.
4. When the pending-correction count crosses the threshold (default 50),
   the backend fires a `repository_dispatch` event
   (`feedback-threshold-reached`) at the main repo. A weekly scheduled run
   of `retrain.yml` acts as a safety net.
5. `retrain.yml` merges feedback, trains a challenger, evaluates the
   promotion gates, and — only if they pass — opens a PR.
6. A human reviews and merges (or closes) the PR. Merging publishes the
   new model Release; extensions pick up the new config automatically.

## 6. Required secrets & variables

### GitHub → Settings → Secrets and variables → Actions

| Name | Kind | Purpose |
| --- | --- | --- |
| `FEEDBACK_REPO_TOKEN` | Secret | Fine-grained PAT: Contents **read** on the private feedback repo only (used by `retrain.yml` to check it out) |
| `FEEDBACK_REPO` | Variable | `owner/name` of the private feedback repo, e.g. `7gnikhil/fraud-feedback-private` |
| `GITHUB_TOKEN` | built-in | Used automatically for PR creation, releases and artifact download — no setup needed. Repo Settings → Actions → Workflow permissions must allow **read and write** + **PR creation** |

### Vercel → Project → Settings → Environment Variables

`FEEDBACK_GITHUB_TOKEN` (Contents: **write** on the feedback repo),
`FEEDBACK_REPO`, `FEEDBACK_BRANCH`, `INGEST_API_KEY`,
`RETRAIN_DISPATCH_TOKEN`, `ALLOWED_ORIGINS` — details in
[BACKEND_VERCEL.md](BACKEND_VERCEL.md).

### Never commit

Real tokens, the ingest key, or raw (un-anonymised) feedback. Raw user
messages exist only transiently in the backend request; what lands in the
feedback repo is already anonymised.

## 7. CI/CD integration contract

The workflows assume the following from the `ml/`, `backend/` and
`chrome-extension/` components. If a component changes one of these, the
workflows must change in the same PR.

| Contract | Expected by |
| --- | --- |
| `python -m ml.data.build_dataset --sheet-export … --out …` | retrain.yml |
| `python -m ml.data.merge_feedback --feedback-dir … --base … --out …` | retrain.yml |
| `python -m ml.train --config ml/configs/challenger.yaml --dataset … --out artifacts/challenger` (writes `config.json`) | retrain.yml |
| `python -m ml.train --smoke --config ml/configs/smoke.yaml` (fixture embeddings, no downloads) | ci.yml |
| `python -m ml.evaluate --champion models/champion/config.json --challenger <dir> --gates ml/configs/promotion_gates.yaml --out <report.json>`, exit 0 ⇔ gates pass | retrain.yml |
| `models/champion/config.json` — the current champion, in git | retrain.yml |
| `backend/api/index.py` exposing `app`; `backend/requirements.txt`; `backend/tests/` | ci.yml, Vercel |
| `chrome-extension/manifest.json` (MV3) + JS at `chrome-extension/` root | ci.yml, release-extension.yml |
| `scripts/validate_manifest.py`, `scripts/package_extension.py` (this component) | ci.yml, release-extension.yml |

## 8. Rollback quick reference

| Layer | Rollback |
| --- | --- |
| Model | Revert the promotion PR / re-publish the previous model Release (see MODEL_PROMOTION_POLICY.md) |
| Backend | Vercel dashboard → Deployments → promote the previous deployment |
| Extension | Re-run `release-extension.yml` on the previous tag, or re-attach the previous zips to a new Release; store versions are rolled forward with a fixed package, never back |
| Feedback loop | Disable the backend's dispatch (clear `RETRAIN_DISPATCH_TOKEN`) or disable the `retrain.yml` workflow in the Actions tab |
