# 🛡️ Fraud Detector — Chrome Extension v1.2.0

Checks a message for fraud in **Telugu, Hindi (including Romanized /
Hinglish) and English** — right on the page where you read it: WhatsApp
Web, Gmail, Truecaller web, anywhere.

> **Component note:** this v4 extension component intentionally does
> **not** contain `model/model.onnx`. Before loading it in Chrome, copy
> the 239 MB `model.onnx` from the v3 extension package into `model/`.
> Every other model file, including `model/config.json` and
> `model/vocab.json`, is present here.

---

## Privacy model

**Inference is local.** The message, MuRIL encoder, language heads, and
personal correction adapter all run inside the extension. No message is
sent to a server merely to obtain a verdict.

**Feedback sharing is opt-in and OFF by default.** An anonymized
correction can leave the device only when both of these are true:

1. The user enables **“Share anonymized corrections to improve the
   model”** in the toolbar popup; and
2. The user then clicks **“It's fraud”** or **“It's safe”** under a
   specific verdict.

Before a correction is persisted in the upload queue, the background
service worker runs the same `anonymize.js` used by the model pipeline.
It replaces URLs, email addresses, UPI IDs, account/card-like numbers,
phone numbers, and OTP-like digit runs with placeholders such as
`[URL]`, `[EMAIL]`, `[UPI]`, `[ACCOUNT]`, `[PHONE]`, and `[OTP]`.

The persistent local corrections log also stores anonymized text now.
Turning sharing off prevents future sync attempts. Resetting learning
clears the local correction log and discards any still-unsent queued
feedback events. Corrections made before opt-in are **not** retroactively
queued.

Anonymized text can still be sensitive in context. Treat feedback data
as private operational data on the backend: restrict access, encrypt it
in transit and at rest, and set a retention policy.

---

## Using the extension

1. **It stays on screen.** After install, a round 🛡️ button appears at
   the bottom-right of every page and stays visible. On first run the
   panel opens once to confirm keeping it on screen.
2. **Drag the button anywhere** — it remembers its position on that site.
3. **Click the button** to open the check panel.
4. **Drag a message in**, select text on the page and use the
   **“🛡️ Check if fraud”** bubble, right-click selected text and choose
   **“Check with Fraud Detector,”** or paste/type the message manually.

### Verdict states

- **🚨 FRAUD** — the model probability is above the language threshold
  by more than that language's uncertainty margin.
- **✅ LOOKS SAFE** — the probability is below the threshold by more
  than the uncertainty margin.
- **❓ UNCERTAIN — NEEDS REVIEW** — the probability is within the
  uncertainty margin of the threshold. The extension does not dress this
  up as a confident SAFE or FRAUD answer. The margin comes from
  `uncertainty_margin` on the language head when present, then the
  config-wide `uncertainty_margin`, and defaults to **0.08**.
- **⛔ MODEL UNAVAILABLE** — the trained model could not load or did not
  return a valid result. There is no keyword/demo fallback and no
  substitute verdict is presented as if it came from the model.

---

## Feedback and reinforcement loop

The system has two learning loops with different scopes:

### 1. Immediate local correction adapter

A correction immediately updates a small per-language logistic-head
delta in the offscreen model host:

- The MuRIL encoder remains frozen.
- The base weights in the active `config.json` are not modified.
- Updates use normalised SGD, divided by `||x||²`, so one correction has
  a bounded effect.
- The delta is norm-capped and anchored back toward the base model.
- Deltas persist in `chrome.storage.local` and can be reset from the
  popup, restoring the base model exactly.

### 2. Optional shared feedback for the next training cycle

When sharing is enabled, each actual correction (where the corrected
label differs from the model's binary prediction) creates one feedback
event with `reward: -1`: in reinforcement-learning terms, the model's
prediction action receives a negative reward and `corrected_label`
supplies the target action for preprocessing and retraining. Same-label
confirmations are not uploaded as corrections.

Events are stored in a background-service-worker sync queue under
`chrome.storage.local` key `fdFeedbackSyncQueue`. The worker:

- attempts delivery immediately;
- retries through a Chrome alarm roughly every five minutes;
- applies exponential backoff per failed event, capped at 15 minutes
  plus jitter;
- keeps failed events rather than silently dropping them; and
- offers **Sync now** in the popup.

The local CSV export remains available from the popup. Its message
column contains anonymized text.

---

## Feedback API contract

Configure the API base URL in the popup. The shipped default is the
placeholder:

```text
https://YOUR-BACKEND.vercel.app
```

The placeholder is treated as “backend not configured”: feedback can
queue locally, but no network delivery or model-update check is
attempted until a real backend URL is saved.

### Upload endpoint

```http
POST {apiBase}/v1/feedback
Content-Type: application/json
```

The request body is exactly one event object:

```json
{
  "event_id": "uuid-or-random-id",
  "anonymized_text": "Your account will be blocked. Open [URL] and share [OTP]",
  "language": "English",
  "head": "English",
  "predicted_label": 0,
  "predicted_probability": 0.1832,
  "corrected_label": 1,
  "reward": -1,
  "model_version": "3.0.0",
  "extension_version": "1.2.0",
  "timestamp": "2026-10-30T00:00:00.000Z",
  "consent": true
}
```

Field notes:

- `predicted_label` and `corrected_label` are binary: `1` = fraud,
  `0` = safe. For an UNCERTAIN result, `predicted_label` is the
  threshold-side action associated with the displayed probability.
- `predicted_probability` is the model's original fraud probability
  before the local correction update.
- `reward` is `-1` for uploaded corrections.
- `consent` is always `true` in a queued event because the worker only
  creates events while the opt-in setting is enabled. A production
  backend should nevertheless reject events where it is not `true`,
  validate every field, and deduplicate by `event_id`.
- No raw message, page URL, tab URL, user identity, or browser/device
  identifier is included.

Any 2xx response is treated as delivered and removes the event from the
queue. Non-2xx responses and network errors keep it queued for retry.

---

## Model config updates

The popup can check:

```http
GET {apiBase}/v1/model/latest
Accept: application/json
```

The response must include a model version and either an inline config
object or a config URL. Supported shapes include:

```json
{
  "model_version": "3.1.0",
  "config_url": "https://YOUR-BACKEND.vercel.app/model/3.1.0/config.json"
}
```

or:

```json
{
  "version": "3.1.0",
  "config": {
    "languages": {
      "English": { "arm": "own_struct", "threshold": 0.51 },
      "Hindi": { "arm": "own_struct", "threshold": 0.41 },
      "Telugu": { "arm": "own_struct", "threshold": 0.29 }
    }
  }
}
```

The inline example is abbreviated; a real config must contain the full
coefficient and scaler arrays described below.

Update flow:

1. The background worker compares the advertised semantic version with
   the active config version.
2. If newer, it downloads **only `config.json`**. It never downloads or
   attempts to replace the 239 MB ONNX encoder automatically.
3. It validates the config shape: all English, Hindi, and Telugu heads
   must be present; thresholds and optional uncertainty margins must be
   in range; intercepts must be finite; coefficient arrays must contain
   at least 768 finite numbers; embedding scalers must contain 768
   values; and structured-feature scalers must match any coefficients
   beyond the 768 embedding dimensions.
4. A valid download is **staged**, and the popup reports it.
5. **Apply staged** in the popup swaps the active config, asks the model
   host to reload, and keeps the previous config for rollback.
6. **Rollback** restores the previous downloaded config, or removes the
   downloaded override when the previous state was the bundled config.

As defence in depth, the offscreen model host validates a downloaded
active config again before using it and falls back to the bundled
config if the override is malformed.

---

## Install for development

1. Copy `model.onnx` from the v3 package into this component's `model/`
   folder.
2. Open Chrome → `chrome://extensions`.
3. Turn ON **Developer mode**.
4. Click **Load unpacked** and select this `extension` folder.
5. Open the toolbar popup and leave sharing OFF unless you are testing
   a backend you control.
6. To test uploads, replace the placeholder API base URL with your real
   backend URL and save it.

### Production backend domain — required before store submission

The production backend domain **must be set before store submission** in
two places:

1. Set the real API base URL for the production build/configuration;
   and
2. Replace the placeholder entry in `manifest.json`
   `host_permissions` (`https://your-backend.vercel.app/*`) with the
   exact production origin, for example
   `https://api.example.com/*`.

Do not submit the placeholder manifest to a store. The `<all_urls>`
permission remains because the on-page button, selection bubble, and
panel are core features that operate on pages the user visits; the
backend origin is listed separately so its purpose stays auditable.
The `alarms` permission is used only for feedback retry scheduling.

Your store privacy disclosure must match the behaviour above: local
inference, opt-in anonymized correction upload, purpose, retention, and
the fact that sharing is off by default.

---

## Pipeline inside the extension

```text
message ─► langdetect.js ─► model.js ─► offscreen model host ─► state
                                               │
                                               ├─ features.js
                                               ├─ anonymize.js
                                               ├─ tokenizer.js
                                               ├─ model.onnx (copy from v3)
                                               └─ active config.json
correction ─► local head delta (immediate)
           └─ if opted in ─► anonymize ─► background queue
                              ─► POST /v1/feedback with backoff
```

- **`anonymize.js`** — the train/serve-identical v3 preprocessing and
  placeholder masking. It is used for inference and again by the
  background worker before any feedback event is persisted for upload.
- **`features.js`** — the 30 structured features extracted from the raw
  message before masking.
- **`langdetect.js`** — script and Romanized-language detection; routes
  each message to a trained language head.
- **`tokenizer.js`** — MuRIL WordPiece using `model/vocab.json`.
- **`offscreen.js`** — ONNX Runtime Web model host, local correction
  adapter, active-config selection, and config-shape revalidation.
- **`background.js`** — extension lifecycle, model relay, settings,
  anonymized correction log, sync queue/backoff, and model config
  staging/application/rollback.
- **`model.js`** — prediction/feedback relay and uncertainty-state
  mapping. It contains no demo classifier.

## Model shipped with this component

The included config is the trained v3 config: frozen MuRIL embeddings +
per-language logistic heads fitted on the int8 encoder's own embeddings.
Cross-validated real-only F1 of that evaluated pipeline: English 0.93,
Hindi 0.94, Telugu 0.86. The extension version is 1.2.0; the bundled
model config version used for update comparison is 3.0.0 unless the
config supplies its own `model_version` or `version`.

## Files

| File | Job |
|---|---|
| `manifest.json` | Manifest V3 config and permissions |
| `background.js` | Service worker, feedback queue, model-update manager |
| `content.js` | Floating button, panel, selection bubble, verdict states |
| `content.css` | On-page styling, including UNCERTAIN and UNAVAILABLE |
| `anonymize.js` | Indic normalisation and privacy placeholders |
| `features.js` | 30 structured features |
| `langdetect.js` | Language/head routing |
| `tokenizer.js` | MuRIL WordPiece tokenizer |
| `model.js` | Prediction/feedback front door and uncertainty mapping |
| `offscreen.html/js` | Local ONNX model host and correction adapter |
| `popup.html/js` | Settings, opt-in, CSV export, sync, and model updates |
| `model/` | Config, vocab, ORT Web runtime; add `model.onnx` from v3 |

Note: the floating button cannot appear on `chrome://` internal pages
or extension-store pages — that is a Chrome restriction for every
extension.
