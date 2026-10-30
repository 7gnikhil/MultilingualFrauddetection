# 🛡️ Fraud Detector — Chrome Extension

Checks a message for fraud in **Telugu, Hindi (including Romanized / Hinglish)
and English** — right on the page where you read it: WhatsApp Web, Gmail,
Truecaller web, anywhere.

**Privacy first: the check runs 100% locally inside the extension.
No message is ever sent to a server or leaves your browser.**

---

## How it works (both ways you asked for)

1. **It stays on screen.** After install, a round 🛡️ button appears at the
   bottom-right of every page and stays visible. On first run the panel opens
   once to ask/confirm keeping it on screen. **Drag the button anywhere** —
   it remembers its position on that site.
2. **Click the button** → the check panel opens.
3. **Drag a message in** — drag selected text from the page and drop it onto
   the 🛡️ button or into the panel's drop zone. The verdict appears instantly.
4. **Or just select text** — highlight any message text on a page and a small
   **“🛡️ Check if fraud”** bubble pops up next to your selection. Click it.
5. **Or right-click** selected text → **“Check with Fraud Detector.”**
6. You can also paste/type a message into the panel manually.

**Verdict panel shows:** 🚨 FRAUD / ⚠️ LIKELY FRAUD / ✅ LOOKS SAFE, a
confidence bar, the detected language, and the main signals behind the
verdict (from the model's structured features).

---

## Install (Load Unpacked — takes 1 minute)

1. Unzip `fraud-detector-chrome-extension.zip` (or use the `extension/`
   folder as-is).
2. Open Chrome → `chrome://extensions`
3. Turn ON **Developer mode** (top-right toggle).
4. Click **Load unpacked** → select the **`extension`** folder.
5. Pin it from the 🧩 puzzle menu if you want the toolbar icon.
6. Open any page — the 🛡️ button is bottom-right. Drag a message onto it.

To update later: replace the folder contents, then click the ↻ refresh
icon on the extension card in `chrome://extensions`.

---

## Pipeline inside the extension (identical to training v3)

```
message ─► langdetect.js ─► model.js ─► (offscreen model host) ─► verdict
                              relays      offscreen.js:
                                          features.js  (structured feats, raw text)
                                          anonymize.js (Indic norm + placeholders)
                                          tokenizer.js (MuRIL WordPiece)
                                          model.onnx   (MuRIL encoder, int8)
                                          config.json  (per-language heads + thresholds)
```

- **`anonymize.js`** — a line-for-line port of the v3 training
  preprocessing: Indic NLP normalisation (Telugu/Hindi), native-digit
  translation, then `[OTP]` `[PHONE]` `[ACCOUNT]` `[UPI]` `[EMAIL]`
  `[URL]` placeholder masking. Training on placeholders while serving
  raw digits = train/serve skew, so this file must never drift from
  `training_v3/train.py`.
- **`features.js`** — the 30 structured features training extracts from
  the RAW message before masking (link type, digit patterns, OTP
  context, do-not-share wording…), ported 1:1.
- **`langdetect.js`** — script-based detection (Telugu / Devanagari)
  plus Romanized Telugu & Hinglish cues; routes each message to its
  trained language head.
- **`tokenizer.js`** — MuRIL WordPiece in JS, using `model/vocab.json`,
  the token→id map dumped from the training tokenizer itself.
- **`offscreen.js`** — hosts ONNX Runtime Web + the model in an
  offscreen document, so the ~230 MB model loads **once** and serves
  every tab. `model.js` is just the relay; if the model host ever
  fails, a keyword stub answers instead and the verdict is honestly
  badged **“DEMO MODEL”**.

## ✅ Current model = THE REAL TRAINED MODEL

This build ships the trained v3 model (frozen MuRIL embeddings +
per-language logistic heads on embeddings + structured features,
fitted on the int8 encoder's own embeddings so the deployed model is
exactly the evaluated model) — see `model/README.md`. Verdicts use
each language's tuned threshold, and the JS pipeline was verified
byte-identical to the Python training pipeline on sample messages.
Cross-validated real-only F1 of the shipped model: English 0.93,
Hindi 0.94, Telugu 0.86.

## Files

| File | Job |
|---|---|
| `manifest.json` | Manifest V3 config, permissions (`offscreen` included), CSP for WASM |
| `background.js` | Right-click menu, toolbar toggle, offscreen host management + relay |
| `content.js` | Floating draggable button, panel, drop zone, selection bubble |
| `content.css` | All on-page styling (`fd-` prefixed, no page clashes) |
| `anonymize.js` | Train/serve-identical v3 preprocessing (Indic norm + placeholders) |
| `features.js` | The 30 structured features, ported from training |
| `langdetect.js` | Telugu / Hindi / Romanized / English detection + head routing |
| `tokenizer.js` | MuRIL WordPiece tokenizer (uses `model/vocab.json`) |
| `model.js` | Prediction front door: relays to the model host, demo stub fallback |
| `offscreen.html/js` | The model host page: ORT Web + encoder + heads |
| `popup.html/js` | Toolbar popup |
| `model/` | Trained model files: `model.onnx`, `vocab.json`, `config.json`, ORT Web runtime |

Note: the floating button cannot appear on `chrome://` internal pages or
the Chrome Web Store — that's a Chrome restriction for every extension.
