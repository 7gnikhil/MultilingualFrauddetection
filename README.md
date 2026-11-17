# Multilingual Fraud Message Detection

Classifies SMS/chat messages as fraud or genuine in Telugu, Hindi and English, built for Indian users. The system reads a message and answers one question — *fraud or genuine?* — with a probability and a per-language decision threshold.

**Current status**

- **Model:** 4.0.0 — per-language stacked model (see [Model v4.0.0](#model-v400)).
- **Browser extension:** v1.4.0 — the latest GitHub release is `extension-v1.4.0`. It is published on the [Microsoft Edge Add-ons store](https://microsoftedge.microsoft.com/addons/detail/jkinpaedkhihmiamdboahecceebjddof).
- **What it is:** a classifier. It classifies a message; it cannot block messages.

---

## Contents

1. [Model v4.0.0](#model-v400)
2. [Results](#results)
3. [Browser extension](#browser-extension)
4. [Components](#components)
5. [How a message is classified](#how-a-message-is-classified)
6. [The dataset](#the-dataset)
7. [How to run](#how-to-run)
8. [Limitations](#limitations)
9. [History](#history)
10. [File inventory](#file-inventory)

---

## Model v4.0.0

The final model is a **per-language stacked model**: a hashed lexical logistic model combined with a quantised MuRIL embedding model. Each language has its own decision threshold, tuned on validation data — never a global 0.5.

| Language | Decision threshold |
|---|---:|
| English | 0.40 |
| Hindi | 0.51 |
| Telugu | 0.36 |

A message with a fraud probability at or above its language threshold is classified **FRAUD**; otherwise it is classified **GENUINE** (shown as SAFE in the extension).

**Evaluation rule:** skeleton-grouped cross-validation, thresholds tuned only inside training folds, and the headline metric is **real-only F1** (synthetic paraphrases may help training; they never inflate the score). A skeleton is a message's template family — anonymised text, lowercased, with placeholders and punctuation removed — so near-identical templates never straddle train and test.

**Privacy in the pipeline:** PII — phone numbers, OTPs, URLs and account numbers, as well as emails and UPI ids — is masked to placeholders (`[PHONE]`, `[OTP]`, `[URL]`, `[ACCOUNT]`, `[EMAIL]`, `[UPI]`) identically in training and inference. The same preprocessing code runs at training time and at prediction time; any difference would be train/serve skew.

## Results

Grouped cross-validation F1 on real messages, at the thresholds above:

| Language | Grouped CV F1 (real messages) |
|---|---:|
| English | 0.954 |
| Hindi | 0.989 |
| Telugu | 0.919 |

On a frozen 64-message unseen challenge set, the model was **81.3% correct** — about 4 in 5 unseen messages correct. Cross-validation measures messages like the collected data; truly new messages behave worse, as that challenge result shows.

**Caveats that must travel with these numbers:** Hindi real-only evaluation rests on a small real set (248 real messages, of which only 72 are genuine/ham); Telugu rests on 273 real messages. Thin real genuine data means field performance will be lower than cross-validation. Do not quote the Hindi F1 as a field claim.

## Browser extension

The extension is in `chrome-extension/`. It checks messages entirely on-device: the model runs in the browser with ONNX Runtime Web, and the model never uploads the message for classification. It is published on the [Microsoft Edge Add-ons store](https://microsoftedge.microsoft.com/addons/detail/jkinpaedkhihmiamdboahecceebjddof); the latest GitHub release is `extension-v1.4.0`.

- **What it does:** drag a message onto the shield button, select text on a page, right-click selected text and choose "Check with Fraud Detector", paste or type a message into the panel, or use the full-page checker. It shows a verdict (FRAUD / LIKELY FRAUD / LOOKS SAFE), a confidence bar, the detected language and the main signals behind the verdict.
- **What it does not do:** it classifies; it cannot block messages. A verdict is evidence, not a guarantee.
- **Model in the extension:** the 239 MB int8 model (quantised MuRIL ONNX encoder plus the per-language heads), loaded once in an offscreen document and used by every tab. The JavaScript pipeline (`anonymize.js`, `features.js`, `langdetect.js`, `tokenizer.js`) mirrors the training preprocessing so train and serve see the same masked text.
- **Install from the store:** use the Edge Add-ons link above. To install from a release instead, download the extension package from the `extension-v1.4.0` release, unzip it, open `chrome://extensions` (or `edge://extensions`), enable Developer mode, choose Load unpacked, and select the extension folder.
- **Limitations:** about 4 in 5 correct on unseen messages (81.3% on the frozen 64-message challenge set). Genuine Telugu/Hindi bank alerts are the weakest case — fake alerts imitate real ones word-for-word, so the difference lives in context and structure, not vocabulary. The floating button cannot appear on `chrome://` internal pages or the Chrome Web Store; that is a browser restriction for every extension.

See `chrome-extension/README.md` for the extension's own documentation.

## Components

| Component | What it is |
|---|---|
| Dataset + training | Python, MuRIL (`google/muril-base-cased`, frozen encoder) and scikit-learn. `train.py` is the training pipeline: preprocessing, structured features, embeddings, grouped cross-validation, head selection and export. `fraud_training_v3.ipynb` is the same pipeline as Jupyter cells. |
| Browser extension | JavaScript, ONNX Runtime Web, 239 MB int8 model in `chrome-extension/`. Checks messages entirely on-device. |
| Feedback backend | FastAPI backend in `backend/`, deployed on Vercel (see `docs/BACKEND_VERCEL.md` and `backend/README.md`). It serves the model manifest (`GET /v1/model/latest`) and collects explicit user feedback (`POST /v1/feedback`). Feedback is opt-in and off by default; shared messages are anonymised on-device before upload, and the server rejects events that still look as if they contain a raw phone number, OTP, email, UPI id or URL. |
| CI/CD | GitHub Actions in `.github/workflows/`: `ci.yml`, `retrain.yml`, `release-extension.yml` and `edge-publish.yml`. Retraining is gated: a challenger model is promoted only through evaluation gates via pull request (see `docs/MODEL_PROMOTION_POLICY.md`). Nothing retrains or deploys from the feedback API itself. |

## How a message is classified

1. **Script routing:** Telugu characters → Telugu, Devanagari → Hindi, otherwise English. Romanised Hinglish/Telugish rides the English route unless labelled by hand.
2. **Normalisation:** NFC Unicode normalisation, then Indic NLP Library normalisation per script (removes zero-width joiner/non-joiner and zero-width space, canonicalises nukta and two-part vowel signs), then native Telugu/Devanagari digits → ASCII digits.
3. **Structured features:** 30 features are extracted from the raw text before anything is masked — shape, digit patterns, link/contact presence, amount size (`amount_log10`), OTP context, protective wording ("do not share", "bank never asks") and action/threat cues. The keyword sets include Telugu and Hindi terms.
4. **Masking:** URLs → `[URL]`, emails → `[EMAIL]`, UPI ids → `[UPI]`, grouped/long digit runs → `[ACCOUNT]`, Indian mobile numbers → `[PHONE]`, remaining 4–8-digit runs → `[OTP]`.
5. **Embedding:** the masked text is tokenised by MuRIL's WordPiece tokeniser (max 128 tokens) and passed through the frozen MuRIL encoder; token vectors are mean-pooled into one 768-dimension embedding. Telugu costs roughly 2.02 tokens per word in MuRIL vs about 1.40 for Devanagari.
6. **Stacked scoring:** the hashed lexical logistic model and the quantised MuRIL embedding model are combined per language, and the result is compared with that language's threshold.

Deliberately **not** done: no lowercasing (MuRIL is cased), no stemming/lemmatisation, no stopword removal (in fraud SMS the function words carry the label — "bank will *never* ask", "*do not* share"), no punctuation stripping, no translation to English and no script unification.

Worked example — a real fraud pattern, the "NO-reply" debit-alert scam:

> మీ బ్యాంక్ ఖాతా నుంచి రూ. 70,000 డెబిట్ అయ్యాయి. ఈ లావాదేవీ మీరు చేయకపోతే వెంటనే NO అని రిప్లై ఇవ్వండి.

It is routed to Telugu, the amount is masked to `[OTP]` in the text while its size survives in the structured features (`amount_log10`), and there is no protective warning wording — the message asks for a reply, which real debit alerts never do. A genuine alert ("Rs.2,500 credited… Do not share your OTP. Bank never asks for OTP") takes the identical path but has `mentions_do_not_share = 1` and `mentions_never_ask = 1`.

## The dataset

Training file: `train_balanced_v2.csv` — **7,624 rows**, balanced 1:1 per language, built from the collected-message sheet (5,558 rows after excluding 21 UNCERTAIN + 2 Maybe) plus 2,316 carried-over English corpus rows.

| Language | Fraud | Genuine (ham) |
|---|---:|---:|
| English | 2,785 | 2,785 |
| Hindi | 566 | 566 |
| Telugu | 461 | 461 |

Build rules, in order: dedupe first on anonymised text (rows that differ only in numbers are one template, not many examples), then balance 1:1 per language, dropping synthetic rows before real ones.

**Label audit:** an audit of the Telugu "fraud" seeds found about 35% were actually promotions, legitimate service messages or government awareness messages mislabelled as fraud — spam is not fraud. 49 rows were relabelled with an audit note and 21 uncertain rows are held out of training entirely. The real Hindi ham pool is only 72 unique messages and the real Telugu pool after dedupe is 125 fraud / 150 ham; genuine Telugu/Hindi ham barely exists in public sources, and that scarcity is the binding constraint on the results.

## How to run

```bash
pip install -r requirements.txt
python diagnose.py                        # Phase-0 audit (tokeniser, masking)
python train.py                           # full CV + final heads + ONNX export
python test_batch.py test_new_messages.csv
```

Or use `fraud_training_v3.ipynb` — the identical pipeline split into Jupyter cells (on Colab, select the T4 GPU runtime first; the embedding step is slow on CPU). First run downloads MuRIL (~950 MB); embeddings are cached in `outputs/` afterwards.

Outputs: `outputs/cv_results.json` (per-arm, per-language metrics), `outputs/oof_probs.npz`, `outputs/head_*.joblib`, `outputs/model/config.json` + `model.onnx` + tokenizer files.

**Testing protocol:** never paste Telugu/Hindi into a Windows console (PowerShell encoding mangles the script); always test through a UTF-8 CSV with `test_batch.py`. Test messages must be checked for overlap against training (skeleton + cleaned text) before they count as "unseen". Never tune thresholds on the test set. Report per-language precision/recall/F1 — a single accuracy number hides the Telugu/Hindi genuine-message problem.

For the backend, see `backend/README.md`; for deployment, see `docs/DEPLOYMENT.md`.

## Limitations

- About 4 in 5 correct on unseen messages (81.3% on the frozen 64-message challenge set); cross-validation F1 is higher because it measures messages like the collected data.
- Genuine Telugu/Hindi bank alerts are the weakest case: the model false-alarms on genuine messages that imitate fraud patterns, and real Telugu/Hindi genuine data is scarce (see [The dataset](#the-dataset)).
- Romanised routing: script-based routing sends Hinglish/Telugish to the English route.
- Sender metadata (sender ID / header) is the strongest known discriminator the current features cannot see, because the dataset holds message bodies only.
- The extension classifies; it cannot block messages, and a verdict is evidence, not a guarantee. Always verify a suspicious message with the bank or service through a channel you trust, and never share an OTP.
- Fine-tuning MuRIL (all parameters or the last 2–4 layers) is the researched next lever if more real data lands; frozen embeddings are a baseline, not the ceiling.

## History

- **v1 (24 Oct 2026):** frozen MuRIL + per-language logistic heads (fixed C=1.0, unscaled), NFC-only cleaning. Real-only F1: English 0.845, Hindi 0.784, Telugu 0.686, with Telugu precision 0.56 — 84 false positives on real Telugu rows. Every probability landed between 0.51 and 0.59.
- **v2 (24 Oct 2026):** bigger dataset (7,624 rows), same head; never run, superseded by v3 within a day.
- **v3 (25 Oct 2026):** research and audit first — Indic normalisation, 30 structured features extracted before masking, standardisation + C selected by inner cross-validation, a TF-IDF arm and a joint arm raced on identical splits, and a Telugu precision floor. Real-only CV F1: English 0.952, Hindi 0.992, Telugu 0.912; Telugu precision 0.56 → 0.87. The arms race showed the head and the normalisation, not the structured features, were the main fix. An earlier RoBERTa fine-tune had scored 1.00 through leakage (near-identical mutations in train and test); that is why every later evaluation is skeleton-grouped with a real-only headline.
- **v4.0.0 (current):** per-language stacked model (hashed lexical logistic model + quantised MuRIL embedding model), thresholds EN 0.40 / HI 0.51 / TE 0.36, grouped CV F1 English 0.954, Hindi 0.989, Telugu 0.919, and 81.3% on the frozen 64-message unseen challenge set. Shipped in the browser extension (v1.4.0) with the FastAPI feedback backend and gated retraining.

## File inventory

| File / directory | What it is |
|---|---|
| `train.py` | The training pipeline: preprocessing, features, embeddings, arms, export |
| `fraud_training_v3.ipynb` | The same pipeline as Jupyter cells |
| `diagnose.py` | Phase-0 audit: tokeniser fertility/UNK, placeholder tokenisation, masking audit |
| `test_batch.py` | Batch-test trained heads on a UTF-8 CSV; prints scorecard |
| `train_balanced_v2.csv` | Training data (7,624 rows) |
| `test_messages.csv` | The original 6 example messages |
| `test_new_messages.csv` | 30 unseen messages (6 regression + 24 new) |
| `requirements.txt` | numpy, pandas, scikit-learn, joblib, torch, transformers, onnx, onnxscript, indic-nlp-library |
| `outputs/` | `cv_results.json`, `oof_probs.npz`, `head_*.joblib`, `model/` (config.json, ONNX encoder, tokenizer) |
| `chrome-extension/` | Browser extension: JavaScript, ONNX Runtime Web, 239 MB int8 model, on-device checking |
| `backend/` | FastAPI feedback backend (Vercel), model manifest and opt-in feedback |
| `scripts/` | `package_extension.py`, `validate_manifest.py` |
| `docs/` | `BACKEND_VERCEL.md`, `DEPLOYMENT.md`, `MODEL_PROMOTION_POLICY.md` |
| `.github/workflows/` | `ci.yml`, `retrain.yml`, `release-extension.yml`, `edge-publish.yml` |

---

*Every number in this README comes from a recorded run or a stated release: the September 2025 baseline evaluation, the v1 run of 24 Oct 2026, the v3 run of 25 Oct 2026, and model v4.0.0 / extension v1.4.0. Where a number rests on thin data, the README says so next to the number.*
