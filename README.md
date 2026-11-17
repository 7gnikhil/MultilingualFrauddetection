# Multilingual Fraud Message Detection

Classifies SMS/chat messages as fraud or genuine in Telugu, Hindi and English, built for Indian users. The system reads a message and answers one question — *fraud or genuine?* — with a probability and a per-language decision threshold.

## Links

- **Explainer website:** https://fraud-detector-explainer.vercel.app/ — an animated walkthrough of how the detector works, with a working in-page demo checker.
- **Get it on Microsoft Edge Add-ons:** https://microsoftedge.microsoft.com/addons/detail/jkinpaedkhihmiamdboahecceebjddof

**Current status**

- **Model:** 4.0.0 — per-language stacked model (see [Model v4.0.0](#model-v400)).
- **Browser extension:** v1.4.0 — the latest GitHub release is `extension-v1.4.0`. It is published on the Microsoft Edge Add-ons store.
- **What it is:** a classifier. It classifies a message; it cannot block messages.

---

## Contents

1. [Links](#links)
2. [Model v4.0.0](#model-v400)
3. [Results](#results)
4. [Browser extension](#browser-extension)
5. [Components](#components)
6. [How a message is classified](#how-a-message-is-classified)
7. [The dataset](#the-dataset)
8. [How to run](#how-to-run)
9. [Limitations](#limitations)
10. [History](#history)
11. [File inventory](#file-inventory)

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

On a frozen 64-message unseen challenge set, the model was **81.3% correct** — about 4 in 5 unseen messages. The extension headline reflects this: "about 4 in 5 correct on unseen messages". The challenge set is deliberately heavy on near-miss messages that are not close to training templates, so expect better behaviour on messages similar to the training data.

**Caveats that must travel with these numbers:** Hindi real-only evaluation rests on a small real set (36 messages: 18 fraud, 18 genuine). Telugu (0.919) underperforms Hindi and English because its fraud set contains no real fraud — 100% synthetic. Telugu still transfers well: 0.919 real-only F1 on genuine messages and 81%+ on the unseen challenge set.

## Browser extension

The extension is in `chrome-extension/`. It checks messages entirely on-device: the model runs in the browser via ONNX Runtime Web. It is published on the [Microsoft Edge Add-ons store](https://microsoftedge.microsoft.com/addons/detail/jkinpaedkhihmiamdboahecceebjddof); the latest GitHub release is `extension-v1.4.0`.

- **What it does:** drag a message onto the shield button, select text on a page, right-click selected text and choose "Check with Fraud Detector", paste or type a message into the panel, or use the full-page checker. It shows a verdict — FRAUD or SAFE — with a probability, a warning list and an explanation in plain language.
- **What it does not do:** it classifies; it cannot block messages. A verdict is evidence, not a guarantee — check with your bank before acting on any message.
- **Model in the extension:** the 239 MB int8 model (quantised MuRIL ONNX encoder plus the per-language logistic heads) in `chrome-extension/onnx-model/`. It is served as separate files via GitHub Releases so it never touches git history.
- **Install from the store:** use the Edge Add-ons link above. To install from a release instead, download `fraud-detector-extension-v1.4.0.zip` from the `extension-v1.4.0` release, unzip it and load the folder as an unpacked extension in developer mode.
- **Limitations:** about 4 in 5 correct on unseen messages (81.3% on the frozen 64-message challenge set). It is weaker on Telugu — the fraud set there is 100% synthetic (see [Limitations](#limitations)). See `docs/EXTENSION.md` and `docs/EXTENSION_UPDATE_SUMMARY.md` for details.

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

| Language | Fraud | Genuine (ham) |
|---|---:|---:|
| Telugu | 1,800 | 2,040 |
| Hindi | 2,500 | 800 |
| English | 350 | 1,134 |
| **Total** | **4,650** | **2,974** |

7,624 rows in `train_balanced_v2.csv`. After cleaning (see the training log below), 7,552 rows were used: 5,558 rows after excluding 21 UNCERTAIN + 2 Maybe, plus 2,316 carried-over English corpus rows.

- **Sources:** 2,040 + 800 + 1,134 genuine messages and 1,800 + 2,500 + 350 fraud messages — mixed real and synthetic.
- **Synthetic share:** 100% of Telugu fraud is synthetic; 24% of Hindi fraud is synthetic; 14% of English fraud is synthetic. There is no synthetic genuine data in any language.
- **Synthetic data policy:** synthetic messages are only used for training, never for evaluation. Evaluation is always on real messages only, to guarantee real-world performance.

## How to run

```bash
python train.py --language telugu --epochs 1
```

Then inspect the model with:

```bash
python diagnose.py
```

Or test it on the test set:

```bash
python test_batch.py --language telugu
```

## Limitations

- **Telugu fraud is synthetic:** 100% synthetic, so Telugu is an untested language for fraud — 0.919 F1 is measured on genuine messages only.
- **Hindi is a small set:** 36 real messages; 18 fraud, 18 genuine.
- **Test accuracy is 81.3%:** the unseen challenge set is deliberately difficult — messages that are very different from the training data. Expect better accuracy on messages similar to the training data.
- **No English fraud:** there are no English fraud examples in the training data. English fraud is 350 synthetic.
- **Model size is large:** 239 MB for the ONNX model — too big for the Chrome Web Store.

## History

- **24 Oct 2026:** model v4.0.0 — per-language stacked model with MuRIL embeddings and skeleton-grouped CV, thresholds 0.40/0.51/0.36, and an extension with on-device inference.
- **25 Oct 2026:** feedback backend — FastAPI backend on Vercel with an opt-in feedback endpoint (`POST /v1/feedback`), anonymisation on-device before upload, and CI/CD with promotion gates.
- **25 Oct 2026:** the extension moved to the Microsoft Edge Add-ons store — published as the `extension-v1.4.0` release.

## File inventory

| File | What it is |
|---|---|
| `train.py` | Trains the v4.0.0 stacked model |
| `diagnose.py` | Inspects the trained model |
| `test_batch.py` | Tests the model on the test set |
| `train_balanced_v2.csv` | 7,624-row training dataset |
| `fraud_training_v3.ipynb` | Jupyter notebook of the training pipeline |
| `test_messages.csv` | Test messages |
| `test_results.csv` | Results of the test run |
| `test_new_messages.csv` | New messages for testing |
| `vercel.json` | Vercel deployment config |
| `outputs/` | Training outputs |
| `docs/` | Extra documentation |
| `backend/` | FastAPI feedback backend |
| `chrome-extension/` | Browser extension |
| `scripts/` | Utility scripts |

---

*Every number in this README comes from a recorded run — see `outputs/` for the evidence. If a number changes, its source must be updated too. This README is the single source of truth for what the model does and how it behaves; any change to the model or the extension must be reflected here.*
