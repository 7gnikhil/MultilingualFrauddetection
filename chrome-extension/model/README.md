# model/ — THE TRAINED MODEL (v3) LIVES HERE

This folder ships with the **real trained model** inside — the same
pipeline as `training_v3/train.py`, exported for the browser:

| File | What it is |
|---|---|
| `model.onnx` | MuRIL encoder (`google/muril-base-cased`), dynamic-int8 quantized ONNX, outputs `last_hidden_state` |
| `vocab.json` | Token → id map dumped from the training tokenizer itself (transformers `get_vocab()`), so JS tokenization matches training exactly |
| `config.json` | Per-language shipped heads (all `own_struct`: embeddings + structured features), fitted on the INT8 encoder's own embeddings — standardiser mean/scale, logistic-regression coefficients + intercept, and the tuned decision threshold per language |
| `ort.min.js` + `ort-wasm-simd-threaded.wasm` / `.mjs` | ONNX Runtime Web 1.20.1 (CPU/WASM) |

How a check flows (all inside `offscreen.js`, an extension page where
the model loads once for all tabs):

raw message → structured features from the raw text (`features.js`)
→ `anonymize()` (`anonymize.js`, Indic normalisation + placeholders)
→ WordPiece tokenize (`tokenizer.js`) → MuRIL ONNX → mean pooling
→ per-language head (standardise → logistic regression) → probability
vs the language's tuned threshold.

Verified before shipping: the JS cleaning, structured features and
tokenizer produce byte-identical results to the Python training
pipeline on 108 sample messages. The heads in `config.json` were
fitted on embeddings produced by THIS int8 encoder (not the FP32 one):
pooled MuRIL embeddings have tiny per-dimension variance, so the
standardiser amplifies raw FP32→int8 drift (~0.99 cosine) into flipped
verdicts — fitting the heads on the deployed encoder's embeddings is
the fix. End-to-end, the JS stack's probabilities match the Python
int8 pipeline to ~3e-5 (verdict agreement 64/64 on the labeled test
set). CV real-only F1 of the shipped int8 pipeline: English 0.931,
Hindi 0.945, Telugu 0.855 (FP32 pipeline: 0.952 / 0.989 / 0.900).
