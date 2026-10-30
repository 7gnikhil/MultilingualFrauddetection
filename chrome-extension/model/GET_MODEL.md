# The trained encoder (model.onnx) is NOT in git

`model.onnx` is 239MB — over GitHub's 100MB per-file limit — so it is
distributed as a **Release asset** on this repository instead.

## To run the extension from a fresh clone
1. Download `model.onnx` from the repo's **Releases** page.
2. Put it in this folder (`chrome-extension/model/model.onnx`).
3. Load the extension: `chrome://extensions` → Developer mode →
   **Load unpacked** → select the `chrome-extension` folder.

## What the file is
- Encoder: `google/muril-base-cased`, exported to ONNX (opset 14,
  inputs `input_ids` / `attention_mask`, output `last_hidden_state`),
  then dynamic-int8 quantized (950MB → 239MB).
- The classifier heads in `config.json` were fitted on embeddings
  produced by THIS int8 encoder (not the FP32 one): pooled MuRIL
  embeddings have tiny per-dimension variance, so the standardiser
  amplifies FP32→int8 drift into flipped verdicts. Fitting on the
  deployed encoder's embeddings is the fix.
- Cross-validated real-only F1 of the shipped int8 pipeline:
  English 0.931 · Hindi 0.945 · Telugu 0.855.
