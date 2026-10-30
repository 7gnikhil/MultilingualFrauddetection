/**
 * offscreen.js — the REAL trained-model inference engine.
 *
 * Runs in a Chrome offscreen document (an extension page, so WebAssembly
 * is allowed and the ~230 MB model loads ONCE for all tabs, not per page).
 * Pipeline — identical to training_v3/train.py at inference time:
 *   raw text
 *     -> structured features from the RAW text        (features.js)
 *     -> anonymize()                                   (anonymize.js)
 *     -> MuRIL WordPiece tokenize                      (tokenizer.js)
 *     -> MuRIL encoder ONNX (int8) -> mean pooling     (ort.min.js)
 *     -> per-language head: standardise (+ struct features for the
 *        emb+struct arms) -> logistic regression -> probability
 *     -> compare with the language's tuned threshold
 * Nothing leaves the device: the model files are inside the extension.
 */
(function () {
  "use strict";

  var state = { ready: false, error: null, config: null, tokenizer: null, session: null };
  var loading = null;

  function modelURL(name) { return chrome.runtime.getURL("model/" + name); }

  async function ensureReady() {
    if (state.ready) return;
    if (loading) return loading;
    loading = (async function () {
      try {
        ort.env.wasm.wasmPaths = modelURL("");
        ort.env.wasm.numThreads = 1;
        var cfgResp = await fetch(modelURL("config.json"));
        if (!cfgResp.ok) throw new Error("config.json missing from model/");
        state.config = await cfgResp.json();
        var vocabResp = await fetch(modelURL("vocab.json"));
        if (!vocabResp.ok) throw new Error("vocab.json missing from model/");
        state.tokenizer = FraudTokenizer.createTokenizer(await vocabResp.json());
        state.session = await ort.InferenceSession.create(modelURL("model.onnx"), {
          executionProviders: ["wasm"]
        });
        state.ready = true;
      } catch (e) {
        state.error = (e && e.message) ? e.message : String(e);
        loading = null;
        throw e;
      }
    })();
    return loading;
  }

  function sigmoid(z) { return 1 / (1 + Math.exp(-z)); }

  function scale(vec, mean, scaleArr) {
    var out = new Array(vec.length);
    for (var i = 0; i < vec.length; i++) out[i] = (vec[i] - mean[i]) / scaleArr[i];
    return out;
  }

  async function predict(rawText, headName) {
    await ensureReady();
    var head = state.config.languages[headName];
    if (!head) throw new Error("No trained head for " + headName);

    var struct = FraudFeatures.extractStructFeatures(
      rawText, FraudPreprocess.translateNativeDigits);
    var clean = FraudAnonymize(rawText);
    var enc = state.tokenizer.encode(clean);
    var L = enc.input_ids.length;
    var feeds = {
      input_ids: new ort.Tensor("int64", BigInt64Array.from(enc.input_ids.map(function (x) { return BigInt(x); })), [1, L]),
      attention_mask: new ort.Tensor("int64", BigInt64Array.from(enc.attention_mask.map(function (x) { return BigInt(x); })), [1, L])
    };
    var out = await state.session.run(feeds);
    var hidden = out.last_hidden_state;
    var dims = hidden.dims, data = hidden.data;
    var H = dims[2];
    var emb = new Array(H).fill(0);
    var maskSum = 0;
    for (var i = 0; i < L; i++) {
      var m = enc.attention_mask[i];
      maskSum += m;
      if (!m) continue;
      for (var j = 0; j < H; j++) emb[j] += data[i * H + j];
    }
    for (var k = 0; k < H; k++) emb[k] /= maskSum;

    var x = scale(emb, head.scaler_e_mean, head.scaler_e_scale);
    if (head.arm === "own_struct" || head.arm === "joint_struct") {
      var ss = scale(struct, head.scaler_s_mean, head.scaler_s_scale);
      x = x.concat(ss);
    }
    var z = head.intercept;
    for (var n = 0; n < x.length; n++) z += head.coef[n] * x[n];
    return {
      probability: sigmoid(z),
      threshold: head.threshold,
      arm: head.arm,
      demo: false
    };
  }

  chrome.runtime.onMessage.addListener(function (msg, sender, sendResponse) {
    if (!msg || msg.target !== "fd-offscreen") return;
    if (msg.type === "FD_PREDICT") {
      predict(msg.text, msg.head)
        .then(function (res) { sendResponse({ ok: true, result: res }); })
        .catch(function (e) { sendResponse({ ok: false, error: (e && e.message) || String(e) }); });
      return true;
    }
    if (msg.type === "FD_STATUS") {
      sendResponse({ ok: true, ready: state.ready, error: state.error });
    }
  });

  // Pre-warm so the first real check is fast.
  ensureReady().catch(function () { /* error surfaces on first predict */ });
})();
