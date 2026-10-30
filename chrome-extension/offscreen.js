/**
 * offscreen.js — the REAL trained-model inference engine + on-device
 * feedback learning (v1.1.0).
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
 *
 * FEEDBACK LEARNING (online logistic-regression updates):
 * The encoder stays frozen. When the user flags a verdict ("actually
 * fraud" / "actually safe"), the head for that language takes a few
 * normalised SGD steps on that one message (step divided by ||x||^2,
 * so a single flag moves the logit by a bounded amount). The delta is
 * kept SEPARATE from the trained weights, capped
 * in norm, pulled back toward zero by the anchor term, persisted in
 * chrome.storage.local, and can be reset at any time — the base model
 * in model/config.json is never modified.
 */
(function () {
  "use strict";

  var LEARN_STEP = 2.0;     // normalised SGD step (see learn())
  var LEARN_EPOCHS = 4;     // passes over the single flagged message
  var LEARN_ANCHOR = 0.001; // per-pass pull of the delta back to the base model
  var DELTA_CAP = 6.0;      // max L2 norm of a language's delta
  var DELTAS_KEY = "fdLearnDeltas";

  function sigmoid(z) { return 1 / (1 + Math.exp(-z)); }

  function scale(vec, mean, scaleArr) {
    var out = new Array(vec.length);
    for (var i = 0; i < vec.length; i++) out[i] = (vec[i] - mean[i]) / scaleArr[i];
    return out;
  }

  function createPredictor(deps) {
    var ort = deps.ort, session = deps.session, config = deps.config;
    var tokenizer = deps.tokenizer;
    var deltas = {}; // headName -> { w: number[], b: number }

    function getDelta(headName, dim) {
      var d = deltas[headName];
      if (!d || d.w.length !== dim) {
        d = { w: new Array(dim).fill(0), b: 0 };
        deltas[headName] = d;
      }
      return d;
    }

    async function embedAndFeatures(rawText) {
      var struct = FraudFeatures.extractStructFeatures(
        rawText, FraudPreprocess.translateNativeDigits);
      var clean = FraudAnonymize(rawText);
      var enc = tokenizer.encode(clean);
      var L = enc.input_ids.length;
      var feeds = {
        input_ids: new ort.Tensor("int64", BigInt64Array.from(enc.input_ids.map(function (x) { return BigInt(x); })), [1, L]),
        attention_mask: new ort.Tensor("int64", BigInt64Array.from(enc.attention_mask.map(function (x) { return BigInt(x); })), [1, L])
      };
      var out = await session.run(feeds);
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
      return { emb: emb, struct: struct };
    }

    function headVector(head, emb, struct) {
      var x = scale(emb, head.scaler_e_mean, head.scaler_e_scale);
      if (head.arm === "own_struct" || head.arm === "joint_struct") {
        x = x.concat(scale(struct, head.scaler_s_mean, head.scaler_s_scale));
      }
      return x;
    }

    function score(head, headName, x) {
      var d = deltas[headName];
      var z = head.intercept + (d ? d.b : 0);
      for (var n = 0; n < x.length; n++) {
        z += (head.coef[n] + (d ? d.w[n] : 0)) * x[n];
      }
      return sigmoid(z);
    }

    async function predict(rawText, headName) {
      var head = config.languages[headName];
      if (!head) throw new Error("No trained head for " + headName);
      var ef = await embedAndFeatures(rawText);
      var x = headVector(head, ef.emb, ef.struct);
      return {
        probability: score(head, headName, x),
        threshold: head.threshold,
        arm: head.arm,
        demo: false
      };
    }

    /**
     * learn(rawText, headName, label) — label: 1 = fraud, 0 = safe.
     * Runs SGD on the head delta for this one message, in real time.
     * Returns { probabilityBefore, probability (after), threshold }.
     */
    async function learn(rawText, headName, label) {
      var head = config.languages[headName];
      if (!head) throw new Error("No trained head for " + headName);
      var ef = await embedAndFeatures(rawText);
      var x = headVector(head, ef.emb, ef.struct);
      var d = getDelta(headName, head.coef.length);
      var before = score(head, headName, x);
      var y = label ? 1 : 0;
      // Normalised SGD: the step is divided by ||x||^2, so one flag moves
      // the flagged message's logit by roughly LEARN_STEP — not by the
      // raw (huge) norm of a standardised embedding vector. Unnormalised
      // updates measured ~15 logit units for a single flag and flipped
      // unrelated verdicts; this keeps corrections local and gradual.
      var xNorm2 = 1;
      for (var q = 0; q < x.length; q++) xNorm2 += x[q] * x[q];
      for (var epoch = 0; epoch < LEARN_EPOCHS; epoch++) {
        var p = score(head, headName, x);
        var err = p - y;
        if (Math.abs(err) < 1e-3) break;
        var step = LEARN_STEP * err / xNorm2;
        for (var i = 0; i < x.length; i++) {
          d.w[i] -= step * x[i] + LEARN_ANCHOR * d.w[i];
        }
        d.b -= step + LEARN_ANCHOR * d.b;
      }
      // Cap the delta so personal learning can tune, never hijack.
      var norm = Math.sqrt(d.w.reduce(function (s, v) { return s + v * v; }, 0) + d.b * d.b);
      if (norm > DELTA_CAP) {
        var f = DELTA_CAP / norm;
        for (var j = 0; j < d.w.length; j++) d.w[j] *= f;
        d.b *= f;
      }
      return {
        probabilityBefore: before,
        probability: score(head, headName, x),
        threshold: head.threshold
      };
    }

    function setDeltas(obj) {
      if (!obj) return;
      Object.keys(obj).forEach(function (k) {
        var d = obj[k];
        if (d && Array.isArray(d.w)) {
          deltas[k] = { w: d.w.slice(), b: typeof d.b === "number" ? d.b : 0 };
        }
      });
    }

    function getDeltas() {
      var out = {};
      Object.keys(deltas).forEach(function (k) {
        out[k] = { w: deltas[k].w.slice(), b: deltas[k].b };
      });
      return out;
    }

    function resetLearning() { deltas = {}; }

    function correctionCount() {
      // Deltas alone don't count corrections; background tracks those.
      return Object.keys(deltas).length;
    }

    return {
      predict: predict,
      learn: learn,
      setDeltas: setDeltas,
      getDeltas: getDeltas,
      resetLearning: resetLearning,
      correctionCount: correctionCount
    };
  }

  // Node/test hook (not used in the browser).
  if (typeof module !== "undefined" && module.exports) {
    module.exports = { createPredictor: createPredictor };
    return;
  }

  // ---------------- Browser wiring (offscreen document) ----------------
  var state = { ready: false, error: null, predictor: null };
  var loading = null;

  function modelURL(name) { return chrome.runtime.getURL("model/" + name); }

  function storageGet(key) {
    return new Promise(function (resolve) {
      try {
        chrome.storage.local.get(key, function (o) { resolve(o && o[key]); });
      } catch (e) { resolve(undefined); }
    });
  }
  function storageSet(obj) {
    return new Promise(function (resolve) {
      try { chrome.storage.local.set(obj, function () { resolve(); }); }
      catch (e) { resolve(); }
    });
  }

  async function ensureReady() {
    if (state.ready) return;
    if (loading) return loading;
    loading = (async function () {
      try {
        ort.env.wasm.wasmPaths = modelURL("");
        ort.env.wasm.numThreads = 1;
        var cfgResp = await fetch(modelURL("config.json"));
        if (!cfgResp.ok) throw new Error("config.json missing from model/");
        var config = await cfgResp.json();
        var vocabResp = await fetch(modelURL("vocab.json"));
        if (!vocabResp.ok) throw new Error("vocab.json missing from model/");
        var tokenizer = FraudTokenizer.createTokenizer(await vocabResp.json());
        var session = await ort.InferenceSession.create(modelURL("model.onnx"), {
          executionProviders: ["wasm"]
        });
        state.predictor = createPredictor({
          ort: ort, session: session, config: config, tokenizer: tokenizer
        });
        var saved = await storageGet(DELTAS_KEY);
        if (saved) state.predictor.setDeltas(saved);
        state.ready = true;
      } catch (e) {
        state.error = (e && e.message) ? e.message : String(e);
        loading = null;
        throw e;
      }
    })();
    return loading;
  }

  chrome.runtime.onMessage.addListener(function (msg, sender, sendResponse) {
    if (!msg || msg.target !== "fd-offscreen") return;
    if (msg.type === "FD_PREDICT") {
      ensureReady()
        .then(function () { return state.predictor.predict(msg.text, msg.head); })
        .then(function (res) { sendResponse({ ok: true, result: res }); })
        .catch(function (e) { sendResponse({ ok: false, error: (e && e.message) || String(e) }); });
      return true;
    }
    if (msg.type === "FD_LEARN") {
      ensureReady()
        .then(function () { return state.predictor.learn(msg.text, msg.head, msg.label); })
        .then(function (res) {
          var payload = {};
          payload[DELTAS_KEY] = state.predictor.getDeltas();
          return storageSet(payload).then(function () {
            sendResponse({ ok: true, result: res });
          });
        })
        .catch(function (e) { sendResponse({ ok: false, error: (e && e.message) || String(e) }); });
      return true;
    }
    if (msg.type === "FD_RESET_LEARNING") {
      ensureReady()
        .then(function () {
          state.predictor.resetLearning();
          var payload = {};
          payload[DELTAS_KEY] = {};
          return storageSet(payload);
        })
        .then(function () { sendResponse({ ok: true }); })
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
