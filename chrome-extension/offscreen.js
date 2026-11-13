/**
 * offscreen.js — the REAL trained-model inference engine + on-device
 * feedback learning (model v4 stack; extension v1.3.0).
 *
 * Runs in a Chrome offscreen document (an extension page, so WebAssembly
 * is allowed and the ~230 MB model loads ONCE for all tabs, not per page).
 * Pipeline (ml-final/train_final_v4.py):
 *   raw text
 *     -> structured features from the RAW text        (features.js)
 *     -> anonymize()                                   (anonymize.js)
 *     -> Arm A: hashed lexical LR over the anonymized text — word(1,2)
 *        n-grams @2^14 + char_wb(3,5) n-grams @2^15    (lexical.js)
 *     -> Arm B: MuRIL WordPiece tokenize               (tokenizer.js)
 *        -> MuRIL encoder ONNX (int8) -> mean pooling  (ort.min.js)
 *        -> per-language head: standardise (embeddings + struct
 *           features) -> logistic regression
 *     -> meta stack: sigmoid(a + b*logit(pA) + c*logit(pB))
 *     -> compare with the language's tuned threshold
 * A head without hash/meta fields is scored as a legacy v3 single head
 * (arm B only), so older downloaded configs keep working.
 * Inference itself never leaves the device: the model files are inside
 * the extension. Optional feedback upload is handled separately by the
 * background worker and only after explicit opt-in.
 *
 * FEEDBACK LEARNING (online logistic-regression updates):
 * The encoder stays frozen. When the user flags a verdict ("actually
 * fraud" / "actually safe"), the delta for that language takes a few
 * normalised SGD steps on that one message (step divided by ||x||^2,
 * so a single flag moves the logit by a bounded amount), applied on top
 * of the final stacked logit. The delta is
 * kept SEPARATE from the trained weights, capped
 * in norm, pulled back toward zero by the anchor term, persisted in
 * chrome.storage.local, and can be reset at any time — the base model
 * in model/config_v4.json is never modified.
 */
(function () {
  "use strict";

  var LEARN_STEP = 2.0;     // normalised SGD step (see learn())
  var LEARN_EPOCHS = 4;     // passes over the single flagged message
  var LEARN_ANCHOR = 0.001; // per-pass pull of the delta back to the base model
  var DELTA_CAP = 6.0;      // max L2 norm of a language's delta
  var DELTAS_KEY = "fdLearnDeltas";
  var ACTIVE_CONFIG_KEY = "fdActiveModelConfig";
  var MODEL_STATE_KEY = "fdModelUpdateState";
  var BUNDLED_MODEL_VERSION = "4.0.0";
  var DEFAULT_UNCERTAINTY_MARGIN = 0.08;

  function sigmoid(z) { return 1 / (1 + Math.exp(-z)); }

  function logit(p) {
    var c = Math.min(1 - 1e-6, Math.max(1e-6, p));
    return Math.log(c / (1 - c));
  }

  function scale(vec, mean, scaleArr) {
    var out = new Array(vec.length);
    for (var i = 0; i < vec.length; i++) out[i] = (vec[i] - mean[i]) / scaleArr[i];
    return out;
  }

  function configVersion(config) {
    if (!config) return BUNDLED_MODEL_VERSION;
    var version = config.model_version || config.version || config.modelVersion;
    return version == null ? BUNDLED_MODEL_VERSION : String(version);
  }

  function validMargin(value) {
    return typeof value === "number" && Number.isFinite(value) && value >= 0 && value <= 0.49;
  }

  function uncertaintyMargin(config, head) {
    if (head && validMargin(head.uncertainty_margin)) return head.uncertainty_margin;
    if (config && validMargin(config.uncertainty_margin)) return config.uncertainty_margin;
    return DEFAULT_UNCERTAINTY_MARGIN;
  }

  function isFiniteArray(value, expectedLength) {
    if (!Array.isArray(value)) return false;
    if (typeof expectedLength === "number" && value.length !== expectedLength) return false;
    for (var i = 0; i < value.length; i++) {
      if (typeof value[i] !== "number" || !Number.isFinite(value[i])) return false;
    }
    return true;
  }

  function isFiniteNumber(value) {
    return typeof value === "number" && Number.isFinite(value);
  }

  function isValidCoefPairs(pairs, nFeatures) {
    if (!Array.isArray(pairs)) return false;
    for (var i = 0; i < pairs.length; i++) {
      var p = pairs[i];
      if (!Array.isArray(p) || p.length !== 2 ||
          !Number.isInteger(p[0]) || p[0] < 0 || p[0] >= nFeatures ||
          !isFiniteNumber(p[1])) return false;
    }
    return true;
  }

  // v4 stack fields: a head either carries BOTH hash and meta (stacked
  // model) or neither (legacy v3 single head). Half a stack is unusable.
  function isValidStack(head) {
    var hasHash = !!head.hash, hasMeta = !!head.meta;
    if (!hasHash && !hasMeta) return true;
    if (!hasHash || !hasMeta) return false;
    var h = head.hash, m = head.meta;
    return Number.isInteger(h.word_n_features) && h.word_n_features > 0 &&
      Number.isInteger(h.char_n_features) && h.char_n_features > 0 &&
      isFiniteNumber(h.intercept) &&
      isValidCoefPairs(h.word_coef, h.word_n_features) &&
      isValidCoefPairs(h.char_coef, h.char_n_features) &&
      isFiniteNumber(m.intercept) && isFiniteNumber(m.weight_hash) &&
      isFiniteNumber(m.weight_int8);
  }

  // Defence in depth: background validates a downloaded config before it
  // becomes active, and the model host validates it again before use. A
  // malformed override is ignored in favour of the bundled config.
  function isUsableConfig(config) {
    if (!config || typeof config !== "object" || !config.languages) return false;
    var languages = ["English", "Hindi", "Telugu"];
    for (var i = 0; i < languages.length; i++) {
      var head = config.languages[languages[i]];
      if (!head || typeof head.arm !== "string" ||
          typeof head.threshold !== "number" || head.threshold < 0 || head.threshold > 1 ||
          typeof head.intercept !== "number" || !Number.isFinite(head.intercept) ||
          !isFiniteArray(head.coef) || head.coef.length < 768 ||
          !isFiniteArray(head.scaler_e_mean, 768) ||
          !isFiniteArray(head.scaler_e_scale, 768)) return false;
      var structLength = head.coef.length - 768;
      if (structLength > 0 &&
          (!isFiniteArray(head.scaler_s_mean, structLength) ||
           !isFiniteArray(head.scaler_s_scale, structLength))) return false;
      if (!isValidStack(head)) return false;
    }
    return true;
  }

  function createPredictor(deps) {
    var ort = deps.ort, session = deps.session, config = deps.config;
    var tokenizer = deps.tokenizer;
    var modelVersion = deps.modelVersion || configVersion(config);
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
      return { emb: emb, struct: struct, clean: clean };
    }

    function headVector(head, emb, struct) {
      var x = scale(emb, head.scaler_e_mean, head.scaler_e_scale);
      // Struct features are appended whenever the head carries struct
      // scalers (v3 "own_struct"/"joint_struct" heads and the v4 stack's
      // arm B alike) — keyed on the scalers, not the arm label, so a
      // renamed arm can never silently drop the struct block.
      if (head.scaler_s_mean && head.scaler_s_scale) {
        x = x.concat(scale(struct, head.scaler_s_mean, head.scaler_s_scale));
      }
      return x;
    }

    function armBLogit(head, x) {
      var z = head.intercept;
      for (var n = 0; n < x.length; n++) z += head.coef[n] * x[n];
      return z;
    }

    /**
     * Base stacked logit (no learning delta).
     * v4 head (hash + meta present):
     *   z = meta.intercept + meta.weight_hash * logit(p_hash)
     *                      + meta.weight_int8 * logit(p_emb)
     * Legacy v3 head: z = arm B logit alone.
     * Returns { z, probHash, probEmb } (probHash is null for v3 heads).
     */
    function baseLogit(head, clean, x) {
      var zB = armBLogit(head, x);
      var probEmb = sigmoid(zB);
      if (head.hash && head.meta && typeof FraudLexical !== "undefined") {
        var probHash = FraudLexical.probability(clean, head.hash);
        var z = head.meta.intercept +
          head.meta.weight_hash * logit(probHash) +
          head.meta.weight_int8 * logit(probEmb);
        return { z: z, probHash: probHash, probEmb: probEmb };
      }
      return { z: zB, probHash: null, probEmb: probEmb };
    }

    function score(head, headName, clean, x) {
      var d = deltas[headName];
      var z = baseLogit(head, clean, x).z + (d ? d.b : 0);
      if (d) {
        for (var n = 0; n < x.length; n++) z += d.w[n] * x[n];
      }
      return sigmoid(z);
    }

    async function predict(rawText, headName) {
      var head = config.languages[headName];
      if (!head) throw new Error("No trained head for " + headName);
      var ef = await embedAndFeatures(rawText);
      var x = headVector(head, ef.emb, ef.struct);
      var base = baseLogit(head, ef.clean, x);
      return {
        probability: score(head, headName, ef.clean, x),
        probHash: base.probHash,
        probEmb: base.probEmb,
        threshold: head.threshold,
        uncertaintyMargin: uncertaintyMargin(config, head),
        modelVersion: modelVersion,
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
      var before = score(head, headName, ef.clean, x);
      var y = label ? 1 : 0;
      // Normalised SGD: the step is divided by ||x||^2, so one flag moves
      // the flagged message's logit by roughly LEARN_STEP — not by the
      // raw (huge) norm of a standardised embedding vector. Unnormalised
      // updates measured ~15 logit units for a single flag and flipped
      // unrelated verdicts; this keeps corrections local and gradual.
      var xNorm2 = 1;
      for (var q = 0; q < x.length; q++) xNorm2 += x[q] * x[q];
      for (var epoch = 0; epoch < LEARN_EPOCHS; epoch++) {
        var p = score(head, headName, ef.clean, x);
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
        probability: score(head, headName, ef.clean, x),
        threshold: head.threshold,
        uncertaintyMargin: uncertaintyMargin(config, head),
        modelVersion: modelVersion
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
  var state = { ready: false, error: null, predictor: null, modelVersion: null, configSource: null };
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
        var cfgResp = await fetch(modelURL("config_v4.json"));
        if (!cfgResp.ok) throw new Error("config_v4.json missing from model/");
        var bundledConfig = await cfgResp.json();
        if (!isUsableConfig(bundledConfig)) throw new Error("Bundled config_v4.json has an invalid shape");
        var downloadedConfig = await storageGet(ACTIVE_CONFIG_KEY);
        var modelState = await storageGet(MODEL_STATE_KEY);
        var useDownloaded = isUsableConfig(downloadedConfig);
        var config = useDownloaded ? downloadedConfig : bundledConfig;
        var activeVersion = useDownloaded
          ? ((modelState && modelState.activeVersion) || configVersion(config))
          : configVersion(bundledConfig);
        state.modelVersion = String(activeVersion);
        state.configSource = useDownloaded ? "downloaded" : "bundled";
        var vocabResp = await fetch(modelURL("vocab.json"));
        if (!vocabResp.ok) throw new Error("vocab.json missing from model/");
        var tokenizer = FraudTokenizer.createTokenizer(await vocabResp.json());
        var session = await ort.InferenceSession.create(modelURL("model.onnx"), {
          executionProviders: ["wasm"]
        });
        state.predictor = createPredictor({
          ort: ort, session: session, config: config, tokenizer: tokenizer,
          modelVersion: state.modelVersion
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

  async function reloadModel() {
    state = { ready: false, error: null, predictor: null, modelVersion: null, configSource: null };
    loading = null;
    await ensureReady();
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
    if (msg.type === "FD_RELOAD_MODEL") {
      reloadModel()
        .then(function () {
          sendResponse({ ok: true, ready: state.ready, modelVersion: state.modelVersion, configSource: state.configSource });
        })
        .catch(function (e) { sendResponse({ ok: false, error: (e && e.message) || String(e) }); });
      return true;
    }
    if (msg.type === "FD_STATUS") {
      sendResponse({
        ok: true, ready: state.ready, error: state.error,
        modelVersion: state.modelVersion, configSource: state.configSource
      });
    }
  });

  // Pre-warm so the first real check is fast.
  ensureReady().catch(function () { /* error surfaces on first predict */ });
})();
