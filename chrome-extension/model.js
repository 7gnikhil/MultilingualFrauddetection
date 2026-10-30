/**
 * model.js — content-side front door for prediction and feedback.
 *
 * The trained model runs in the offscreen host: structured features +
 * anonymize + MuRIL ONNX + a per-language logistic head. This file only
 * relays requests and translates probabilities into display states.
 *
 * There is deliberately NO keyword/demo fallback. If the trained model
 * cannot run, predict() throws a MODEL UNAVAILABLE error and the panel
 * says so; it never presents a heuristic guess as a model verdict.
 */
(function (global) {
  "use strict";

  var DEFAULT_UNCERTAINTY_MARGIN = 0.08;

  // Feature-name -> human phrase, for the "why" line.
  var FEATURE_REASONS = [
    ["otp_is_requested", "asks you to share an OTP / code"],
    ["url_is_shortener", "shortened link"],
    ["url_has_ip", "link hides an IP address"],
    ["mentions_prize", "prize / lottery lure"],
    ["mentions_blocked", "threatens account block / suspension"],
    ["mentions_kyc", "KYC pressure"],
    ["has_url", "contains a link"],
    ["otp_word_near_digits", "OTP code in the message"],
    ["has_long_digit_run", "long account-like number"],
    ["mentions_click", "pushes you to click / tap a link"]
  ];

  function reasonsFor(rawText) {
    try {
      var names = FraudFeatures.FEATURE_NAMES;
      var vals = FraudFeatures.extractStructFeatures(
        rawText, FraudPreprocess.translateNativeDigits);
      var out = [];
      FEATURE_REASONS.forEach(function (pair) {
        var idx = names.indexOf(pair[0]);
        if (idx >= 0 && vals[idx]) out.push(pair[1]);
      });
      return out.slice(0, 5);
    } catch (e) { return []; }
  }

  function normalizeMargin(margin) {
    return typeof margin === "number" && Number.isFinite(margin) && margin >= 0 && margin <= 0.49
      ? margin : DEFAULT_UNCERTAINTY_MARGIN;
  }

  /**
   * Display state for a fraud probability. A probability inside the
   * per-language uncertainty band around the decision threshold is not
   * presented as a confident SAFE or FRAUD answer.
   */
  function verdictFrom(probability, threshold, uncertaintyMargin) {
    var margin = normalizeMargin(uncertaintyMargin);
    if (Math.abs(probability - threshold) <= margin) return "uncertain";
    return probability > threshold ? "fraud" : "safe";
  }

  function sendToBackground(message, timeoutMs) {
    return new Promise(function (resolve, reject) {
      var timer = setTimeout(function () {
        reject(new Error("Model host timed out (first load can take ~30s — try again)"));
      }, timeoutMs || 180000);
      try {
        chrome.runtime.sendMessage(message, function (resp) {
          clearTimeout(timer);
          if (chrome.runtime.lastError) {
            reject(new Error(chrome.runtime.lastError.message));
          } else if (!resp) {
            reject(new Error("No response from model host"));
          } else if (!resp.ok) {
            reject(new Error(resp.error || "Model host error"));
          } else {
            resolve(resp);
          }
        });
      } catch (e) {
        clearTimeout(timer);
        reject(e);
      }
    });
  }

  /**
   * predict(rawText, langInfo) — langInfo from FraudDetectLanguage.
   * -> { label, confidence, probability, threshold, uncertaintyMargin,
   *      modelVersion, reasons }
   * label is "fraud", "safe", or "uncertain".
   * Throws an error beginning "MODEL UNAVAILABLE" if the real model
   * cannot produce a result.
   */
  async function predict(rawText, langInfo) {
    var headName = (langInfo && langInfo.head) || "English";
    var resp;
    try {
      resp = await sendToBackground(
        { type: "FD_PREDICT_REQUEST", text: rawText, head: headName });
    } catch (e) {
      throw new Error("MODEL UNAVAILABLE: " + ((e && e.message) || String(e)));
    }
    var res = resp.result;
    if (!res || typeof res.probability !== "number" || typeof res.threshold !== "number") {
      throw new Error("MODEL UNAVAILABLE: The model host returned an invalid result.");
    }
    var margin = normalizeMargin(res.uncertaintyMargin);
    var label = verdictFrom(res.probability, res.threshold, margin);
    return {
      label: label,
      probability: res.probability,
      confidence: label === "safe" ? 1 - res.probability : res.probability,
      threshold: res.threshold,
      uncertaintyMargin: margin,
      modelVersion: res.modelVersion || null,
      demo: false,
      reasons: label === "safe" ? [] : reasonsFor(rawText)
    };
  }

  /**
   * learn(rawText, langInfo, label, prediction) — the user corrected a
   * verdict. label: 1 = actually fraud, 0 = actually safe. prediction
   * is the result that was on screen, used as the feedback event's
   * predicted label/probability when sharing is explicitly enabled.
   * The background worker decides whether an anonymized event may be
   * queued; this function never uploads anything itself.
   */
  async function learn(rawText, langInfo, label, prediction) {
    var headName = (langInfo && langInfo.head) || "English";
    var resp = await sendToBackground({
      type: "FD_FEEDBACK_REQUEST",
      text: rawText,
      head: headName,
      label: label ? 1 : 0,
      predictedLabel: prediction ? prediction.label : undefined,
      predictedProbability: prediction ? prediction.probability : undefined,
      threshold: prediction ? prediction.threshold : undefined,
      modelVersion: prediction ? prediction.modelVersion : undefined
    });
    return resp.result;
  }

  async function learningState() {
    return sendToBackground({ type: "FD_LEARNING_STATE_REQUEST" }, 15000);
  }

  global.FraudModel = {
    predict: predict,
    learn: learn,
    learningState: learningState,
    verdictFrom: verdictFrom
  };
})(typeof window !== "undefined" ? window : globalThis);
