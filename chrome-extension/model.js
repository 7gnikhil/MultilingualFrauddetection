/**
 * model.js — content-side front door for prediction.
 *
 * Real path: forwards the RAW message to the offscreen model host
 * (offscreen.js), where the trained v3 pipeline runs: structured
 * features + anonymize + MuRIL ONNX + per-language logistic head.
 * The verdict uses each language's tuned threshold (not 0.5).
 *
 * Fallback: if the model host cannot run (e.g. very old Chrome), a
 * small keyword stub answers instead and the result is BADGED AS DEMO
 * (demo: true) — it never pretends to be the trained model.
 */
(function (global) {
  "use strict";

  var LIKELY_BAND = 0.15; // within this below the threshold -> "likely"

  // Feature-name -> human phrase, for the "why" line (real model only).
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

  function verdictFrom(probability, threshold) {
    if (probability >= threshold) return "fraud";
    if (probability >= threshold - LIKELY_BAND) return "likely";
    return "safe";
  }

  function askModelHost(rawText, headName) {
    return new Promise(function (resolve, reject) {
      var timer = setTimeout(function () {
        reject(new Error("Model host timed out (first load can take ~30s — try again)"));
      }, 180000);
      try {
        chrome.runtime.sendMessage(
          { type: "FD_PREDICT_REQUEST", text: rawText, head: headName },
          function (resp) {
            clearTimeout(timer);
            if (chrome.runtime.lastError) {
              reject(new Error(chrome.runtime.lastError.message));
            } else if (!resp) {
              reject(new Error("No response from model host"));
            } else if (!resp.ok) {
              reject(new Error(resp.error || "Model host error"));
            } else {
              resolve(resp.result);
            }
          }
        );
      } catch (e) {
        clearTimeout(timer);
        reject(e);
      }
    });
  }

  // --- Demo-only heuristic stub (fallback; always badged DEMO) -----------
  var FRAUD_CUES = [
    "blocked", "suspended", "expire", "immediately", "urgent", "last chance",
    "within 24 hours", "deactivat", "legal action", "arrest", "court",
    "won", "winner", "lottery", "prize", "cashback", "free gift", "reward",
    "double your money", "guaranteed profit", "investment return",
    "share your otp", "enter your otp", "send otp", "share otp", "your pin",
    "cvv", "verify your account", "kyc update", "update your kyc",
    "click the link", "click here", "download this app", "anydesk",
    "teamviewer", "screen share", "pay a small fee", "processing fee",
    "dabbulu", "gelicharu", "bhejo", "turant", "band ho jayega", "inaam"
  ];

  function stubPredict(anonymized) {
    var t = " " + anonymized.toLowerCase() + " ";
    var hits = FRAUD_CUES.filter(function (cue) { return t.indexOf(cue) !== -1; });
    var score = Math.min(0.08 + hits.length * 0.18, 0.96);
    if (t.indexOf("[otp]") !== -1 && hits.length) score = Math.min(score + 0.12, 0.97);
    if (t.indexOf("[url]") !== -1 && hits.length) score = Math.min(score + 0.08, 0.97);
    if (!hits.length) score = 0.08;
    var label = "safe";
    if (score >= 0.62) label = "fraud";
    else if (score >= 0.34) label = "likely";
    return {
      label: label,
      probability: score,
      confidence: label === "safe" ? 1 - score : score,
      threshold: 0.5,
      demo: true,
      reasons: hits.slice(0, 5)
    };
  }

  /**
   * predict(rawText, langInfo) — langInfo from FraudDetectLanguage.
   * -> { label, confidence, probability, threshold, demo, reasons }
   */
  async function predict(rawText, langInfo) {
    var headName = (langInfo && langInfo.head) || "English";
    try {
      var res = await askModelHost(rawText, headName);
      var label = verdictFrom(res.probability, res.threshold);
      return {
        label: label,
        probability: res.probability,
        confidence: label === "safe" ? 1 - res.probability : res.probability,
        threshold: res.threshold,
        demo: false,
        reasons: label === "safe" ? [] : reasonsFor(rawText)
      };
    } catch (e) {
      var anonymized = global.FraudAnonymize ? global.FraudAnonymize(rawText) : rawText;
      var stub = stubPredict(anonymized);
      stub.fallbackError = (e && e.message) || String(e);
      return stub;
    }
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
   * learn(rawText, langInfo, label) — the user corrected a verdict.
   * label: 1 = actually fraud, 0 = actually safe. The offscreen host
   * updates the language head's delta in real time and the correction
   * is logged for CSV export / future retraining.
   * -> { probability, threshold, corrections }
   */
  async function learn(rawText, langInfo, label) {
    var headName = (langInfo && langInfo.head) || "English";
    var resp = await sendToBackground(
      { type: "FD_FEEDBACK_REQUEST", text: rawText, head: headName, label: label ? 1 : 0 });
    return resp.result;
  }

  async function learningState() {
    var resp = await sendToBackground({ type: "FD_LEARNING_STATE_REQUEST" }, 15000);
    return resp;
  }

  global.FraudModel = {
    predict: predict,
    learn: learn,
    learningState: learningState,
    verdictFrom: verdictFrom
  };
})(typeof window !== "undefined" ? window : globalThis);
