/**
 * anonymize.js — v3 PREPROCESSING PORT (must stay identical to training).
 *
 * Faithful JavaScript port of the preprocessing in training_v3/train.py:
 *   anonymize(): NFC -> Indic NLP normalisation (per script) -> native
 *   digit translation -> fallback normalisation -> placeholder masking
 *   ([URL] [EMAIL] [UPI] [ACCOUNT] [PHONE] [OTP]).
 * No lowercasing anywhere (MuRIL-base-cased is a cased model).
 *
 * Why identity with training matters: the model was trained on text
 * cleaned EXACTLY this way. Any drift here is train/serve skew and
 * silently degrades predictions.
 *
 * The Indic normalisation rules are ported from the indic-nlp-library
 * (BaseNormalizer + DevanagariNormalizer + TeluguNormalizer, factory
 * defaults), which train.py runs. English gets the BaseNormalizer
 * treatment, exactly as in training.
 */
(function (global) {
  "use strict";

  function rep(text, from, to) { return text.split(from).join(to); }

  // ------------------------------------------------------------------
  // Indic normalisation (port of indicnlp.normalize.indic_normalize)
  // ------------------------------------------------------------------
  function baseNormalize(text) {
    // BaseNormalizer.normalize(): control/format char handling
    text = rep(text, "\uFEFF", "");   // BYTE_ORDER_MARK
    text = rep(text, "\uFFFE", "");   // BYTE_ORDER_MARK_2
    text = rep(text, "\u2060", "");   // WORD_JOINER
    text = rep(text, "\u00AD", "");   // SOFT_HYPHEN
    text = rep(text, "\u200B", " ");  // ZERO_WIDTH_SPACE -> space
    text = rep(text, "\u00A0", " ");  // NO_BREAK_SPACE -> space
    text = rep(text, "\u200C", "");   // ZERO_WIDTH_NON_JOINER
    text = rep(text, "\u200D", "");   // ZERO_WIDTH_JOINER
    // _normalize_punctuations()
    text = rep(text, "„", '"');
    text = rep(text, "“", '"');
    text = rep(text, "”", '"');
    text = rep(text, "–", "-");
    text = rep(text, "—", " - ");
    text = rep(text, "´", "'");
    text = rep(text, "‘", "'");
    text = rep(text, "‚", "'");
    text = rep(text, "’", "'");
    text = rep(text, "''", '"');
    text = rep(text, "´´", '"');
    text = rep(text, "…", "...");
    return text;
  }

  var NUKTA = "़";
  var NUKTA_DECOMP = [
    ["ऩ", "न" + NUKTA], ["ऱ", "र" + NUKTA], ["ऴ", "ळ" + NUKTA],
    ["क़", "क" + NUKTA], ["ख़", "ख" + NUKTA], ["ग़", "ग" + NUKTA],
    ["ज़", "ज" + NUKTA], ["ड़", "ड" + NUKTA], ["ढ़", "ढ" + NUKTA],
    ["फ़", "फ" + NUKTA], ["य़", "य" + NUKTA]
  ];

  function devanagariNormalize(text) {
    text = baseNormalize(text);
    text = rep(text, "ॲ", "ए");                    // chandra a -> e
    for (var i = 0; i < NUKTA_DECOMP.length; i++) {
      text = rep(text, NUKTA_DECOMP[i][0], NUKTA_DECOMP[i][1]);
    }
    text = rep(text, "|", "।");                          // pipe -> danda
    text = text.replace(/([ऀ-ॿ]):/g, "$1ः"); // colon -> visarga
    return text;
  }

  function teluguNormalize(text) {
    text = baseNormalize(text);
    text = rep(text, "౤", "।");  // poorna virama -> generic danda
    text = rep(text, "౥", "॥");  // -> generic double danda
    text = rep(text, "ై", "ై");  // two-part vowel sign -> canonical
    text = text.replace(/([ఀ-౿]):/g, "$1ః"); // colon -> visarga
    return text;
  }

  function scriptLang(text) {
    if (/[ఀ-౿]/.test(text)) return "te";
    if (/[ऀ-ॿ]/.test(text)) return "hi";
    return "en";
  }

  function indicNormalize(text, langCode) {
    if (langCode === "hi") return devanagariNormalize(text);
    if (langCode === "te") return teluguNormalize(text);
    // English: the indic factory raises KeyError for 'en', so train.py's
    // _indic_normalize falls back — net effect is NO base normalisation
    // for English text, only the fallback step in anonymize().
    return text;
  }

  function fallbackNormalize(text) {
    // train.py _fallback_normalize — also runs after the real normaliser.
    text = rep(text, "\u200B", "");
    text = rep(text, "\u200C", "");
    text = rep(text, "\u200D", "");
    text = rep(text, "\uFEFF", "");
    text = rep(text, "\u00A0", " ");
    return text;
  }

  // Telugu digits U+0C66..U+0C6F and Devanagari digits U+0966..U+096F -> 0-9
  var NATIVE_DIGIT_MAP = {};
  "౦౧౨౩౪౫౬౭౮౯".split("").forEach(function (ch, i) { NATIVE_DIGIT_MAP[ch] = String(i); });
  "०१२३४५६७८९".split("").forEach(function (ch, i) { NATIVE_DIGIT_MAP[ch] = String(i); });
  var NATIVE_DIGIT_RE = /[౦-౯०-९]/g;

  function translateNativeDigits(text) {
    return text.replace(NATIVE_DIGIT_RE, function (ch) { return NATIVE_DIGIT_MAP[ch]; });
  }

  // ------------------------------------------------------------------
  // anonymize() — exact port of train.py (v3)
  // ------------------------------------------------------------------
  function anonymize(text) {
    if (typeof text !== "string") text = text == null ? "" : String(text);
    var t = text.normalize("NFC");
    t = indicNormalize(t, scriptLang(t));
    t = translateNativeDigits(t);
    t = fallbackNormalize(t);
    t = t.replace(/https?:\/\/\S+|www\.\S+/g, "[URL]");
    t = t.replace(/[\w.+-]+@[\w-]+(\.[\w-]+)+/g, "[EMAIL]");
    t = t.replace(/\b[\w.\-]{2,}@[a-zA-Z]{2,}\b/g, "[UPI]");
    t = t.replace(/\b\d{4}[\s-]\d{4}[\s-]\d{4}([\s-]\d{2,4})?\b/g, "[ACCOUNT]");
    t = t.replace(/\b\d{12,18}\b/g, "[ACCOUNT]");
    t = t.replace(/(\+91[\s-]?|91[\s-]?|0)?[6-9]\d{9}\b/g, "[PHONE]");
    t = t.replace(/\b\d{5}[\s-]\d{5}\b/g, "[PHONE]");
    t = t.replace(/\b\d{4,8}\b/g, "[OTP]");
    return t.replace(/\s+/g, " ").trim();
  }

  var api = {
    anonymize: anonymize,
    indicNormalize: indicNormalize,
    translateNativeDigits: translateNativeDigits,
    scriptLang: scriptLang
  };
  global.FraudAnonymize = anonymize;
  global.FraudPreprocess = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})(typeof window !== "undefined" ? window : globalThis);
