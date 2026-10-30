/**
 * features.js — STRUCTURED FEATURES, ported 1:1 from training_v3/train.py
 * (extract_struct_features). Computed from the RAW message (only NFC +
 * native-digit translation — NO masking, NO Indic normalisation), exactly
 * as in training. Feature order matches STRUCT_FEATURE_NAMES in training;
 * the shipped head config relies on this order.
 */
(function (global) {
  "use strict";

  var SHORTENERS = [
    "bit.ly", "tinyurl", "t.co", "cutt.ly", "rb.gy", "shorturl",
    "goo.gl", "ow.ly", "is.gd", "rebrand.ly", "short.io"
  ];
  var OTP_WORDS = /otp|one[-\s]?time|verification code|security code|pin\b|ఓటిపి|ఒటిపి|ओटीपी|पिन/g;
  var DEBIT_WORDS = /debit|debited|withdraw|ఖర్చు|డెబిట్|कट|डेबिट|निकाल/;
  var CREDIT_WORDS = /credit|credited|deposit|జమ|క్రెడిట్|जमा|क्रेडिट/;
  var BLOCK_WORDS = /block|suspend|freeze|deactivat|బ్లాక్|నిలిపి|ब्लॉक|बंद हो|रोक/;
  var NO_SHARE_WORDS = /do not share|don't share|never share|ఎవరితోనూ పంచుకో|పంచుకోవద్దు|साझा न करें|मत बत|share cheyoddhu|share cheyak/;
  var NEVER_ASK_WORDS = /never ask|never asks|ఎప్పుడూ అడగ|कभी नहीं मांग|कभी नहीं पूछ/;
  var KYC_WORDS = /kyc|केवाईसी|కేవైసీ/;
  var PRIZE_WORDS = /won|winner|prize|lottery|jackpot|గెలుచు|బహుమతి|जीत|इनाम|लॉटरी/;
  var CLICK_WORDS = /click|tap here|link|లింక్|లింకు|क्लिक|लिंक/;

  function codePointLength(s) { return [...s].length; }

  function extractStructFeatures(text, translateNativeDigits) {
    if (typeof text !== "string") text = text == null ? "" : String(text);
    var t = translateNativeDigits(text.normalize("NFC"));
    var low = t.toLowerCase();
    var nChars = Math.max(codePointLength(t), 1);
    var words = t.split(/\s+/).filter(function (w) { return w.length > 0; });
    var digitRuns = t.match(/\d+/g) || [];
    var nDigits = digitRuns.reduce(function (a, r) { return a + r.length; }, 0);
    var urls = low.match(/https?:\/\/\S+|www\.\S+/g) || [];
    var urlIsShort = urls.some(function (u) {
      return SHORTENERS.some(function (s) { return u.indexOf(s) !== -1; });
    }) ? 1 : 0;
    var urlHasIp = urls.some(function (u) { return /\d{1,3}(\.\d{1,3}){3}/.test(u); }) ? 1 : 0;
    var amounts = [];
    digitRuns.forEach(function (r) {
      var v = parseFloat(r);
      if (!isNaN(v)) amounts.push(v);
    });
    var commaRe = /\d{1,3}(?:,\d{2,3})+(?:\.\d+)?/g, cm;
    while ((cm = commaRe.exec(t)) !== null) {
      var cv = parseFloat(cm[0].replace(/,/g, ""));
      if (!isNaN(cv)) amounts.push(cv);
    }
    var maxAmount = amounts.length ? Math.max.apply(null, amounts) : 0.0;
    var mentionsOtp = OTP_WORDS.test(low) ? 1 : 0;
    OTP_WORDS.lastIndex = 0;
    var otpNear = 0;
    var m;
    var otpRe = new RegExp(OTP_WORDS.source, "g");
    while ((m = otpRe.exec(low)) !== null) {
      var start = Math.max(0, m.index - 40);
      var window = low.slice(start, m.index + m[0].length + 40);
      if (/\d{4,8}/.test(window)) { otpNear = 1; break; }
      if (m[0].length === 0) otpRe.lastIndex++;
    }
    var otpRequested = (mentionsOtp &&
      /share|send|reply|enter|provide|tell|పంచుకోండి|भेज|बता/.test(low) &&
      !NO_SHARE_WORDS.test(low)) ? 1 : 0;
    var latin = t.match(/[A-Za-z]/g) || [];
    var caps = latin.filter(function (c) { return c >= "A" && c <= "Z"; }).length;
    var maxRun = digitRuns.reduce(function (a, r) { return Math.max(a, r.length); }, 0);

    return [
      nChars,
      words.length,
      nDigits / nChars,
      digitRuns.length,
      maxRun,
      digitRuns.some(function (r) { return r.length >= 10; }) ? 1 : 0,
      /(\+91[\s-]?|0)?[6-9]\d{9}\b/.test(t) ? 1 : 0,
      urls.length ? 1 : 0,
      urls.length,
      urlIsShort,
      urlHasIp,
      /[\w.+-]+@[\w-]+(\.[\w-]+)+/.test(t) ? 1 : 0,
      /\b[\w.\-]{2,}@[a-zA-Z]{2,}\b/.test(t) ? 1 : 0,
      /₹|\brs\.?\b|inr|రూ|रु/.test(low) ? 1 : 0,
      Math.log10(1.0 + maxAmount),
      maxAmount > 0 ? 1 : 0,
      mentionsOtp,
      otpNear,
      otpRequested,
      NO_SHARE_WORDS.test(low) ? 1 : 0,
      NEVER_ASK_WORDS.test(low) ? 1 : 0,
      DEBIT_WORDS.test(low) ? 1 : 0,
      CREDIT_WORDS.test(low) ? 1 : 0,
      BLOCK_WORDS.test(low) ? 1 : 0,
      KYC_WORDS.test(low) ? 1 : 0,
      PRIZE_WORDS.test(low) ? 1 : 0,
      CLICK_WORDS.test(low) ? 1 : 0,
      t.split("!").length - 1,
      t.split("?").length - 1,
      latin.length ? caps / latin.length : 0.0
    ];
  }

  var FEATURE_NAMES = [
    "n_chars", "n_words", "digit_ratio", "n_digit_runs", "max_run_len",
    "has_long_digit_run", "has_phone", "has_url", "n_urls",
    "url_is_shortener", "url_has_ip", "has_email", "has_upi",
    "has_currency", "amount_log10", "has_amount",
    "mentions_otp", "otp_word_near_digits", "otp_is_requested",
    "mentions_do_not_share", "mentions_never_ask",
    "mentions_debit", "mentions_credit", "mentions_blocked",
    "mentions_kyc", "mentions_prize", "mentions_click",
    "exclaim_count", "question_count", "caps_ratio"
  ];

  var api = { extractStructFeatures: extractStructFeatures, FEATURE_NAMES: FEATURE_NAMES };
  global.FraudFeatures = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})(typeof window !== "undefined" ? window : globalThis);
