/**
 * langdetect.js — language routing for the per-language heads.
 *
 * Routing follows the training convention (train.py detect_language):
 * any Telugu script -> Telugu head; any Devanagari -> Hindi head;
 * otherwise English — extended with Romanized detection, because the
 * training data folded Romanized Telugu into Telugu and Hinglish into
 * Hindi. Codes: en | hi | hi-en | te | te-en. `head` is the trained
 * language name the message routes to.
 */
(function (global) {
  "use strict";

  var ROMANIZED_TELUGU = [
    "meeru", "nenu", "naku", "meeku", "cheppu", "chey", "undi", "ledu",
    "kada", "inka", "eppudu", "ekkada", "ela", "enti", "avunu", "kadhu",
    "dabbulu", "pampandi", "cheyyandi", "vastundi", "potundi", "garu"
  ];
  var ROMANIZED_HINDI = [
    "aap", "aapka", "aapko", "hai", "hain", "mera", "meri", "kya", "nahi",
    "kar", "karo", "karein", "bhejo", "bhej", "paisa", "paise", "abhi",
    "turant", "band", "ho jayega", "jayega", "karna", "chahiye", "kyun"
  ];

  function detectLanguage(text) {
    var t = String(text || "");
    if (/[ఀ-౿]/.test(t)) return { code: "te", label: "Telugu", head: "Telugu" };
    if (/[ऀ-ॿ]/.test(t)) return { code: "hi", label: "Hindi", head: "Hindi" };

    var lower = " " + t.toLowerCase().replace(/[^a-z\s]/g, " ") + " ";
    function hits(list) {
      return list.filter(function (w) { return lower.indexOf(" " + w + " ") !== -1; }).length;
    }
    var teHits = hits(ROMANIZED_TELUGU);
    var hiHits = hits(ROMANIZED_HINDI);
    if (teHits >= 2 && teHits >= hiHits) return { code: "te-en", label: "Telugu (Romanized)", head: "Telugu" };
    if (hiHits >= 2) return { code: "hi-en", label: "Hindi (Romanized / Hinglish)", head: "Hindi" };

    return { code: "en", label: "English", head: "English" };
  }

  global.FraudDetectLanguage = detectLanguage;
  if (typeof module !== "undefined" && module.exports) module.exports = detectLanguage;
})(typeof window !== "undefined" ? window : globalThis);
