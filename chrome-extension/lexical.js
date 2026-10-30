/**
 * lexical.js — Arm A of the v4 model: hashed lexical logistic regression,
 * a faithful JavaScript port of the exact scikit-learn recipe used in
 * training (ml-final/train_final_v4.py):
 *
 *   HashingVectorizer(analyzer="word",   ngram_range=(1,2), n_features=2^14,
 *                     alternate_sign=False, norm="l2", lowercase=True)
 * + HashingVectorizer(analyzer="char_wb", ngram_range=(3,5), n_features=2^15,
 *                     alternate_sign=False, norm="l2", lowercase=True)
 *   -> one LogisticRegression over the concatenated blocks.
 *
 * Fidelity notes (all verified against sklearn 1.9.1 — see
 * ml-final/BUILD_REPORT.md):
 *  - Hash bin = abs(murmurhash3_32(utf8(ngram), seed=0)) % n_features.
 *    (sklearn's signed hash, absolute value — NOT the unsigned modulo.)
 *  - The word analyser's token pattern (?u)\b\w\w+\b uses CPython `re`
 *    semantics, where \w = Unicode categories L* + N* + '_'. Combining
 *    marks (Mn/Mc) are NOT word characters, so Indic vowel signs and the
 *    virama BREAK word tokens exactly as they do in Python ("आपके" ->
 *    token "आपक"). This port reproduces that behaviour deliberately —
 *    it is what the model was trained on.
 *  - char_wb: the text is split on whitespace; each word is padded with
 *    one space on both sides (" word ") and char n-grams of length 3..5
 *    are taken inside it. N-grams are code-point slices.
 *  - Each block is L2-normalised separately (as in training, where the
 *    two vectorizers were hstacked after individual normalisation).
 *  - Input must be the anonymized text (same string Python trains on).
 */
(function (global) {
  "use strict";

  var textEncoder = (typeof TextEncoder !== "undefined") ? new TextEncoder() : null;

  function utf8Bytes(str) {
    if (textEncoder) return textEncoder.encode(str);
    // Fallback UTF-8 encoder (older engines without TextEncoder).
    var out = [];
    for (var i = 0; i < str.length; i++) {
      var cp = str.codePointAt(i);
      if (cp > 0xffff) i++; // skip low surrogate
      if (cp < 0x80) out.push(cp);
      else if (cp < 0x800) out.push(0xc0 | (cp >> 6), 0x80 | (cp & 0x3f));
      else if (cp < 0x10000) out.push(0xe0 | (cp >> 12), 0x80 | ((cp >> 6) & 0x3f), 0x80 | (cp & 0x3f));
      else out.push(0xf0 | (cp >> 18), 0x80 | ((cp >> 12) & 0x3f), 0x80 | ((cp >> 6) & 0x3f), 0x80 | (cp & 0x3f));
    }
    return out;
  }

  /** MurmurHash3 x86_32 over bytes; returns an unsigned 32-bit integer. */
  function murmur3_32(bytes, seed) {
    var h = (seed || 0) | 0;
    var c1 = 0xcc9e2d51, c2 = 0x1b873593;
    var len = bytes.length;
    var nblocks = (len / 4) | 0;
    for (var i = 0; i < nblocks; i++) {
      var j = i * 4;
      var k = bytes[j] | (bytes[j + 1] << 8) | (bytes[j + 2] << 16) | (bytes[j + 3] << 24);
      k = Math.imul(k, c1);
      k = (k << 15) | (k >>> 17);
      k = Math.imul(k, c2);
      h ^= k;
      h = (h << 13) | (h >>> 19);
      h = (Math.imul(h, 5) + 0xe6546b64) | 0;
    }
    var tail = len & 3;
    var t = nblocks * 4;
    var k1 = 0;
    if (tail === 3) k1 ^= bytes[t + 2] << 16;
    if (tail >= 2) k1 ^= bytes[t + 1] << 8;
    if (tail >= 1) {
      k1 ^= bytes[t];
      k1 = Math.imul(k1, c1);
      k1 = (k1 << 15) | (k1 >>> 17);
      k1 = Math.imul(k1, c2);
      h ^= k1;
    }
    h ^= len;
    h ^= h >>> 16;
    h = Math.imul(h, 0x85ebca6b);
    h ^= h >>> 13;
    h = Math.imul(h, 0xc2b2ae35);
    h ^= h >>> 16;
    return h >>> 0;
  }

  /** sklearn bin: abs(signed murmur3) % n_features. */
  function hashBin(str, nFeatures) {
    var u = murmur3_32(utf8Bytes(str), 0);
    var signed = u > 0x7fffffff ? u - 4294967296 : u;
    return Math.abs(signed) % nFeatures;
  }

  // CPython re \w: categories L*, N* and '_'. Marks are NOT included.
  var WORD_RUN = /[\p{L}\p{N}_]+/gu;

  function wordTokens(text) {
    var lower = String(text).toLowerCase();
    var runs = lower.match(WORD_RUN) || [];
    var toks = [];
    for (var i = 0; i < runs.length; i++) {
      // \w\w+ : at least 2 code points.
      if (Array.from(runs[i]).length >= 2) toks.push(runs[i]);
    }
    return toks;
  }

  function bump(map, bin) { map.set(bin, (map.get(bin) || 0) + 1); }

  function wordBlockCounts(text, nFeatures) {
    var toks = wordTokens(text);
    var counts = new Map();
    for (var i = 0; i < toks.length; i++) {
      bump(counts, hashBin(toks[i], nFeatures));
      if (i + 1 < toks.length) bump(counts, hashBin(toks[i] + " " + toks[i + 1], nFeatures));
    }
    return counts;
  }

  function charBlockCounts(text, nFeatures) {
    var lower = String(text).toLowerCase();
    var words = lower.split(/\s+/u).filter(function (w) { return w.length > 0; });
    var counts = new Map();
    for (var wi = 0; wi < words.length; wi++) {
      var cps = Array.from(" " + words[wi] + " ");
      for (var n = 3; n <= 5; n++) {
        for (var i = 0; i + n <= cps.length; i++) {
          bump(counts, hashBin(cps.slice(i, i + n).join(""), nFeatures));
        }
      }
    }
    return counts;
  }

  function l2Normalize(counts) {
    var norm2 = 0;
    counts.forEach(function (v) { norm2 += v * v; });
    var out = new Map();
    if (norm2 <= 0) return out;
    var norm = Math.sqrt(norm2);
    counts.forEach(function (v, k) { out.set(k, v / norm); });
    return out;
  }

  /** Sparse normalised features: { word: Map<bin,val>, char: Map<bin,val> }. */
  function features(cleanText, wordNFeatures, charNFeatures) {
    return {
      word: l2Normalize(wordBlockCounts(cleanText, wordNFeatures || 16384)),
      char: l2Normalize(charBlockCounts(cleanText, charNFeatures || 32768))
    };
  }

  var coefCache = new WeakMap();
  function coefMaps(hashCfg) {
    var maps = coefCache.get(hashCfg);
    if (maps) return maps;
    function toMap(pairs) {
      var m = new Map();
      (pairs || []).forEach(function (p) { m.set(p[0], p[1]); });
      return m;
    }
    maps = { word: toMap(hashCfg.word_coef), char: toMap(hashCfg.char_coef) };
    coefCache.set(hashCfg, maps);
    return maps;
  }

  function sigmoid(z) { return 1 / (1 + Math.exp(-z)); }

  /**
   * probability(cleanText, hashCfg) — Arm A fraud probability.
   * hashCfg: { word_n_features, char_n_features, intercept,
   *            word_coef: [[bin, value], ...], char_coef: [[bin, value], ...] }
   */
  function probability(cleanText, hashCfg) {
    var f = features(cleanText, hashCfg.word_n_features, hashCfg.char_n_features);
    var maps = coefMaps(hashCfg);
    var z = hashCfg.intercept;
    f.word.forEach(function (v, bin) { var c = maps.word.get(bin); if (c) z += c * v; });
    f.char.forEach(function (v, bin) { var c = maps.char.get(bin); if (c) z += c * v; });
    return sigmoid(z);
  }

  var api = {
    probability: probability,
    features: features,
    wordTokens: wordTokens,
    hashBin: hashBin,
    _murmur3_32: murmur3_32,
    _utf8Bytes: utf8Bytes
  };
  global.FraudLexical = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})(typeof window !== "undefined" ? window : globalThis);
