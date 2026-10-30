/**
 * tokenizer.js — MuRIL (BERT) WordPiece tokenizer, ported to JS.
 *
 * Mirrors the Hugging Face fast tokenizer used in training
 * (google/muril-base-cased, do_lower_case=false, strip_accents=false):
 *   clean control chars -> isolate punctuation / split on whitespace
 *   (with CJK spacing) -> greedy longest-match WordPiece over vocab.txt
 *   -> [CLS] ... [SEP], truncated to MAX_LEN (128, as in training).
 *
 * Special-token ids are read from the vocab itself ([PAD] 0, [UNK] 100,
 * [CLS] 104, [SEP] 105 for MuRIL) — never hard-coded.
 */
(function (global) {
  "use strict";

  var MAX_LEN = 128;
  var MAX_WORD_CHARS = 100;

  function isWhitespace(ch) {
    return ch === " " || ch === "\t" || ch === "\n" || ch === "\r" || /\s/.test(ch);
  }
  function isControl(ch) {
    if (ch === "\t" || ch === "\n" || ch === "\r") return false;
    var cp = ch.codePointAt(0);
    // Cc + Cf ranges (approximation of unicodedata.category in {Cc, Cf})
    return (cp >= 0x00 && cp <= 0x1F) || (cp >= 0x7F && cp <= 0x9F) ||
      cp === 0x200B || cp === 0x200E || cp === 0x200F ||
      (cp >= 0x202A && cp <= 0x202E) || (cp >= 0x2060 && cp <= 0x2064) ||
      cp === 0xFEFF;
  }
  function isPunctuation(ch) {
    var cp = ch.codePointAt(0);
    if ((cp >= 33 && cp <= 47) || (cp >= 58 && cp <= 64) ||
        (cp >= 91 && cp <= 96) || (cp >= 123 && cp <= 126)) return true;
    return /\p{P}/u.test(ch);
  }
  function isChineseChar(ch) {
    var cp = ch.codePointAt(0);
    return (cp >= 0x4E00 && cp <= 0x9FFF) || (cp >= 0x3400 && cp <= 0x4DBF) ||
      (cp >= 0x20000 && cp <= 0x2A6DF) || (cp >= 0x2A700 && cp <= 0x2B73F) ||
      (cp >= 0x2B740 && cp <= 0x2B81F) || (cp >= 0x2B820 && cp <= 0x2CEAF) ||
      (cp >= 0xF900 && cp <= 0xFAFF) || (cp >= 0x2F800 && cp <= 0x2FA1F);
  }

  function cleanText(text) {
    var out = "";
    for (var ch of text) {
      var cp = ch.codePointAt(0);
      if (cp === 0xFFFD || isControl(ch)) continue;
      out += isWhitespace(ch) ? " " : ch;
    }
    return out;
  }

  function basicTokenize(text) {
    var cleaned = cleanText(text);
    // CJK spacing
    var spaced = "";
    for (var ch of cleaned) spaced += isChineseChar(ch) ? " " + ch + " " : ch;
    var rawTokens = spaced.split(/\s+/).filter(function (w) { return w.length > 0; });
    var tokens = [];
    for (var tok of rawTokens) {
      var cur = "";
      for (var c of tok) {
        if (isPunctuation(c)) {
          if (cur) { tokens.push(cur); cur = ""; }
          tokens.push(c);
        } else {
          cur += c;
        }
      }
      if (cur) tokens.push(cur);
    }
    return tokens;
  }

  function createTokenizer(vocabObj) {
    // vocabObj: {token: id} dumped from the TRAINING tokenizer itself
    // (transformers get_vocab()). MuRIL's raw vocab.txt has loader
    // quirks (whitespace-suffixed duplicate entries whose handling
    // differs between loaders), so the extension ships the exact map
    // training used — parity by construction.
    var vocab = new Map();
    for (var tok of Object.keys(vocabObj)) vocab.set(tok, vocabObj[tok]);
    var UNK = vocab.get("[UNK]"), CLS = vocab.get("[CLS]"), SEP = vocab.get("[SEP]");

    function wordpiece(token) {
      var chars = [...token];
      if (chars.length > MAX_WORD_CHARS) return [UNK];
      var subTokens = [];
      var start = 0;
      var bad = false;
      while (start < chars.length) {
        var end = chars.length;
        var curStr = null;
        while (start < end) {
          var substr = chars.slice(start, end).join("");
          if (start > 0) substr = "##" + substr;
          if (vocab.has(substr)) { curStr = substr; break; }
          end -= 1;
        }
        if (curStr === null) { bad = true; break; }
        subTokens.push(vocab.get(curStr));
        start = end;
      }
      return bad ? [UNK] : subTokens;
    }

    function encode(text) {
      var ids = [CLS];
      var basics = basicTokenize(text);
      for (var tok of basics) {
        var pieces = wordpiece(tok);
        for (var id of pieces) ids.push(id);
      }
      ids.push(SEP);
      if (ids.length > MAX_LEN) {
        // HF truncation keeps the closing [SEP] as the final token.
        ids = ids.slice(0, MAX_LEN - 1);
        ids.push(SEP);
      }
      var mask = ids.map(function () { return 1; });
      return { input_ids: ids, attention_mask: mask };
    }

    return { encode: encode, vocabSize: vocab.size, ids: { unk: UNK, cls: CLS, sep: SEP } };
  }

  var api = { createTokenizer: createTokenizer, MAX_LEN: MAX_LEN };
  global.FraudTokenizer = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})(typeof window !== "undefined" ? window : globalThis);
