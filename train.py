#!/usr/bin/env python3
"""
train.py — Multilingual fraud-message classifier v3 (Telugu / Hindi / English)
==============================================================================

WHAT CHANGED IN v3 (from the preprocessing research, Oct 2026)
  v2's probabilities clustered at 0.51-0.59 — near chance. The research
  pointed at four suspects in OUR pipeline. v3 fixes/tests each one:

  1. MASKING DELETED THE SIGNAL. Fake bank alerts copy real ones word for
     word; what differs is the link type, the number type, the digit
     patterns. v2 masked all of that away. v3 extracts STRUCTURED FEATURES
     from the raw message BEFORE masking (has_url, shortener?, phone type,
     digit-run stats, amount bucket, OTP-request context, "do not share"
     wording, debit/credit, ...) and feeds them to the head alongside the
     embedding. The text itself is still masked (privacy + consistency).
  2. NFC IS NOT ENOUGH FOR TELUGU/HINDI. v3 runs the Indic NLP Library
     normaliser (removes ZWJ/ZWNJ, canonicalises nukta and two-part vowel
     signs, standardises danda) before tokenisation. If the library is not
     installed, a built-in fallback removes the zero-width characters.
     (v2 actually KEPT ZWJ/ZWNJ on purpose — that was the wrong call.)
  3. THE HEAD WAS UNDERFIT. v2 used LogisticRegression(C=1.0) on raw,
     unscaled 768-d embeddings with class_weight="balanced" and no C
     search — logits shrink toward the intercept, so every probability
     huddles near the prior. v3 standardises features and SELECTS C by
     inner cross-validation per language/arm.
  4. NO BASELINE. v3 trains a word+char TF-IDF arm on the SAME grouped
     splits. If TF-IDF beats frozen MuRIL for a language, we learn the
     problem is the embedding pipeline — and TF-IDF becomes a candidate
     head for that language.

  Also: per-language thresholds can use a PRECISION FLOOR (Telugu: prefer
  thresholds with precision >= 0.75, to cut the ham false alarms), ROC-AUC
  and the spread (std) of out-of-fold probabilities are reported per arm —
  AUC >> 0.5 with clustered probs means a head/calibration problem;
  AUC ~ 0.5 means a representation problem. That one check tells us where
  we stand after this run.

  Kept from v2 (these were right): skeleton-grouped CV, nested threshold
  tuning, real-only F1 as the headline, synthetic rows weigh 0.5, no
  lowercasing (MuRIL is cased), no stemming/stopword removal.

RUN
  pip install -r requirements.txt
  python diagnose.py                  # Phase-0 audit (tokeniser, masking)
  python train.py                     # full run (CV arms + final + export)
  python train.py --demo "message"    # after a run; or use test_batch.py
  python test_batch.py test_messages.csv

First run downloads MuRIL (~900 MB). Embeddings are cached in outputs/.
NOTE: v3 heads may use structured features, so the Chrome extension is
NOT updated by this run (config.json says extension_ready: false). v3 is
the experiment that decides what the extension should become.
"""

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------
MODEL_NAME = "google/muril-base-cased"
MAX_LEN = 128
BATCH_SIZE = 32
N_SPLITS = 5
RANDOM_STATE = 42
LANGUAGES = ["English", "Hindi", "Telugu"]
SYNTHETIC_WEIGHT = 0.5
C_GRID = [0.01, 0.1, 1.0, 10.0, 100.0]
# Prefer thresholds whose validation precision clears the floor (fraud
# flagging that cries wolf on real bank alerts is the failure we are
# fixing). Falls back to plain best-F1 if no threshold clears the floor.
PRECISION_FLOOR = {"Telugu": 0.75}
OUT_DIR = Path("outputs")
PREPROCESS_VERSION = "v3-indicnorm+struct+cgrid"

# ==========================================================================
# STEP 2a — INDIC NORMALISATION (NEW IN v3)
# NFC alone leaves visually identical Telugu/Hindi words as different code
# sequences: ZWJ/ZWNJ from different keyboards, composed vs decomposed
# nukta, two-part vowel signs written two ways. The Indic NLP Library
# normaliser canonicalises all of that. Fallback keeps the essentials.
# ==========================================================================
_NORMALIZERS = {}


def _indic_normalize(text: str, lang_code: str) -> str:
    try:
        if lang_code not in _NORMALIZERS:
            from indicnlp.normalize.indic_normalize import (
                IndicNormalizerFactory,
            )

            _NORMALIZERS[lang_code] = IndicNormalizerFactory().get_normalizer(
                lang_code
            )
        return _NORMALIZERS[lang_code].normalize(text)
    except Exception:
        return _fallback_normalize(text)


def _fallback_normalize(text: str) -> str:
    # Zero-width joiner / non-joiner / space and BOM: invisible, typed
    # inconsistently across keyboards, and they change WordPiece splits.
    for ch in ("\u200b", "\u200c", "\u200d", "\ufeff"):
        text = text.replace(ch, "")
    # No-break space -> ordinary space.
    text = text.replace("\u00a0", " ")
    return text


def _script_lang(text: str) -> str:
    if re.search(r"[ఀ-౿]", text):
        return "te"
    if re.search(r"[ऀ-ॿ]", text):
        return "hi"
    return "en"


# ==========================================================================
# STEP 2b — CLEANING / ANONYMIZATION (v3)
# Same masking contract as v2 (placeholders, no lowercasing), but Indic
# normalisation now runs FIRST. The structured signal that masking removes
# is captured separately by extract_struct_features() below.
# !!! train.py and test_batch.py share this via import — keep it that way.
# ==========================================================================
_NATIVE_DIGITS = str.maketrans(
    "౦౧౨౩౪౫౬౭౮౯"  # Telugu digits 0-9
    "०१२३४५६७८९",  # Devanagari digits 0-9
    "0123456789" "0123456789",
)


def anonymize(text: str) -> str:
    """Normalise Indic encoding, then mask private numbers/links."""
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    text = unicodedata.normalize("NFC", text)
    text = _indic_normalize(text, _script_lang(text))
    text = text.translate(_NATIVE_DIGITS)
    text = _fallback_normalize(text)  # belt & braces if library was used
    text = re.sub(r"https?://\S+|www\.\S+", "[URL]", text)
    text = re.sub(r"[\w.+-]+@[\w-]+(\.[\w-]+)+", "[EMAIL]", text)
    text = re.sub(r"\b[\w.\-]{2,}@[a-zA-Z]{2,}\b", "[UPI]", text)
    text = re.sub(r"\b\d{4}[\s-]\d{4}[\s-]\d{4}([\s-]\d{2,4})?\b", "[ACCOUNT]", text)
    text = re.sub(r"\b\d{12,18}\b", "[ACCOUNT]", text)
    text = re.sub(r"(\+91[\s-]?|91[\s-]?|0)?[6-9]\d{9}\b", "[PHONE]", text)
    text = re.sub(r"\b\d{5}[\s-]\d{5}\b", "[PHONE]", text)
    text = re.sub(r"\b\d{4,8}\b", "[OTP]", text)
    # NOTE: no lowercasing — MuRIL-base-cased is a CASED model.
    return re.sub(r"\s+", " ", text).strip()


def skeleton(text: str) -> str:
    """Template-family name for grouped splitting (unchanged from v2)."""
    t = anonymize(text).lower()
    t = re.sub(r"\[[a-z]+\]", " ", t)
    t = re.sub(r"[^\w\s]", " ", t, flags=re.UNICODE)
    return re.sub(r"\s+", " ", t).strip()


def detect_language(text: str) -> str:
    if re.search(r"[ఀ-౿]", text):
        return "Telugu"
    if re.search(r"[ऀ-ॿ]", text):
        return "Hindi"
    return "English"


# ==========================================================================
# STEP 2c — STRUCTURED FEATURES (NEW IN v3)
# Extracted from the RAW message (digits normalised, nothing masked yet).
# These capture exactly what masking deletes: link/phone/digit behaviour
# and the wording contexts that separate a real bank alert from a copy.
# ==========================================================================
_SHORTENERS = (
    "bit.ly", "tinyurl", "t.co", "cutt.ly", "rb.gy", "shorturl",
    "goo.gl", "ow.ly", "is.gd", "rebrand.ly", "short.io",
)
_OTP_WORDS = (
    r"otp|one[-\s]?time|verification code|security code|pin\b|"
    r"ఓటిపి|ఒటిపి|ओटीपी|पिन"
)
_DEBIT_WORDS = r"debit|debited|withdraw|ఖర్చు|డెబిట్|कट|डेबिट|निकाल"
_CREDIT_WORDS = r"credit|credited|deposit|జమ|క్రెడిట్|जमा|क्रेडिट"
_BLOCK_WORDS = (
    r"block|suspend|freeze|deactivat|బ్లాక్|నిలిపి|ब्लॉक|बंद हो|रोक"
)
_NO_SHARE_WORDS = (
    r"do not share|don'?t share|never share|ఎవరితోనూ పంచుకో|"
    r"పంచుకోవద్దు|साझा न करें|मत बत|share cheyoddhu|share cheyak"
)
_NEVER_ASK_WORDS = (
    r"never ask|never asks|ఎప్పుడూ అడగ|कभी नहीं मांग|कभी नहीं पूछ"
)
_KYC_WORDS = r"kyc|केवाईसी|కేవైసీ"
_PRIZE_WORDS = r"won|winner|prize|lottery|jackpot|గెలుచు|బహుమతి|जीत|इनाम|लॉटरी"
_CLICK_WORDS = r"click|tap here|link|లింక్|లింకు|क्लिक|लिंक"

STRUCT_FEATURE_NAMES = [
    "n_chars", "n_words", "digit_ratio", "n_digit_runs", "max_run_len",
    "has_long_digit_run", "has_phone", "has_url", "n_urls",
    "url_is_shortener", "url_has_ip", "has_email", "has_upi",
    "has_currency", "amount_log10", "has_amount",
    "mentions_otp", "otp_word_near_digits", "otp_is_requested",
    "mentions_do_not_share", "mentions_never_ask",
    "mentions_debit", "mentions_credit", "mentions_blocked",
    "mentions_kyc", "mentions_prize", "mentions_click",
    "exclaim_count", "question_count", "caps_ratio",
]


def extract_struct_features(text: str) -> np.ndarray:
    """One row of structured features for one raw message."""
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    t = unicodedata.normalize("NFC", text).translate(_NATIVE_DIGITS)
    low = t.lower()
    n_chars = max(len(t), 1)
    words = t.split()
    digit_runs = re.findall(r"\d+", t)
    n_digits = sum(len(r) for r in digit_runs)
    urls = re.findall(r"https?://\S+|www\.\S+", low)
    url_is_short = float(any(s in u for u in urls for s in _SHORTENERS))
    url_has_ip = float(any(re.search(r"\d{1,3}(\.\d{1,3}){3}", u) for u in urls))
    amounts = []
    for r in digit_runs:
        try:
            amounts.append(float(r))
        except ValueError:
            pass
    # Comma-grouped amounts like 70,000 — catch before run-splitting logic.
    for m in re.finditer(r"\d{1,3}(?:,\d{2,3})+(?:\.\d+)?", t):
        try:
            amounts.append(float(m.group(0).replace(",", "")))
        except ValueError:
            pass
    max_amount = max(amounts) if amounts else 0.0
    mentions_otp = float(bool(re.search(_OTP_WORDS, low)))
    otp_near = 0.0
    for m in re.finditer(_OTP_WORDS, low):
        window = low[max(0, m.start() - 40): m.end() + 40]
        if re.search(r"\d{4,8}", window):
            otp_near = 1.0
            break
    # OTP *requested* (fraud pattern) vs OTP delivered with a do-not-share
    # warning (real-bank pattern) — the context, not the digits.
    otp_requested = float(
        mentions_otp
        and bool(re.search(
            r"share|send|reply|enter|provide|tell|పంచుకోండి|भेज|बता", low
        ))
        and not re.search(_NO_SHARE_WORDS, low)
    )
    latin = re.findall(r"[A-Za-z]", t)
    caps = sum(1 for c in latin if c.isupper())
    return np.array([
        float(n_chars),
        float(len(words)),
        n_digits / n_chars,
        float(len(digit_runs)),
        float(max((len(r) for r in digit_runs), default=0)),
        float(any(len(r) >= 10 for r in digit_runs)),
        float(bool(re.search(r"(\+91[\s-]?|0)?[6-9]\d{9}\b", t))),
        float(bool(urls)),
        float(len(urls)),
        url_is_short,
        url_has_ip,
        float(bool(re.search(r"[\w.+-]+@[\w-]+(\.[\w-]+)+", t))),
        float(bool(re.search(r"\b[\w.\-]{2,}@[a-zA-Z]{2,}\b", t))),
        float(bool(re.search(r"₹|\brs\.?\b|inr|రూ|रु", low))),
        float(np.log10(1.0 + max_amount)),
        float(max_amount > 0),
        mentions_otp,
        otp_near,
        otp_requested,
        float(bool(re.search(_NO_SHARE_WORDS, low))),
        float(bool(re.search(_NEVER_ASK_WORDS, low))),
        float(bool(re.search(_DEBIT_WORDS, low))),
        float(bool(re.search(_CREDIT_WORDS, low))),
        float(bool(re.search(_BLOCK_WORDS, low))),
        float(bool(re.search(_KYC_WORDS, low))),
        float(bool(re.search(_PRIZE_WORDS, low))),
        float(bool(re.search(_CLICK_WORDS, low))),
        float(t.count("!")),
        float(t.count("?")),
        (caps / len(latin)) if latin else 0.0,
    ], dtype=np.float32)


def struct_matrix(texts) -> np.ndarray:
    return np.vstack([extract_struct_features(t) for t in texts])


# ==========================================================================
# STEP 1 — LOAD DATA
# ==========================================================================
def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8")
    need = {"text", "label", "language", "is_synthetic"}
    missing = need - set(df.columns)
    if missing:
        raise SystemExit(f"CSV is missing columns: {missing}. Found: {list(df.columns)}")
    df = df.dropna(subset=["text", "label", "language"]).copy()
    df["label"] = df["label"].astype(int)
    df = df[df["language"].isin(LANGUAGES)]
    df["clean"] = df["text"].map(anonymize)
    df["skeleton"] = df["text"].map(skeleton)
    df = df[df["clean"].str.len() > 0]
    before = len(df)
    df = df.drop_duplicates(subset=["clean", "label"]).reset_index(drop=True)
    print(f"Loaded {before} rows -> {len(df)} after dedupe")
    print(df.groupby(["language", "label"]).size().unstack(fill_value=0))
    return df


# ==========================================================================
# STEP 4 — FROZEN MuRIL EMBEDDINGS (cached) — unchanged from v2 except the
# cache key now includes the preprocessing version, so v2/v3 embeddings
# can never be mixed up.
# ==========================================================================
def get_embeddings(texts, out_dir: Path) -> np.ndarray:
    out_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1(
        (MODEL_NAME + "|" + PREPROCESS_VERSION + "|" + "|".join(texts)).encode("utf-8")
    ).hexdigest()[:16]
    cache = out_dir / f"embeddings_{key}.npy"
    if cache.exists():
        print(f"Using cached embeddings: {cache}")
        return np.load(cache)

    import torch
    from transformers import AutoModel, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Encoding {len(texts)} texts with {MODEL_NAME} on {device} ...")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()

    embs = []
    with torch.no_grad():
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i : i + BATCH_SIZE]
            enc = tok(
                batch, padding=True, truncation=True,
                max_length=MAX_LEN, return_tensors="pt",
            ).to(device)
            out = model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1)
            pooled = (out * mask).sum(1) / mask.sum(1)
            embs.append(pooled.cpu().numpy())
            print(f"  {min(i + BATCH_SIZE, len(texts))}/{len(texts)}", end="\r")
    X = np.vstack(embs).astype(np.float32)
    np.save(cache, X)
    print(f"\nSaved embeddings {X.shape} -> {cache}")
    return X


# ==========================================================================
# STEPS 5-6 — HEADS + HONEST EVALUATION
# A HeadModel = scalers (fit on TRAIN folds only) + logistic regression.
# C is selected by inner grouped CV (PR-AUC), never on test rows.
# ==========================================================================
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

# Make the model class pickle-stable for joblib loads across scripts.
sys.modules.setdefault("train", sys.modules[__name__])


class HeadModel:
    """Logistic head on [scaled embeddings] (+ [scaled struct features])."""

    def __init__(self, use_struct: bool, C: float = 1.0):
        self.use_struct = use_struct
        self.C = C
        self.scaler_e = StandardScaler()
        self.scaler_s = StandardScaler() if use_struct else None
        self.lr = LogisticRegression(
            C=C, class_weight="balanced", max_iter=1000,
            random_state=RANDOM_STATE,
        )

    def _join(self, X, S, fit=False):
        Xe = self.scaler_e.fit_transform(X) if fit else self.scaler_e.transform(X)
        if not self.use_struct:
            return Xe
        Ss = self.scaler_s.fit_transform(S) if fit else self.scaler_s.transform(S)
        return np.hstack([Xe, Ss])

    def fit(self, X, S, y, sample_weight=None):
        Z = self._join(X, S, fit=True)
        self.lr.fit(Z, y, sample_weight=sample_weight)
        return self

    def predict_proba(self, X, S):
        return self.lr.predict_proba(self._join(X, S, fit=False))[:, 1]


HeadModel.__module__ = "train"


def tune_threshold(y_true, probs, min_precision=None) -> float:
    """Best-F1 threshold; with min_precision, the best F1 among thresholds
    that clear the precision floor (fallback: plain best F1)."""
    if len(y_true) == 0:
        return 0.5

    def f1_at(t):
        pred = (probs >= t).astype(int)
        p, _, f1, _ = precision_recall_fscore_support(
            y_true, pred, average="binary", zero_division=0
        )
        return f1, p

    best_t, best_f1 = 0.5, -1.0
    floor_t, floor_f1 = None, -1.0
    for t in np.arange(0.05, 0.96, 0.01):
        f1, p = f1_at(t)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
        if min_precision is not None and p >= min_precision and f1 > floor_f1:
            floor_f1, floor_t = f1, float(t)
    return floor_t if floor_t is not None else best_t


def metrics_at(y_true, probs, thr, mask=None) -> dict:
    if mask is not None:
        y_true, probs = y_true[mask], probs[mask]
    if len(y_true) == 0 or len(set(y_true)) < 2:
        return {"n": int(len(y_true))}
    pred = (probs >= thr).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, pred, average="binary", zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "n": int(len(y_true)),
        "accuracy": round(float(accuracy_score(y_true, pred)), 4),
        "precision": round(float(p), 4),
        "recall": round(float(r), 4),
        "f1": round(float(f1), 4),
        "pr_auc": round(float(average_precision_score(y_true, probs)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, probs)), 4),
        # If roc_auc is decent but prob_std is tiny, the ranking is fine
        # and the head/calibration was the problem. If roc_auc ~ 0.5,
        # the representation itself carries no signal.
        "prob_std": round(float(np.std(probs)), 4),
        "prob_mean": round(float(np.mean(probs)), 4),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def _pick_C(X, S, y, groups, use_struct, w=None) -> float:
    """Inner 3-fold grouped CV: the C with the best mean PR-AUC."""
    inner = StratifiedGroupKFold(3, shuffle=True, random_state=7)
    scores = {}
    for C in C_GRID:
        aps = []
        for itr, ite in inner.split(X, y, groups):
            m = HeadModel(use_struct, C=C).fit(
                X[itr], None if S is None else S[itr], y[itr],
                sample_weight=None if w is None else w[itr],
            )
            p = m.predict_proba(X[ite], None if S is None else S[ite])
            if len(set(y[ite])) > 1:
                aps.append(average_precision_score(y[ite], p))
        scores[C] = float(np.mean(aps)) if aps else 0.0
    return max(scores, key=scores.get)


def evaluate_arm(X, S, y, groups, is_real, name, use_struct,
                 min_precision=None, w=None):
    """Outer 5-fold grouped CV for one embedding arm. Per fold: pick C by
    inner CV, tune the threshold on inner OOF probs, then score the fold."""
    outer = StratifiedGroupKFold(N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    oof = np.zeros(len(y))
    fold_thresholds, fold_Cs = [], []
    for tr, te in outer.split(X, y, groups):
        S_tr = None if S is None else S[tr]
        w_tr = None if w is None else w[tr]
        C = _pick_C(X[tr], S_tr, y[tr], groups[tr], use_struct, w_tr)
        fold_Cs.append(C)
        inner = StratifiedGroupKFold(3, shuffle=True, random_state=1)
        inner_oof = np.zeros(len(tr))
        for itr, ite in inner.split(X[tr], y[tr], groups[tr]):
            m = HeadModel(use_struct, C=C).fit(
                X[tr][itr], None if S_tr is None else S_tr[itr], y[tr][itr],
                sample_weight=None if w_tr is None else w_tr[itr],
            )
            inner_oof[ite] = m.predict_proba(
                X[tr][ite], None if S_tr is None else S_tr[ite]
            )
        fold_thresholds.append(
            tune_threshold(y[tr], inner_oof, min_precision=min_precision)
        )
        model = HeadModel(use_struct, C=C).fit(X[tr], S_tr, y[tr], sample_weight=w_tr)
        oof[te] = model.predict_proba(X[te], None if S is None else S[te])
    final_thr = float(np.median(fold_thresholds))
    report = {
        "threshold": round(final_thr, 3),
        "C_values": fold_Cs,
        "all_rows": metrics_at(y, oof, final_thr),
        "real_only": metrics_at(y, oof, final_thr, mask=is_real),
        "at_0.5_all_rows": metrics_at(y, oof, 0.5),
    }
    print(f"\n--- {name} ---  threshold={final_thr:.2f}  C per fold={fold_Cs}")
    print(f"  all rows : {report['all_rows']}")
    print(f"  real only: {report['real_only']}")
    return report, final_thr, oof


def evaluate_tfidf(texts, y, groups, is_real, name, min_precision=None, w=None):
    """Word + char TF-IDF arm on the SAME grouped splits (the baseline
    v2 never had). Char n-grams are strong on small Indic SMS data."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import FeatureUnion, Pipeline

    def make():
        return Pipeline([
            ("tf", FeatureUnion([
                ("word", TfidfVectorizer(
                    ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                    max_features=20000)),
                ("char", TfidfVectorizer(
                    analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                    sublinear_tf=True, max_features=30000)),
            ])),
            ("lr", LogisticRegression(
                C=1.0, class_weight="balanced", max_iter=1000,
                random_state=RANDOM_STATE)),
        ])

    outer = StratifiedGroupKFold(N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    oof = np.zeros(len(y))
    fold_thresholds = []
    for tr, te in outer.split(texts, y, groups):
        inner = StratifiedGroupKFold(3, shuffle=True, random_state=1)
        inner_oof = np.zeros(len(tr))
        for itr, ite in inner.split(texts[tr], y[tr], groups[tr]):
            m = make().fit(
                texts[tr][itr], y[tr][itr],
                lr__sample_weight=None if w is None else w[tr][itr],
            )
            inner_oof[ite] = m.predict_proba(texts[tr][ite])[:, 1]
        fold_thresholds.append(
            tune_threshold(y[tr], inner_oof, min_precision=min_precision)
        )
        model = make().fit(
            texts[tr], y[tr],
            lr__sample_weight=None if w is None else w[tr],
        )
        oof[te] = model.predict_proba(texts[te])[:, 1]
    final_thr = float(np.median(fold_thresholds))
    report = {
        "threshold": round(final_thr, 3),
        "all_rows": metrics_at(y, oof, final_thr),
        "real_only": metrics_at(y, oof, final_thr, mask=is_real),
        "at_0.5_all_rows": metrics_at(y, oof, 0.5),
    }
    print(f"\n--- {name} (TF-IDF) ---  threshold={final_thr:.2f}")
    print(f"  all rows : {report['all_rows']}")
    print(f"  real only: {report['real_only']}")
    return report, final_thr, oof


def evaluate_joint_struct(X, S, y, groups, lang_arr, is_real, w=None):
    """One pooled emb+struct head; per-language thresholds tuned on that
    language's OOF probs (comparison arm, like v2's joint)."""
    outer = StratifiedGroupKFold(N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    oof = np.zeros(len(y))
    for tr, te in outer.split(X, y, groups):
        C = _pick_C(X[tr], S[tr], y[tr], groups[tr], True, None if w is None else w[tr])
        m = HeadModel(True, C=C).fit(
            X[tr], S[tr], y[tr], sample_weight=None if w is None else w[tr]
        )
        oof[te] = m.predict_proba(X[te], S[te])
    report = {}
    for lang in LANGUAGES:
        ml = lang_arr == lang
        if not ml.any():
            continue
        thr = tune_threshold(
            y[ml], oof[ml], min_precision=PRECISION_FLOOR.get(lang)
        )
        report[lang] = {
            "threshold": round(thr, 3),
            "all_rows": metrics_at(y[ml], oof[ml], thr),
            "real_only": metrics_at(y[ml], oof[ml], thr, mask=is_real[ml]),
        }
        print(f"\n--- JOINT emb+struct, {lang} slice ---  threshold={thr:.2f}")
        print(f"  real only: {report[lang]['real_only']}")
    return report, oof


# ==========================================================================
# STEP 7 — FINAL TRAINING + EXPORT
# The winning arm per language (by real-only CV F1) is retrained on 100%
# of that language's data and saved. v3 heads can depend on structured
# features, so they ship as joblib objects for test_batch.py; the ONNX
# encoder export still runs, but config.json marks extension_ready=false
# until the extension learns the v3 features. Honest staging, no fakery.
# ==========================================================================
def train_final_and_export(df, X, S, thresholds, head_source, out_dir: Path):
    import joblib

    model_dir = out_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    y = df["label"].to_numpy()
    groups = df["skeleton"].to_numpy()
    lang_arr = df["language"].to_numpy()
    texts = df["clean"].to_numpy()
    w = np.where(df["is_synthetic"].astype(int) == 0, 1.0, SYNTHETIC_WEIGHT)

    for lang in LANGUAGES:
        m = lang_arr == lang
        src = head_source[lang]
        if src == "tfidf":
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.pipeline import FeatureUnion, Pipeline

            model = Pipeline([
                ("tf", FeatureUnion([
                    ("word", TfidfVectorizer(
                        ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                        max_features=20000)),
                    ("char", TfidfVectorizer(
                        analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                        sublinear_tf=True, max_features=30000)),
                ])),
                ("lr", LogisticRegression(
                    C=1.0, class_weight="balanced", max_iter=1000,
                    random_state=RANDOM_STATE)),
            ])
            model.fit(texts[m], y[m], lr__sample_weight=w[m])
            joblib.dump(model, out_dir / f"head_tfidf_{lang.lower()}.joblib")
        elif src == "joint_struct":
            C = _pick_C(X, S, y, groups, True, w)
            model = HeadModel(True, C=C).fit(X, S, y, sample_weight=w)
            joblib.dump(model, out_dir / "head_joint_struct.joblib")
        else:
            use_struct = src == "own_struct"
            C = _pick_C(X[m], S[m], y[m], groups[m], use_struct, w[m])
            model = HeadModel(use_struct, C=C).fit(
                X[m], S[m], y[m], sample_weight=w[m]
            )
            joblib.dump(model, out_dir / f"head_{lang.lower()}.joblib")
        print(f"Final head for {lang}: {src}")

    config = {
        "preprocess_version": PREPROCESS_VERSION,
        "encoder": MODEL_NAME,
        "max_len": MAX_LEN,
        "thresholds": thresholds,
        "head_source": head_source,
        "struct_feature_names": STRUCT_FEATURE_NAMES,
        "labels": {"0": "ham", "1": "fraud"},
        # v3 heads may need structured features the extension does not
        # compute yet. Do NOT copy this model folder into the extension
        # until that is built; test_batch.py is the v3 test path.
        "extension_ready": False,
    }
    (model_dir / "config.json").write_text(json.dumps(config, indent=2))
    print(f"Saved heads + config.json -> {model_dir}")

    try:
        import torch
        from transformers import AutoModel, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(MODEL_NAME)
        tok.save_pretrained(model_dir)
        model = AutoModel.from_pretrained(MODEL_NAME).eval()
        dummy = tok(["hello"], return_tensors="pt")
        torch.onnx.export(
            model,
            (dummy["input_ids"], dummy["attention_mask"]),
            str(model_dir / "model.onnx"),
            input_names=["input_ids", "attention_mask"],
            output_names=["last_hidden_state"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "seq"},
                "attention_mask": {0: "batch", 1: "seq"},
                "last_hidden_state": {0: "batch", 1: "seq"},
            },
            opset_version=14,
        )
        print(f"Exported ONNX encoder -> {model_dir / 'model.onnx'}")
    except Exception as e:
        print(f"ONNX export skipped ({e}). Heads/config are saved.")


# ==========================================================================
# DEMO — one message through the shipped head (UTF-8 file input is safer;
# Windows console paste mangles Telugu/Hindi — prefer test_batch.py)
# ==========================================================================
def demo(message: str, out_dir: Path):
    import joblib

    config = json.loads((out_dir / "model" / "config.json").read_text())
    lang = detect_language(message)
    clean = anonymize(message)
    src = config.get("head_source", {}).get(lang, "own_struct")
    thr = config["thresholds"][lang]
    if src == "tfidf":
        head = joblib.load(out_dir / f"head_tfidf_{lang.lower()}.joblib")
        prob = float(head.predict_proba([clean])[0, 1])
    else:
        if src == "joint_struct":
            head = joblib.load(out_dir / "head_joint_struct.joblib")
        else:
            head = joblib.load(out_dir / f"head_{lang.lower()}.joblib")
        X = get_embeddings([clean], out_dir / "demo_cache")
        S = struct_matrix([message])
        prob = float(head.predict_proba(X, S)[0])
    verdict = "FRAUD" if prob >= thr else "SAFE"
    print(f"language={lang}  head={src}  cleaned={clean!r}")
    print(f"fraud probability={prob:.3f}  threshold={thr}  -> {verdict}")


# ==========================================================================
# MAIN — arms per language: emb | emb+struct | tfidf, plus joint emb+struct
# ==========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="train_balanced_v2.csv",
                    help="dataset CSV (default: train_balanced_v2.csv)")
    ap.add_argument("--demo", default=None)
    ap.add_argument("--no-export", action="store_true")
    args = ap.parse_args()

    df = load_data(args.csv)
    X = get_embeddings(df["clean"].tolist(), OUT_DIR)
    S = struct_matrix(df["text"].tolist())
    y = df["label"].to_numpy()
    groups = df["skeleton"].to_numpy()
    is_real = (df["is_synthetic"].astype(int) == 0).to_numpy()
    lang_arr = df["language"].to_numpy()
    texts = df["clean"].to_numpy()
    w = np.where(is_real, 1.0, SYNTHETIC_WEIGHT)

    results, thresholds, oofs = {}, {}, {}
    for lang in LANGUAGES:
        m = lang_arr == lang
        floor = PRECISION_FLOOR.get(lang)
        results[lang] = {}
        rep, thr, oof = evaluate_arm(
            X[m], None, y[m], groups[m], is_real[m],
            f"{lang} (emb only, v3 head)", False, min_precision=floor, w=w[m],
        )
        results[lang]["emb"] = rep; oofs[f"{lang}_emb"] = (m, oof)
        rep, thr_s, oof = evaluate_arm(
            X[m], S[m], y[m], groups[m], is_real[m],
            f"{lang} (emb + struct)", True, min_precision=floor, w=w[m],
        )
        results[lang]["emb_struct"] = rep; oofs[f"{lang}_struct"] = (m, oof)
        rep, thr_t, oof = evaluate_tfidf(
            texts[m], y[m], groups[m], is_real[m],
            f"{lang}", min_precision=floor, w=w[m],
        )
        results[lang]["tfidf"] = rep; oofs[f"{lang}_tfidf"] = (m, oof)
        results[lang]["_thresholds"] = {
            "emb": thr, "emb_struct": thr_s, "tfidf": thr_t,
        }
    joint_report, joint_oof = evaluate_joint_struct(
        X, S, y, groups, lang_arr, is_real, w=w
    )
    results["joint_struct"] = joint_report
    oofs["joint_struct"] = (np.ones(len(y), dtype=bool), joint_oof)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "cv_results.json").write_text(json.dumps(results, indent=2))
    np.savez(OUT_DIR / "oof_probs.npz",
             **{k: v[1] for k, v in oofs.items()})
    print(f"\nSaved CV results -> {OUT_DIR / 'cv_results.json'}")

    print("\n=========== SUMMARY (real-only) ===========")
    print("arm choice per language = best real-only F1")
    head_source, final_thresholds = {}, {}
    arm_key = {"own_emb": "emb", "own_struct": "emb_struct", "tfidf": "tfidf"}
    for lang in LANGUAGES:
        cands = {
            "own_emb": results[lang]["emb"]["real_only"].get("f1") or 0,
            "own_struct": results[lang]["emb_struct"]["real_only"].get("f1") or 0,
            "tfidf": results[lang]["tfidf"]["real_only"].get("f1") or 0,
            "joint_struct": joint_report[lang]["real_only"].get("f1") or 0,
        }
        pick = max(cands, key=cands.get)
        head_source[lang] = pick
        if pick == "joint_struct":
            final_thresholds[lang] = joint_report[lang]["threshold"]
        else:
            final_thresholds[lang] = results[lang]["_thresholds"][arm_key[pick]]
        auc = results[lang]["emb_struct"]["real_only"].get("roc_auc")
        std = results[lang]["emb_struct"]["real_only"].get("prob_std")
        print(
            f"{lang:8s} " +
            "  ".join(f"{k} F1={v:.3f}" for k, v in cands.items()) +
            f"   -> ship {pick.upper()}"
        )
        print(
            f"         emb+struct real-only ROC-AUC={auc}  prob_std={std}"
            "  (AUC>>0.5 + tiny std = head problem; AUC~0.5 = representation)"
        )

    if not args.no_export:
        train_final_and_export(df, X, S, final_thresholds, head_source, OUT_DIR)
    if args.demo:
        demo(args.demo, OUT_DIR)


if __name__ == "__main__":
    main()
