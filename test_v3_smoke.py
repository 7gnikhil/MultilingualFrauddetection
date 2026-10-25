#!/usr/bin/env python3
"""Smoke test for v3 (no torch / no MuRIL download needed).
Verifies: Indic fallback normalisation, struct features separating a
real bank alert from a fake one, HeadModel + C-pick + threshold floor,
the TF-IDF arm end-to-end, and evaluate_arm on dummy embeddings."""
import numpy as np

import train as T

fails = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        fails.append(name)


# 1. Zero-width chars removed by cleaning (v2 kept them!)
te_zwj = "మీ ఖాతా‍ బ్లాక్‌ చేయబడుతుంది"  # ZWJ + ZWNJ inside
assert "‍" in te_zwj and "‌" in te_zwj
cleaned = T.anonymize(te_zwj)
check("ZWJ/ZWNJ removed by anonymize", "‍" not in cleaned and "‌" not in cleaned)

# 2. Masking contract still intact
c = T.anonymize("Your OTP is 482913, call 9876543210 or visit https://x.in")
check("masking still works", "[OTP]" in c and "[PHONE]" in c and "[URL]" in c)

# 3. Structured features: real alert vs fake alert must differ
real = ("Dear customer, Rs.5,000 debited from your A/c XX1234 on 12-Oct. "
        "Your OTP is 482913. Do not share it with anyone. Bank never asks "
        "for OTP. Visit https://www.hdfcbank.com")
fake = ("URGENT! Your account will be blocked today. Rs.70,000 debited. "
        "Share your OTP 991122 to verify now: https://bit.ly/xyz123 "
        "call 9876543210")
fr, ff = T.extract_struct_features(real), T.extract_struct_features(fake)
names = T.STRUCT_FEATURE_NAMES
check("struct feature length", len(fr) == len(names) == 30)
d = dict(zip(names, ff - fr))
check("fake has shortener, real does not",
      ff[names.index("url_is_shortener")] == 1 and fr[names.index("url_is_shortener")] == 0)
check("real flagged do-not-share", fr[names.index("mentions_do_not_share")] == 1)
check("real flagged never-ask", fr[names.index("mentions_never_ask")] == 1)
check("fake OTP marked requested", ff[names.index("otp_is_requested")] == 1)
check("real OTP NOT marked requested", fr[names.index("otp_is_requested")] == 0)
check("amount bucket fake>real", ff[names.index("amount_log10")] > fr[names.index("amount_log10")])

# Telugu + Hindi struct features don't crash and see native digits
te = "మీ ఖాతా నుంచి రూ. ౭౦,౦౦౦ డెబిట్ అయ్యాయి, OTP పంచుకోండి https://bit.ly/a"
hi = "आपके खाते से ₹५,००० डेबिट हुआ, OTP किसी को मत बताओ"
fte, fhi = T.extract_struct_features(te), T.extract_struct_features(hi)
check("telugu amount parsed from native digits", fte[names.index("amount_log10")] > 4)
check("telugu shortener seen", fte[names.index("url_is_shortener")] == 1)
check("hindi amount parsed", fhi[names.index("amount_log10")] > 3)
check("telugu cleaned keeps script", "ఖాతా" in T.anonymize(te))

# 4. HeadModel + C selection + precision-floor threshold on dummy data
rng = np.random.default_rng(0)
n = 120
X = rng.normal(size=(n, 16)).astype(np.float32)
S = rng.normal(size=(n, 30)).astype(np.float32)
y = (X[:, 0] + S[:, 5] > 0).astype(int)
groups = np.array([f"s{i//3}" for i in range(n)])
C = T._pick_C(X, S, y, groups, True, None)
check("C picked from grid", C in T.C_GRID)
rep, thr, oof = T.evaluate_arm(
    X, S, y, groups, np.ones(n, bool), "smoke emb+struct", True,
    min_precision=0.75, w=np.ones(n),
)
check("evaluate_arm runs, sane F1", rep["real_only"]["f1"] > 0.5)
check("roc_auc reported", "roc_auc" in rep["real_only"])

# 5. TF-IDF arm end-to-end on tiny texts
texts = np.array([
    "win prize money click now", "lottery jackpot claim urgent",
    "your otp share verify blocked", "free cash prize winner",
    "meeting at office tomorrow", "lunch with family sunday",
    "your package delivered today", "bill payment received thanks",
] * 3)
yy = np.array([1, 1, 1, 1, 0, 0, 0, 0] * 3)
gg = np.array([f"g{i}" for i in range(len(texts))])
rep2, thr2, _ = T.evaluate_tfidf(texts, yy, gg, np.ones(len(yy), bool), "smoke")
check("tfidf arm runs", rep2["real_only"]["f1"] >= 0.5)

print("\nRESULT:", "ALL PASS" if not fails else f"{len(fails)} FAILURES: {fails}")
raise SystemExit(1 if fails else 0)
