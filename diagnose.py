#!/usr/bin/env python3
"""
diagnose.py — Phase-0 audit for the v3 fraud pipeline (run BEFORE train.py)
===========================================================================
Checks the four suspects behind the 0.51-0.59 probability clustering:

  D1 Tokenisation audit — per language: tokens/word (fertility), [UNK]
     rate, % of messages truncated at MAX_LEN. High Telugu fertility or
     UNKs on hand-typed text = encoding/tokenisation trouble.
  D2 Placeholder audit — how MuRIL's WordPiece actually tokenises
     [OTP] / [URL] / [PHONE] / [ACCOUNT] / [EMAIL] / [UPI]. The v2/v3
     tokenizer never registered them as special tokens, so expect each
     to split into pieces; this prints the proof.
  D3 Masking audit — how many 4-8 digit runs get masked as [OTP] WITHOUT
     any OTP word nearby (likely amounts / reference numbers being
     swallowed), per language.
  D4 Indic-normalisation effect — how many messages CHANGE under the
     Indic normaliser vs NFC-only, and the mean token-count change.

RUN:  python diagnose.py [--csv train_balanced_v2.csv] [--skip-tokenizer]
The tokenizer audit needs the MuRIL tokenizer files (~a few MB download);
everything else runs offline on the CSV alone.
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from train import (
    LANGUAGES,
    MAX_LEN,
    MODEL_NAME,
    _fallback_normalize,
    _indic_normalize,
    _script_lang,
    anonymize,
)

PLACEHOLDERS = ["[URL]", "[EMAIL]", "[UPI]", "[ACCOUNT]", "[PHONE]", "[OTP]"]
OTP_NEAR = re.compile(r"otp|one[-\s]?time|pin\b|ఓటిపి|ఒటిపి|ओटीपी|पिन", re.I)


def masking_audit(df: pd.DataFrame) -> dict:
    out = {}
    for lang in LANGUAGES:
        sub = df[df["language"] == lang]
        masked_runs, total_runs, msgs_hit = 0, 0, 0
        for text in sub["text"].astype(str):
            low = text.lower()
            runs = list(re.finditer(r"\d{4,8}", text))
            total_runs += len(runs)
            hit = False
            for r in runs:
                window = low[max(0, r.start() - 40): r.end() + 40]
                if not OTP_NEAR.search(window):
                    masked_runs += 1
                    hit = True
            msgs_hit += int(hit)
        out[lang] = {
            "messages": int(len(sub)),
            "digit_runs_4_8": int(total_runs),
            "runs_masked_without_otp_word_nearby": int(masked_runs),
            "messages_affected": int(msgs_hit),
        }
        print(
            f"{lang:8s} digit runs 4-8: {total_runs}, "
            f"masked with NO otp word nearby: {masked_runs} "
            f"({msgs_hit} messages) <- amounts/refs swallowed as [OTP]"
        )
    return out


def tokenizer_audit(df: pd.DataFrame) -> dict:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    print(f"\nTokenizer: {MODEL_NAME}  vocab={tok.vocab_size}")
    ph = {}
    for p in PLACEHOLDERS:
        pieces = tok.tokenize(p)
        ph[p] = pieces
        print(f"  {p:10s} -> {pieces}  ({len(pieces)} pieces)")
    print("  (v3 does not register these as special tokens; the pieces\n"
          "   above are what the frozen model actually sees.)")

    per_lang = {}
    for lang in LANGUAGES:
        sub = df[df["language"] == lang]
        fert, unk_rate, trunc = [], [], 0
        for clean in sub["clean"]:
            ids = tok(clean, add_special_tokens=True)["input_ids"]
            words = max(len(str(clean).split()), 1)
            fert.append(len(ids) / words)
            toks = tok.convert_ids_to_tokens(ids)
            unk_rate.append(sum(t == "[UNK]" for t in toks) / max(len(toks), 1))
            trunc += int(len(ids) >= MAX_LEN)
        per_lang[lang] = {
            "fertility_tokens_per_word": round(float(np.mean(fert)), 3),
            "unk_rate": round(float(np.mean(unk_rate)), 5),
            "pct_truncated_at_max_len": round(100.0 * trunc / max(len(sub), 1), 2),
        }
        print(
            f"{lang:8s} fertility={per_lang[lang]['fertility_tokens_per_word']} "
            f"unk_rate={per_lang[lang]['unk_rate']} "
            f"truncated={per_lang[lang]['pct_truncated_at_max_len']}%"
        )

    # D4: Indic normalisation vs NFC-only + fallback
    changed, token_delta = 0, []
    for clean_nfc in df["clean"].head(2000):
        norm = _indic_normalize(str(clean_nfc), _script_lang(str(clean_nfc)))
        if norm != clean_nfc:
            changed += 1
        token_delta.append(
            len(tok(norm)["input_ids"]) - len(tok(str(clean_nfc))["input_ids"])
        )
    d4 = {
        "messages_changed_by_indic_normalizer_of_first_2000": int(changed),
        "mean_token_count_delta": round(float(np.mean(token_delta)), 3),
    }
    print(f"\nIndic normaliser changes {changed}/2000 sampled messages; "
          f"mean token-count delta {d4['mean_token_count_delta']}")
    return {"placeholders": ph, "per_language": per_lang, "indic_norm": d4}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="train_balanced_v2.csv")
    ap.add_argument("--skip-tokenizer", action="store_true",
                    help="only run the offline masking audit")
    args = ap.parse_args()

    df = pd.read_csv(args.csv, encoding="utf-8")
    df = df.dropna(subset=["text", "language"]).copy()
    df["clean"] = df["text"].map(anonymize)
    print(f"Diagnosing {len(df)} rows from {args.csv}\n")
    print("--- D3 masking audit ---")
    report = {"masking": masking_audit(df)}
    if not args.skip_tokenizer:
        print("\n--- D1/D2/D4 tokenizer audits ---")
        try:
            report["tokenizer"] = tokenizer_audit(df)
        except Exception as e:
            print(f"Tokenizer audit skipped ({e}). Re-run online to get it.")
    out = Path("outputs")
    out.mkdir(exist_ok=True)
    (out / "diagnostics.json").write_text(json.dumps(report, indent=2))
    print(f"\nSaved -> {out / 'diagnostics.json'}")


if __name__ == "__main__":
    main()
