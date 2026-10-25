#!/usr/bin/env python3
"""
test_batch.py (v3) — Batch-test the v3 classifier on NEW messages.
==================================================================
Same contract as v2: CSV in (text [, label, language]), verdicts +
scorecard out, test_results.csv saved. What changed: v3 heads can be
embedding heads, embedding+structured-feature heads, a joint head, or
TF-IDF pipelines — this script loads whatever config.json says won.

RUN (after train.py has produced outputs/):
  python test_batch.py test_messages.csv
NEVER paste Telugu/Hindi into a Windows terminal — always use a UTF-8
CSV file, or the script you type will be mangled before it reaches us.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from train import anonymize, detect_language, get_embeddings, struct_matrix


def load_heads(outputs: Path, config):
    import joblib

    head_source = config.get("head_source", {})
    heads = {}
    joint = None
    for lang in ("English", "Hindi", "Telugu"):
        src = head_source.get(lang, "own_struct")
        if src == "tfidf":
            heads[lang] = ("tfidf", joblib.load(
                outputs / f"head_tfidf_{lang.lower()}.joblib"))
        elif src == "joint_struct":
            if joint is None:
                joint = joblib.load(outputs / "head_joint_struct.joblib")
            heads[lang] = ("headmodel", joint)
        else:
            heads[lang] = ("headmodel", joblib.load(
                outputs / f"head_{lang.lower()}.joblib"))
    return heads


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="CSV with text [, label, language] columns")
    ap.add_argument("--outputs", default="outputs")
    args = ap.parse_args()

    outputs = Path(args.outputs)
    if not (outputs / "model" / "config.json").exists():
        raise SystemExit(f"No trained model in {outputs}/ — run train.py first.")
    config = json.loads((outputs / "model" / "config.json").read_text())
    thresholds = config["thresholds"]
    heads = load_heads(outputs, config)

    df = pd.read_csv(args.csv, encoding="utf-8")
    if "text" not in df.columns:
        raise SystemExit("CSV needs a 'text' column.")
    df["clean"] = df["text"].map(anonymize)
    if "language" not in df.columns:
        df["language"] = None
    df["language"] = df.apply(
        lambda r: r["language"]
        if isinstance(r["language"], str) and r["language"].strip()
        else detect_language(str(r["text"])),
        axis=1,
    )

    probs = np.zeros(len(df))
    # TF-IDF heads score from cleaned text; HeadModels need embeddings.
    need_emb = [i for i, lang in enumerate(df["language"])
                if heads[lang][0] == "headmodel"]
    if need_emb:
        print(f"Embedding {len(need_emb)} messages ...")
        X = get_embeddings(df["clean"].tolist(), outputs / "batch_cache")
        S = struct_matrix(df["text"].tolist())
    for i, lang in enumerate(df["language"]):
        kind, head = heads[lang]
        if kind == "tfidf":
            probs[i] = head.predict_proba([df["clean"].iloc[i]])[0, 1]
        else:
            probs[i] = head.predict_proba(X[i:i + 1], S[i:i + 1])[0]
    df["fraud_prob"] = probs.round(3)
    df["threshold"] = df["language"].map(thresholds)
    df["verdict"] = np.where(df["fraud_prob"] >= df["threshold"], "FRAUD", "SAFE")

    has_labels = "label" in df.columns and df["label"].notna().any()
    if has_labels:
        df["label"] = df["label"].astype(int)
        df["correct"] = (df["verdict"] == "FRAUD") == (df["label"] == 1)
    else:
        df["correct"] = ""

    print("\n================ RESULTS ================")
    for _, r in df.iterrows():
        mark = ""
        if has_labels:
            mark = "  ✓" if r["correct"] else "  ✗ WRONG"
        print(
            f"[{r['language']:7s}] p={r['fraud_prob']:.2f} "
            f"(thr {r['threshold']}) -> {r['verdict']:5s}{mark}  "
            f"{str(r['text'])[:70]}"
        )

    if has_labels:
        from sklearn.metrics import precision_recall_fscore_support

        def scorecard(d, name):
            if len(d) == 0 or d["label"].nunique() < 2:
                print(f"{name}: n={len(d)} (need both fraud and ham rows)")
                return
            pred = (d["verdict"] == "FRAUD").astype(int)
            p, r, f1, _ = precision_recall_fscore_support(
                d["label"], pred, average="binary", zero_division=0
            )
            acc = (pred == d["label"]).mean()
            print(
                f"{name}: n={len(d)}  accuracy={acc:.3f}  "
                f"precision={p:.3f}  recall={r:.3f}  F1={f1:.3f}"
            )

        print("\n================ SCORECARD ================")
        scorecard(df, "ALL     ")
        for lang in ("English", "Hindi", "Telugu"):
            scorecard(df[df["language"] == lang], f"{lang:8s}")

    out_path = Path("test_results.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved full results -> {out_path}")


if __name__ == "__main__":
    main()
