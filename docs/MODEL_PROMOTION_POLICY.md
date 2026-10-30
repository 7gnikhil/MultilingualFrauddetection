# Model Promotion Policy — Champion / Challenger

This policy governs how a newly trained model (the **challenger**) may replace
the model currently shipped to users (the **champion**). It exists because the
v3 extension showed what happens without one: a model that looked strong on
cross-validation still misbehaved on real messages, and there was no locked
test, no challenge set, and no rollback path.

## Principles

1. **The champion is never overwritten automatically.** The retrain workflow
   may train and evaluate a challenger and open a pull request. Only a human
   merging that PR promotes the challenger.
2. **All headline metrics are real-only.** Synthetic/paraphrase rows may be
   used for training (weighted down) but are excluded from every reported
   metric. A score computed on synthetic data is not evidence.
3. **No leakage.** Splits are grouped by message *skeleton* (the message with
   numbers, links, names and amounts stripped), so near-identical copies of
   one message can never appear in both train and test. Near-100% scores are
   treated as a leakage alarm, not a success.
4. **The test and challenge sets are locked.** They are never trained on,
   never used for threshold tuning, and never edited to make a challenger
   pass. Changing them requires a separate, reviewed PR and invalidates
   comparisons with earlier champions.
5. **Train/serve parity.** The exact anonymisation used in training
   (`[OTP]`, `[PHONE]`, `[ACCOUNT]`, `[EMAIL]`, `[UPI]`, `[URL]`) and the same
   preprocessing run at inference, in the extension and in evaluation.

## Evaluation protocol

- **Grouped CV:** skeleton-grouped 5-fold cross-validation over the real
  dataset; metrics reported per language (English, Hindi, Telugu) and macro.
- **Locked challenge set:** a small, curated, 100%-real set of hard cases:
  - fraud-awareness / warning messages (these are **ham** — the v3 failure mode),
  - legitimate bank/UPI alerts that resemble scams,
  - new scam templates collected from user feedback,
  - romanised Telugu / Hinglish code-mixed messages,
  - messages with links, OTP requests and payment demands (true fraud).
- **Recent-feedback slice:** the newest user corrections are held out of
  training for one cycle and used as an extra evaluation slice, so the loop
  cannot silently overfit to the feedback it just consumed.
- Thresholds are re-tuned on validation folds only, per language, then frozen
  into the challenger config before locked evaluation.

## Promotion gates

A challenger is eligible for a promotion PR only if **all** gates pass.
Defaults live in `ml/configs/promotion_gates.yaml`; changing them requires
review like any other code change.

| Gate | Requirement |
| --- | --- |
| Macro F1 (real-only) | challenger ≥ champion + 0.003 |
| Per-language F1 regression | no language drops more than **0.010** vs champion |
| Precision floor (real-only) | EN ≥ 0.85, HI ≥ 0.80, TE ≥ 0.75 |
| Precision regression | no language drops more than **0.020** vs champion |
| Recall regression | no language drops more than **0.030** vs champion |
| Challenge set | accuracy ≥ 0.95 **and** not worse than champion |
| Awareness subset | false positives must not increase vs champion, and FP rate ≤ 2% |
| Recent-feedback slice | F1 regression ≤ 0.020 vs champion |
| Minimum evidence | ≥ 150 real evaluation rows per language, else the PR is labelled `low-evidence` and needs an explicit human note to merge |

Rationale for the shape of these gates: for a fraud-warning tool, a false
alarm on a legitimate bank alert or an awareness message destroys trust
faster than a missed scam, so precision has both a floor and a regression
cap; but recall has a regression cap too, because a "safe" verdict on a real
scam is the most dangerous error the product can make.

## Promotion procedure

1. `retrain.yml` trains the challenger, evaluates the gates, and uploads the
   full evaluation report as a workflow artifact.
2. If the gates pass, the workflow opens a PR adding the challenger config
   and its evaluation report under `models/candidate/`.
3. A human reviewer checks: the per-language diff table, the challenge-set
   results (especially awareness false positives), feedback volume and the
   label spot-check note, and the dataset snapshot hash.
4. Merging the PR is the promotion decision. The merge triggers a model
   Release (`model-v<version>`) containing the ONNX encoder (if changed),
   `config.json` (heads, scalers, thresholds), and the evaluation report.
5. The extension fetches the latest model config, verifies its SHA-256
   against the release checksums, and falls back to its bundled config if
   verification or download fails.

## Rollback

- The last **3 champion configs** are kept in the repo history
  (`models/champion/` via git) and as Release assets; nothing is deleted.
- **Rollback = revert the promotion PR** (or re-publish the previous config
  as a new Release). Target: under 15 minutes, no retraining involved.
- The extension pins the model-config version it was built against as a
  minimum and ignores configs older than the one it already runs, so a bad
  config cannot be re-served after rollback.
- Roll back immediately if any of these are observed in the field:
  a spike in user "It's safe / It's fraud" corrections against the new model,
  a challenge-set regression discovered after release, or signs of feedback
  poisoning (see below).

## Feedback-data governance (poisoning & quality guards)

User corrections are training data, so they are an attack surface:

- Corrections are **anonymised before storage** (same placeholders as
  training) and stored only in the private feedback repo.
- A retrain triggers only after a minimum volume of new corrections
  (default 50) — single users cannot steer the model.
- Per-account/per-device contribution is capped in preprocessing; bursts of
  near-identical corrections are deduplicated by skeleton and down-weighted.
- A human spot-checks a sample of new labels before they enter a training
  run; the spot-check result is recorded in the evaluation report.
- If poisoning is suspected: freeze the loop (disable the dispatch trigger),
  roll back to the last clean champion, and drop the suspect feedback range.

## Exceptions

There is no emergency path that skips the gates. A urgent fix ships as a
new challenger through the same pipeline; if the champion is actively
harmful, roll it back first (previous section) and let the fix follow the
normal route.
