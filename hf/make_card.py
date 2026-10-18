"""Writes the Hugging Face model card from the Kaggle run's metrics, so every number matches the run that made the model."""
import json, sys

d = sys.argv[1]  # folder with metrics.json, thresholds.json, browser_agreement.json
m = json.load(open(f"{d}/metrics.json"))
t = json.load(open(f"{d}/thresholds.json"))
a = json.load(open(f"{d}/browser_agreement.json"))
pct = lambda x: f"{x * 100:.1f}%"
B, D = "baseline", "distilbert"
tox = {k: m["per_label"][k]["toxicity"] for k in (B, D)}
pol = m["policies"]
fair = {k: {g["group"]: g["fpr"] for g in m["fairness"][k]["groups"]} for k in (B, D)}
rows = "\n".join(f"| {g.replace('_', ' ')} | {pct(fair[B][g])} | {pct(fair[D][g])} |"
                 for g in sorted(fair[D], key=fair[D].get, reverse=True))
worst = max(fair[D].values()) / m["fairness"][D]["overall_fpr_identity_labelled"]
cats = ", ".join(f"{l.replace('_', ' ')} {m['per_label'][D][l]['roc_auc']:.3f}" for l in ["obscene", "identity_attack", "insult", "threat"])

print(f"""---
license: apache-2.0
language: [en]
datasets: [google/civil_comments]
base_model: distilbert-base-uncased
pipeline_tag: text-classification
tags: [content-moderation, toxicity, trust-and-safety, fairness, portfolio]
---
# DistilBERT moderation classifier (Civil Comments)

> **Content warning:** trained on real toxic comments. **Portfolio demo, not a production moderation system.**

A portfolio project by [Malav Naik](https://malavnaik99.github.io). Code: [github.com/malavnaik99/moderation-classifier](https://github.com/malavnaik99/moderation-classifier) · Notebook: [Kaggle](https://www.kaggle.com/code/malavnaik/moderation-classifier-civil-comments) · Demo: [Space](https://huggingface.co/spaces/Malavnaik/moderation-classifier-demo)

**What it does:** gives a 0–1 score for five categories: `toxicity`, `obscene`, `identity_attack`, `insult`, `threat` (in that output order; sigmoid per output, multi-label).

**Training:** `distilbert-base-uncased`, 1 epoch on a random {m['bert_train_rows']:,} comments from the training split of the [Jigsaw Unintended Bias / Civil Comments](https://www.kaggle.com/competitions/jigsaw-unintended-bias-in-toxicity-classification) data (CC0), max 128 tokens, Kaggle free GPU. A label counts as positive when ≥ 50% of raters said yes. `severe_toxicity` was left out: too rare to measure.

**Intended use:** learning and demonstration of moderation metrics. **Out of scope:** making real decisions about people or content.

## Results ({m['test_rows']:,} held-out test comments, {pct(m['prevalence']['true_test_share'])} toxic)

| Toxicity | TF-IDF + logistic regression baseline | This model |
|---|---|---|
| ROC-AUC | {tox[B]['roc_auc']:.3f} | **{tox[D]['roc_auc']:.3f}** |
| PR-AUC | {tox[B]['pr_auc']:.3f} | **{tox[D]['pr_auc']:.3f}** |
| Recall at ≥ 90% precision ("auto-action" policy) | {pct(pol[B]['auto_action']['recall'])} | **{pct(pol[D]['auto_action']['recall'])}** |
| Share of comments sent to human review to catch ~80% of toxic comments | {pct(m['review_queue'][B]['review_share'])} | **{pct(m['review_queue'][D]['review_share'])}** |

Per-category ROC-AUC: {cats}.

Thresholds (chosen on the validation set): auto-action ≥ **{t['auto_action']:.3f}**, send to review ≥ **{t['send_to_review']:.3f}** (`thresholds.json`).

## Fairness: false-positive rate by identity mentioned

Share of **non-toxic** comments flagged for review, on test comments with identity annotations.

| Comment mentions | Baseline | This model |
|---|---|---|
| (all identity-annotated comments) | {pct(m['fairness'][B]['overall_fpr_identity_labelled'])} | **{pct(m['fairness'][D]['overall_fpr_identity_labelled'])}** |
{rows}

**Limitation:** harmless comments that mention some identities are still flagged up to **{worst:.1f}×** more often than average. Do not use it to make decisions about people or content without human review.

## Browser version (the demo)
The demo runs a compressed copy (`onnx/model_quantized.onnx`, int8) in the visitor's browser. On {a['comments']:,} random Civil Comments test comments it gave the **same publish / review / auto-action decision as the full model {pct(a['same_decision_share'])} of the time**; toxicity scores differed by {a['mean_abs_diff']:.3f} on average (largest difference {a['max_abs_diff']:.2f}).

## Other limits
- English news-site comments from 2015–2017; other platforms, languages and slang will score differently.
- Labels reflect crowd raters' judgements, which carry their own biases.
- One training run, no hyper-parameter search.
""")
