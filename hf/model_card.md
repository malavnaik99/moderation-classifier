---
license: apache-2.0
language: [en]
datasets: [google/civil_comments]
base_model: distilbert-base-uncased
pipeline_tag: text-classification
tags: [content-moderation, toxicity, trust-and-safety, fairness, portfolio]
---
# DistilBERT moderation classifier (Civil Comments)

> **Content warning:** trained on real toxic comments. **Portfolio demo, not a production moderation system.**

A portfolio project by [Malav Naik](https://malavnaik99.github.io). Code and full analysis: [github.com/malavnaik99/moderation-classifier](https://github.com/malavnaik99/moderation-classifier).

**What it does:** gives a 0–1 score for five categories: `toxicity`, `obscene`, `identity_attack`, `insult`, `threat` (in that output order; sigmoid per output, multi-label).

**Training:** `distilbert-base-uncased`, 1 epoch on a random 300,000 comments from the training split of the [Jigsaw Unintended Bias / Civil Comments](https://www.kaggle.com/competitions/jigsaw-unintended-bias-in-toxicity-classification) data (CC0), max 128 tokens, Kaggle free GPU. A label counts as positive when ≥ 50% of raters said yes. `severe_toxicity` was left out: too rare to measure (3 examples in the test set).

## Results (180,488 held-out test comments, 8.0% toxic)

| Toxicity | TF-IDF + logistic regression baseline | This model |
|---|---|---|
| ROC-AUC | 0.941 | **0.957** |
| PR-AUC | 0.706 | **0.753** |
| Recall at 90% precision ("auto-action" policy) | 31% | **38%** |
| Share of comments sent to human review to catch ~80% of toxic comments | 10.7% | **8.3%** |

Per-category ROC-AUC: obscene 0.981, identity_attack 0.978, insult 0.968, threat 0.972.

Thresholds (chosen on the validation set): auto-action ≥ **0.812**, send to review ≥ **0.217** (`thresholds.json`).

## Fairness: false-positive rate by identity mentioned

Share of **non-toxic** comments flagged for review (threshold 0.217), on test comments with identity annotations.

| Comment mentions | Baseline | This model |
|---|---|---|
| (all identity-annotated comments) | 12.8% | **9.5%** |
| black | 49.6% | 39.1% |
| white | 44.3% | 32.0% |
| homosexual, gay or lesbian | 38.6% | 34.4% |
| muslim | 37.2% | 25.6% |
| transgender | 31.6% | 19.8% |
| psychiatric or mental illness | 29.6% | 21.1% |
| jewish | 23.8% | 16.6% |
| male | 20.9% | 15.3% |
| female | 19.7% | 13.1% |
| christian | 12.4% | 8.1% |
| asian | 14.2% | 10.2% |

**Limitation:** the model is fairer than the baseline for every group, but harmless comments that mention some identities are still flagged **2–4× more often** than average. Do not use it to make decisions about people or content without human review.

## Browser version (the demo)
The [demo](https://huggingface.co/spaces/Malavnaik/moderation-classifier-demo) runs a compressed copy (`onnx/model_quantized.onnx`, int8, 67 MB) in the visitor's browser. On 2,000 random Civil Comments test comments it gave the **same publish / review / auto-action decision as the full model 97% of the time**; toxicity scores differed by 0.014 on average (largest difference 0.21).

## Other limits
- English news-site comments from 2015–2017; other platforms, languages and slang will score differently.
- Labels reflect crowd raters' judgements, which carry their own biases.
- One training run, no hyper-parameter search.
