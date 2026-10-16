# %% [markdown]
# # Content moderation classifier on Civil Comments
# **Content warning:** this dataset contains real toxic comments. This notebook never prints comment text; it shows only counts, scores and charts.
#
# A portfolio project by [Malav Naik](https://malavnaik99.github.io). Public data only. This is a portfolio demo, **not a production moderation system**.
#
# **Data:** [Jigsaw Unintended Bias in Toxicity Classification](https://www.kaggle.com/competitions/jigsaw-unintended-bias-in-toxicity-classification) (Civil Comments). The data and the comment text are released under CC0.
#
# **Plan:** explore → TF-IDF + logistic regression baseline → precision/recall and two threshold policies → simulated review queue → prevalence estimate with a confidence interval → false-positive rates by identity group.

# %%
import json, os
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import train_test_split

# Kaggle's mount folder for competition data has changed over time, so search for it.
DATA = os.environ.get("DATA") or next(Path("/kaggle/input").rglob("jigsaw-unintended-bias*/train.csv"), None)
assert DATA, f"Competition data not attached. Found: {[str(p) for p in Path('/kaggle/input').rglob('*')][:20]}"
OUT = Path(os.environ.get("OUT", "/kaggle/working"))
(OUT / "model").mkdir(parents=True, exist_ok=True)
SEED = 42
MIN_GROUP = int(os.environ.get("MIN_GROUP", 200))  # smallest group size reported in the fairness check
LABELS = ["toxicity", "severe_toxicity", "obscene", "identity_attack", "insult", "threat"]
IDENTITIES = ["male", "female", "transgender", "homosexual_gay_or_lesbian", "christian", "jewish", "muslim",
              "black", "white", "asian", "latino", "psychiatric_or_mental_illness"]
metrics = {}

# %% [markdown]
# ## 1. Explore
# Each comment was rated by several people; a label's value is the share of raters who said yes. As in the competition, a comment counts as positive when that share is ≥ 0.5.

# %%
df = pd.read_csv(DATA, usecols=["id", "comment_text", "target", *LABELS[1:], *IDENTITIES])
df = df.rename(columns={"target": "toxicity"}).dropna(subset=["comment_text"])
y_all = (df[LABELS] >= 0.5).astype(int)
lengths = df["comment_text"].str.len()
metrics["rows"] = len(df)
metrics["positive_rate"] = y_all.mean().round(4).to_dict()
metrics["median_length_chars"] = int(lengths.median())
metrics["rows_with_identity_labels"] = int(df[IDENTITIES].notna().all(axis=1).sum())
print(json.dumps({k: metrics[k] for k in ["rows", "positive_rate", "median_length_chars", "rows_with_identity_labels"]}, indent=1))

fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
(y_all.mean() * 100).sort_values().plot.barh(ax=ax[0], color="#4c72b0")
ax[0].set_xlabel("% of comments labelled positive")
ax[0].set_title("How rare each category is")
lengths.clip(upper=1000).plot.hist(bins=50, ax=ax[1], color="#4c72b0")
ax[1].set_xlabel("Comment length (characters, capped at 1,000)")
ax[1].set_title("Comment lengths")
plt.tight_layout(); plt.savefig(OUT / "explore.png", dpi=120); plt.show()

# %% [markdown]
# ## 2. Split and baseline model
# 80% train, 10% validation (to choose thresholds), 10% test (touched once, for every reported number). One TF-IDF vocabulary, one logistic regression per category.

# %%
idx_train, idx_tmp = train_test_split(df.index, test_size=0.2, random_state=SEED, stratify=y_all["toxicity"])
idx_val, idx_test = train_test_split(idx_tmp, test_size=0.5, random_state=SEED, stratify=y_all.loc[idx_tmp, "toxicity"])
vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=200_000, sublinear_tf=True, strip_accents="unicode")
X_train = vec.fit_transform(df.loc[idx_train, "comment_text"])
X_val = vec.transform(df.loc[idx_val, "comment_text"])
X_test = vec.transform(df.loc[idx_test, "comment_text"])

models, p_val, p_test = {}, {}, {}
for label in LABELS:
    m = LogisticRegression(C=4, max_iter=1000, solver="liblinear")
    m.fit(X_train, y_all.loc[idx_train, label])
    models[label] = m
    p_val[label] = m.predict_proba(X_val)[:, 1]
    p_test[label] = m.predict_proba(X_test)[:, 1]

y_val, y_test = y_all.loc[idx_val], y_all.loc[idx_test]
metrics["test_rows"] = len(idx_test)
metrics["per_label"] = {
    l: {"roc_auc": round(roc_auc_score(y_test[l], p_test[l]), 4),
        "pr_auc": round(average_precision_score(y_test[l], p_test[l]), 4),
        "test_positives": int(y_test[l].sum())}
    for l in LABELS}
pd.DataFrame(metrics["per_label"]).T

# %% [markdown]
# ## 3. Precision, recall and two threshold policies
# A single "accuracy" number hides the real trade-off. Two policies, both chosen on the **validation** set and then measured on the **test** set:
# - **Auto-action:** the lowest threshold with ≥ 90% precision. Wrongly removing a comment is costly, so only act alone when very sure.
# - **Send to review:** the highest threshold that still catches ≥ 80% of toxic comments. Humans check everything above it.

# %%
def threshold_for(y, p, *, precision=None, recall=None):
    prec, rec, thr = precision_recall_curve(y, p)
    prec, rec = prec[:-1], rec[:-1]
    if precision is not None:
        ok = np.where(prec >= precision)[0]
        return float(thr[ok[0]]) if len(ok) else 1.0
    ok = np.where(rec >= recall)[0]
    return float(thr[ok[-1]])

def at(y, p, t):
    pred = p >= t
    tp = int((pred & (y == 1)).sum())
    return {"threshold": round(t, 4), "precision": round(tp / max(pred.sum(), 1), 4),
            "recall": round(tp / max((y == 1).sum(), 1), 4), "flagged_share": round(float(pred.mean()), 4)}

yv, yt, pv, pt = y_val["toxicity"].values, y_test["toxicity"].values, p_val["toxicity"], p_test["toxicity"]
t_auto = threshold_for(yv, pv, precision=0.90)
t_review = threshold_for(yv, pv, recall=0.80)
metrics["policies"] = {"auto_action": at(yt, pt, t_auto), "send_to_review": at(yt, pt, t_review)}
print(json.dumps(metrics["policies"], indent=1))

prec, rec, _ = precision_recall_curve(yt, pt)
plt.figure(figsize=(5.5, 4))
plt.plot(rec, prec, color="#4c72b0", label=f"Test PR curve (PR-AUC {metrics['per_label']['toxicity']['pr_auc']:.2f})")
for name, m, c in [("Auto-action", metrics["policies"]["auto_action"], "#c44e52"), ("Send to review", metrics["policies"]["send_to_review"], "#dd8452")]:
    plt.scatter(m["recall"], m["precision"], color=c, zorder=3, label=f"{name} (t={m['threshold']:.2f})")
plt.xlabel("Recall: share of toxic comments caught"); plt.ylabel("Precision: share of flags that are right")
plt.title("Toxicity: precision vs recall"); plt.legend(loc="lower left"); plt.grid(alpha=.3)
plt.tight_layout(); plt.savefig(OUT / "precision_recall.png", dpi=120); plt.show()

# %% [markdown]
# ## 4. Simulated review queue
# Apply both policies to the test set as if it were one day of new comments: auto-action above the auto threshold, humans review the band in between, the rest is published.

# %%
auto = pt >= t_auto
review = (pt >= t_review) & ~auto
allow = ~(auto | review)
toxic = yt == 1
queue = {
    "comments": int(len(pt)),
    "auto_actioned": int(auto.sum()), "auto_actioned_wrongly": int((auto & ~toxic).sum()),
    "sent_to_review": int(review.sum()), "review_share": round(float(review.mean()), 4),
    "toxic_in_review": int((review & toxic).sum()),
    "published": int(allow.sum()), "toxic_missed": int((allow & toxic).sum()),
    "toxic_caught_share": round(float(((auto | review) & toxic).sum() / toxic.sum()), 4),
}
metrics["review_queue"] = queue
pd.Series(queue)

# %% [markdown]
# ## 5. Estimating prevalence with a confidence interval
# Trust & safety teams report "what share of content is toxic" from a small human-labelled random sample, not from the classifier's flags. Here: 1,000 random test comments act as the "labelled sample"; the Wilson 95% interval is compared with the true test share and with the share the model flags.

# %%
def wilson(k, n, z=1.96):
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half

rng = np.random.default_rng(SEED)
n = min(1000, len(yt))
k = int(yt[rng.choice(len(yt), n, replace=False)].sum())
lo, hi = wilson(k, n)
metrics["prevalence"] = {"sample_size": n, "sample_estimate": round(k / n, 4), "ci95_low": round(lo, 4), "ci95_high": round(hi, 4),
                         "true_test_share": round(float(yt.mean()), 4),
                         "share_flagged_at_review_threshold": round(float((pt >= t_review).mean()), 4)}
metrics["prevalence"]

# %% [markdown]
# ## 6. Fairness: false-positive rate by identity group
# Do non-toxic comments that **mention** a group get flagged more often? (Annotators marked which identities each comment mentions, on a subset of rows.) Measured at the send-to-review threshold, on test rows with identity labels, for groups with at least 200 non-toxic test comments.

# %%
test = df.loc[idx_test].assign(y=yt, p=pt)
labelled = test[test[IDENTITIES].notna().all(axis=1)]
nontoxic = labelled[labelled.y == 0]
overall_fpr = float((nontoxic.p >= t_review).mean())
rows = []
for g in IDENTITIES:
    sub = labelled[labelled[g] >= 0.5]
    nt = sub[sub.y == 0]
    if len(nt) >= MIN_GROUP:
        rows.append({"group": g, "nontoxic_comments": len(nt), "fpr": round(float((nt.p >= t_review).mean()), 4),
                     "subgroup_auc": round(roc_auc_score(sub.y, sub.p), 4) if sub.y.nunique() == 2 else None})
fair = pd.DataFrame(rows).sort_values("fpr")
metrics["fairness"] = {"threshold": round(t_review, 4), "overall_fpr_identity_labelled": round(overall_fpr, 4), "groups": rows}

plt.figure(figsize=(6.5, 0.35 * len(fair) + 1.2))
plt.barh(fair.group.str.replace("_", " "), fair.fpr * 100, color="#4c72b0")
plt.axvline(overall_fpr * 100, color="#c44e52", ls="--", label=f"All identity-labelled comments ({overall_fpr:.1%})")
plt.xlabel("% of NON-toxic comments flagged for review"); plt.title("False-positive rate when a comment mentions a group")
plt.legend(loc="lower right"); plt.tight_layout(); plt.savefig(OUT / "fairness_fpr.png", dpi=120); plt.show()
fair

# %% [markdown]
# ## 7. Save the model and the numbers

# %%
joblib.dump({"vectorizer": vec, "models": models, "labels": LABELS,
             "thresholds": {"auto_action": t_auto, "send_to_review": t_review}}, OUT / "model" / "model.joblib", compress=3)
(OUT / "metrics.json").write_text(json.dumps(metrics, indent=1))
print(json.dumps(metrics, indent=1))

# %% [markdown]
# ## What I learned
# _To be written by Malav._
