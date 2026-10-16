# %% [markdown]
# # Content moderation classifier on Civil Comments
# **Content warning:** this dataset contains real toxic comments. This notebook never prints comment text; it shows only counts, scores and charts.
#
# A portfolio project by [Malav Naik](https://malavnaik99.github.io). Public data only. This is a portfolio demo, **not a production moderation system**.
#
# **Data:** [Jigsaw Unintended Bias in Toxicity Classification](https://www.kaggle.com/competitions/jigsaw-unintended-bias-in-toxicity-classification) (Civil Comments). The data and the comment text are released under CC0.
#
# **Plan:** explore → TF-IDF + logistic regression baseline → DistilBERT → precision/recall and two threshold policies → simulated review queue → prevalence estimate with a confidence interval → false-positive rates by identity group.

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
# severe_toxicity is left out: only ~0.002% of comments, ~3 in the whole test set, too few to measure.
LABELS = ["toxicity", "obscene", "identity_attack", "insult", "threat"]
QUICK = False  # the run workflow flips this to True for a 5-minute check run on a small sample
IDENTITIES = ["male", "female", "transgender", "homosexual_gay_or_lesbian", "christian", "jewish", "muslim",
              "black", "white", "asian", "latino", "psychiatric_or_mental_illness"]
metrics = {}

# %% [markdown]
# ## 1. Explore
# Each comment was rated by several people; a label's value is the share of raters who said yes. As in the competition, a comment counts as positive when that share is ≥ 0.5.

# %%
df = pd.read_csv(DATA, usecols=["id", "comment_text", "target", *LABELS[1:], *IDENTITIES])
df = df.rename(columns={"target": "toxicity"}).dropna(subset=["comment_text"])
if QUICK:
    df = df.sample(20_000, random_state=SEED)
    MIN_GROUP = 10
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

models, p_val, p_test = {}, {}, {}  # baseline scores per label
for label in LABELS:
    m = LogisticRegression(C=4, max_iter=1000, solver="liblinear")
    m.fit(X_train, y_all.loc[idx_train, label])
    models[label] = m
    p_val[label] = m.predict_proba(X_val)[:, 1]
    p_test[label] = m.predict_proba(X_test)[:, 1]


# %% [markdown]
# ## 3. Modern model: DistilBERT on Kaggle's free GPU
# A small transformer reads words in context ("I am a gay man" vs an insult), which a bag of words cannot. Fine-tuned for one epoch on a random 300,000 training comments (the free GPU's time limit), one output per category. Validation and test sets are exactly the same as for the baseline.

# %%
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME, MAX_LEN, BATCH = "distilbert-base-uncased", 128, 64
torch.manual_seed(SEED)  # repeatable weights init and batch order
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)
tok = AutoTokenizer.from_pretrained(MODEL_NAME)
bert = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=len(LABELS), problem_type="multi_label_classification").to(device)

def encode(idx):
    e = tok(df.loc[idx, "comment_text"].tolist(), truncation=True, max_length=MAX_LEN, padding="max_length", return_tensors="pt")
    return e["input_ids"], e["attention_mask"]

bert_train = pd.Index(np.random.default_rng(SEED).choice(idx_train, min(300_000, len(idx_train)), replace=False))
ids, mask = encode(bert_train)
train_dl = DataLoader(TensorDataset(ids, mask, torch.tensor(y_all.loc[bert_train, LABELS].values, dtype=torch.float32)), batch_size=BATCH, shuffle=True)
opt = torch.optim.AdamW(bert.parameters(), lr=3e-5)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-5, total_steps=len(train_dl), pct_start=0.06)
scaler = torch.cuda.amp.GradScaler(enabled=device == "cuda")
bert.train()
for step, (i, m, y) in enumerate(train_dl):
    with torch.autocast(device, enabled=device == "cuda"):
        loss = bert(input_ids=i.to(device), attention_mask=m.to(device), labels=y.to(device)).loss
    opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
    if step % 500 == 0:
        print(f"step {step}/{len(train_dl)} loss {loss.item():.4f}")

@torch.no_grad()
def bert_scores(idx):
    bert.eval(); out = []
    i_all, m_all = encode(idx)
    for i, m in DataLoader(TensorDataset(i_all, m_all), batch_size=256):
        with torch.autocast(device, enabled=device == "cuda"):
            out.append(torch.sigmoid(bert(input_ids=i.to(device), attention_mask=m.to(device)).logits.float()).cpu().numpy())
    return np.vstack(out)

bv, bt = bert_scores(idx_val), bert_scores(idx_test)
b_val = {l: bv[:, k] for k, l in enumerate(LABELS)}
b_test = {l: bt[:, k] for k, l in enumerate(LABELS)}

# %% [markdown]
# ## 4. Scores per category (test set)

# %%
y_val, y_test = y_all.loc[idx_val], y_all.loc[idx_test]
metrics["test_rows"] = len(idx_test)
metrics["bert_train_rows"] = len(bert_train)
MODELS = {"baseline": (p_val, p_test), "distilbert": (b_val, b_test)}
metrics["per_label"] = {
    name: {l: {"roc_auc": round(roc_auc_score(y_test[l], pt_[l]), 4), "pr_auc": round(average_precision_score(y_test[l], pt_[l]), 4),
               "test_positives": int(y_test[l].sum())} for l in LABELS}
    for name, (_, pt_) in MODELS.items()}
pd.concat({n: pd.DataFrame(v).T for n, v in metrics["per_label"].items()}, axis=1)

# %% [markdown]
# ## 5. Two threshold policies, a review queue and prevalence
# A single "accuracy" number hides the real trade-off. Two policies, chosen on the **validation** set and measured on the **test** set:
# - **Auto-action:** the lowest threshold with ≥ 90% precision. Wrongly removing a comment is costly, so only act alone when very sure.
# - **Send to review:** the highest threshold that still catches ≥ 80% of toxic comments. Humans check everything in between.
#
# Then the test set is treated as one day of new comments to size the human review queue.

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

yv, yt = y_val["toxicity"].values, y_test["toxicity"].values
toxic = yt == 1
thresholds, metrics["policies"], metrics["review_queue"] = {}, {}, {}
for name, (pv_, pt_) in MODELS.items():
    pv, pt = pv_["toxicity"], pt_["toxicity"]
    t_auto, t_review = threshold_for(yv, pv, precision=0.90), threshold_for(yv, pv, recall=0.80)
    thresholds[name] = {"auto_action": t_auto, "send_to_review": t_review}
    metrics["policies"][name] = {"auto_action": at(yt, pt, t_auto), "send_to_review": at(yt, pt, t_review)}
    auto = pt >= t_auto
    review = (pt >= t_review) & ~auto
    allow = ~(auto | review)
    metrics["review_queue"][name] = {
        "comments": int(len(pt)), "auto_actioned": int(auto.sum()), "auto_actioned_wrongly": int((auto & ~toxic).sum()),
        "sent_to_review": int(review.sum()), "review_share": round(float(review.mean()), 4), "toxic_in_review": int((review & toxic).sum()),
        "published": int(allow.sum()), "toxic_missed": int((allow & toxic).sum()),
        "toxic_caught_share": round(float(((auto | review) & toxic).sum() / toxic.sum()), 4)}
print(json.dumps({"policies": metrics["policies"], "review_queue": metrics["review_queue"]}, indent=1))

plt.figure(figsize=(6, 4.2))
for (name, (_, pt_)), c in zip(MODELS.items(), ["#4c72b0", "#55a868"]):
    prec, rec, _ = precision_recall_curve(yt, pt_["toxicity"])
    plt.plot(rec, prec, color=c, label=f"{name} (PR-AUC {metrics['per_label'][name]['toxicity']['pr_auc']:.2f})")
    for pol, mk in [("auto_action", "o"), ("send_to_review", "s")]:
        m = metrics["policies"][name][pol]
        plt.scatter(m["recall"], m["precision"], color=c, marker=mk, zorder=3)
plt.xlabel("Recall: share of toxic comments caught"); plt.ylabel("Precision: share of flags that are right")
plt.title("Toxicity: precision vs recall (● auto-action, ■ send to review)"); plt.legend(loc="lower left"); plt.grid(alpha=.3)
plt.tight_layout(); plt.savefig(OUT / "precision_recall.png", dpi=120); plt.show()

# %% [markdown]
# **Prevalence.** Trust & safety teams report "what share of content is toxic" from a small human-labelled random sample, not from the classifier's flags. Here 1,000 random test comments act as that sample; the Wilson 95% interval is compared with the true share and with the share each model flags.

# %%
def wilson(k, n, z=1.96):
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half

n = min(1000, len(yt))
k = int(yt[np.random.default_rng(SEED).choice(len(yt), n, replace=False)].sum())
lo, hi = wilson(k, n)
metrics["prevalence"] = {"sample_size": n, "sample_estimate": round(k / n, 4), "ci95_low": round(lo, 4), "ci95_high": round(hi, 4),
                         "true_test_share": round(float(yt.mean()), 4),
                         "share_flagged_at_review_threshold": {name: metrics["policies"][name]["send_to_review"]["flagged_share"] for name in MODELS}}
metrics["prevalence"]

# %% [markdown]
# ## 6. Fairness: false-positive rate by identity group
# Do non-toxic comments that **mention** a group get flagged more often? Annotators marked which identities each comment mentions, on a subset of rows. Measured at each model's send-to-review threshold, on test rows with identity labels, for groups with at least 200 non-toxic test comments.

# %%
test = df.loc[idx_test].assign(y=yt)
has_ids = test[IDENTITIES].notna().all(axis=1).values
metrics["fairness"] = {}
for name, (_, pt_) in MODELS.items():
    t = thresholds[name]["send_to_review"]
    lab = test[has_ids].assign(p=pt_["toxicity"][has_ids])
    nontoxic = lab[lab.y == 0]
    rows = []
    for g in IDENTITIES:
        sub = lab[lab[g] >= 0.5]
        nt = sub[sub.y == 0]
        if len(nt) >= MIN_GROUP:
            rows.append({"group": g, "nontoxic_comments": len(nt), "fpr": round(float((nt.p >= t).mean()), 4),
                         "subgroup_auc": round(roc_auc_score(sub.y, sub.p), 4) if sub.y.nunique() == 2 else None})
    metrics["fairness"][name] = {"threshold": round(t, 4), "overall_fpr_identity_labelled": round(float((nontoxic.p >= t).mean()), 4), "groups": rows}

fair = pd.DataFrame({name: {r["group"]: r["fpr"] for r in v["groups"]} for name, v in metrics["fairness"].items()}).sort_values("baseline")
ax = (fair * 100).plot.barh(figsize=(7, 0.45 * len(fair) + 1.5), color=["#4c72b0", "#55a868"])
for name, c in [("baseline", "#4c72b0"), ("distilbert", "#55a868")]:
    ax.axvline(metrics["fairness"][name]["overall_fpr_identity_labelled"] * 100, color=c, ls="--")
ax.set_yticklabels([g.replace("_", " ") for g in fair.index])
ax.set_xlabel("% of NON-toxic comments flagged for review (dashed: all identity-labelled comments)")
ax.set_title("False-positive rate when a comment mentions a group")
plt.tight_layout(); plt.savefig(OUT / "fairness_fpr.png", dpi=120); plt.show()
fair

# %% [markdown]
# ## 7. Save the models and the numbers

# %%
joblib.dump({"vectorizer": vec, "models": models, "labels": LABELS, "thresholds": thresholds["baseline"]},
            OUT / "model" / "model.joblib", compress=3)
bert.save_pretrained(OUT / "distilbert"); tok.save_pretrained(OUT / "distilbert")
(OUT / "distilbert" / "thresholds.json").write_text(json.dumps(thresholds["distilbert"]))
(OUT / "metrics.json").write_text(json.dumps(metrics, indent=1))
print(json.dumps(metrics, indent=1))

# %% [markdown]
# ## What I learned
# **What surprised me most:** harmless comments that simply *mention* a group get flagged far more often. With the simple model, half of the harmless comments mentioning "Black" were sent to review (50%), against 13% of harmless comments overall. The modern model cut that to 39% against 9.5%, but it didn't make the gap go away. A single headline score (0.96 out of 1) hides this completely, so I would always break results down by group before trusting a moderation model.
#
# **What I'd do next:** tackle that bias directly. I'd add more harmless training examples that mention these groups, track the false-positive rate for each group separately, and only call the model "better" when those gaps shrink.
#
# *Malav Naik*
