# %% [markdown]
# # Threshold sweep for the moderation sandbox on malavnaik.com
# **Content warning:** the data is real toxic comments. This notebook prints only counts and rates, never comment text.
#
# It re-scores the same 180,488 held-out test comments with the published DistilBERT model (the output of
# [the main notebook](https://www.kaggle.com/code/malavnaik/moderation-classifier-civil-comments)), with no retraining,
# then saves precision, recall, share flagged and false-positive rate by identity group at every threshold from 0.02 to 0.98.
# Before saving, it checks it reproduces the published numbers (ROC-AUC 0.9574 and the send-to-review results).

# %%
import json, os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Same data, labels, seed and split as the main notebook.
DATA = next(Path("/kaggle/input").rglob("jigsaw-unintended-bias*/train.csv"), None)
MODEL_DIR = next((p.parent for p in Path("/kaggle/input").rglob("thresholds.json")), None)
assert DATA and MODEL_DIR, [str(p) for p in Path("/kaggle/input").rglob("*")][:30]
OUT = Path("/kaggle/working")
SEED, MIN_GROUP = 42, 200
LABELS = ["toxicity", "obscene", "identity_attack", "insult", "threat"]
IDENTITIES = ["male", "female", "transgender", "homosexual_gay_or_lesbian", "christian", "jewish", "muslim",
              "black", "white", "asian", "latino", "psychiatric_or_mental_illness"]
df = pd.read_csv(DATA, usecols=["id", "comment_text", "target", *LABELS[1:], *IDENTITIES])
df = df.rename(columns={"target": "toxicity"}).dropna(subset=["comment_text"])
y_all = (df[LABELS] >= 0.5).astype(int)
idx_train, idx_tmp = train_test_split(df.index, test_size=0.2, random_state=SEED, stratify=y_all["toxicity"])
idx_val, idx_test = train_test_split(idx_tmp, test_size=0.5, random_state=SEED, stratify=y_all.loc[idx_tmp, "toxicity"])

# %%
device = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(MODEL_DIR)
bert = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device).eval()
enc = tok(df.loc[idx_test, "comment_text"].tolist(), truncation=True, max_length=128, padding="max_length", return_tensors="pt")
scores = []
with torch.no_grad():
    for i, m in DataLoader(TensorDataset(enc["input_ids"], enc["attention_mask"]), batch_size=256):
        with torch.autocast(device, enabled=device == "cuda"):
            scores.append(torch.sigmoid(bert(input_ids=i.to(device), attention_mask=m.to(device)).logits.float())[:, 0].cpu().numpy())
p = np.concatenate(scores)
y = y_all.loc[idx_test, "toxicity"].values

# %% [markdown]
# ## Check: does this reproduce the published results?

# %%
review_t = json.loads((MODEL_DIR / "thresholds.json").read_text())["send_to_review"]
auc = roc_auc_score(y, p)
flag = p >= review_t
check = {"test_rows": int(len(y)), "roc_auc": round(auc, 4), "review_threshold": round(review_t, 4),
         "review_precision": round(float((flag & (y == 1)).sum() / flag.sum()), 4),
         "review_recall": round(float((flag & (y == 1)).sum() / (y == 1).sum()), 4),
         "review_flagged_share": round(float(flag.mean()), 4)}
print(json.dumps(check, indent=1))
assert check["test_rows"] == 180488 and abs(check["roc_auc"] - 0.9574) < 0.0006, "does not reproduce the published model"

# %% [markdown]
# ## Sweep every threshold

# %%
test = df.loc[idx_test].assign(y=y, p=p)
lab = test[test[IDENTITIES].notna().all(axis=1)]
groups = [g for g in IDENTITIES if ((lab[g] >= 0.5) & (lab.y == 0)).sum() >= MIN_GROUP]
rows = []
for t in np.round(np.arange(0.02, 0.981, 0.01), 2):
    f = p >= t
    tp = int((f & (y == 1)).sum())
    row = {"t": float(t), "precision": round(tp / max(int(f.sum()), 1), 4), "recall": round(tp / int((y == 1).sum()), 4),
           "flagged": round(float(f.mean()), 4),
           "fpr_all": round(float((lab[lab.y == 0].p >= t).mean()), 4),
           "fpr": {g: round(float((lab[(lab[g] >= 0.5) & (lab.y == 0)].p >= t).mean()), 4) for g in groups}}
    rows.append(row)
out = {"source": "malavnaik/moderation-classifier-civil-comments (published DistilBERT), test split of Civil Comments",
       "check": check, "groups": groups,
       "group_sizes": {g: int(((lab[g] >= 0.5) & (lab.y == 0)).sum()) for g in groups}, "sweep": rows}
(OUT / "sweep.json").write_text(json.dumps(out, separators=(",", ":")))
print(len(rows), "thresholds; groups:", groups)
print(json.dumps([r for r in rows if r["t"] in (0.22, 0.5, 0.82)], indent=1))
