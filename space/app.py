import json

import gradio as gr
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

LABELS = ["toxicity", "obscene", "identity_attack", "insult", "threat"]
tok = AutoTokenizer.from_pretrained("model")
model = AutoModelForSequenceClassification.from_pretrained("model").eval()
T = json.load(open("model/thresholds.json"))


@torch.no_grad()
def score(text, review_threshold):
    text = (text or "")[:1000]  # keep inputs short on the free CPU
    if not text.strip():
        return {}, ""
    p = torch.sigmoid(model(**tok(text, truncation=True, max_length=128, return_tensors="pt")).logits)[0].tolist()
    scores = dict(zip(LABELS, p))
    t = scores["toxicity"]
    if t >= T["auto_action"]:
        decision = f"🔴 Auto-action (toxicity {t:.2f} ≥ {T['auto_action']:.2f})"
    elif t >= review_threshold:
        decision = f"🟠 Send to human review (toxicity {t:.2f} ≥ {review_threshold:.2f})"
    else:
        decision = f"🟢 Publish (toxicity {t:.2f} < {review_threshold:.2f})"
    return scores, decision


demo = gr.Interface(
    fn=score,
    inputs=[gr.Textbox(label="Comment", lines=3, max_lines=6),
            gr.Slider(0.05, 0.8, value=round(T["send_to_review"], 2), step=0.01, label="Send-to-review threshold")],
    outputs=[gr.Label(label="Scores (0–1)", num_top_classes=5), gr.Textbox(label="What a review queue would do")],
    examples=[["Thanks for the thoughtful article, I learned a lot.", 0.22],
              ["I disagree with this policy, but I see your point.", 0.22],
              ["As a gay man, I found this piece really moving.", 0.22],
              ["This is the dumbest idea I have read all week.", 0.22]],
    title="Moderation classifier demo",
    description="**Portfolio demo, not a production moderation system.** DistilBERT fine-tuned on the public Civil Comments data. "
                "Lower the threshold to catch more toxic comments (and wrongly flag more harmless ones). "
                "Known limit: harmless comments that mention some identities are flagged 2–4× more often than average. "
                "By [Malav Naik](https://malavnaik99.github.io).",
    allow_flagging="never",
)
demo.launch()
