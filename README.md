# Content moderation classifier (Civil Comments)

> **Content warning:** the training data contains real toxic comments. This repo never prints comment text.

A portfolio project by [Malav Naik](https://malavnaik99.github.io): a toxicity classifier with the parts trust & safety teams actually use, which are threshold policies, a simulated review queue, a prevalence estimate with a confidence interval, and false-positive rates by identity group. Portfolio demo, not a production moderation system.

- [`kaggle/notebook.py`](kaggle/notebook.py): the whole analysis (converted to a Kaggle notebook and run on Kaggle's free CPU by [`run-notebook.yml`](.github/workflows/run-notebook.yml)).
- Data: [Jigsaw Unintended Bias in Toxicity Classification](https://www.kaggle.com/competitions/jigsaw-unintended-bias-in-toxicity-classification) (Civil Comments), CC0.

Run locally on any CSV with the same columns: `DATA=train.csv OUT=out python kaggle/notebook.py`
