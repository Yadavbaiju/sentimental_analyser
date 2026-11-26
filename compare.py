import time
from datasets import load_dataset
from sklearn.metrics import accuracy_score, classification_report
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

from hf_model import hf_predict_batch

nltk.download("vader_lexicon", quiet=True)
sia = SentimentIntensityAnalyzer()

# Small sample so it runs quickly on CPU
ds = load_dataset("imdb", split="test").shuffle(seed=42).select(range(500))
texts, labels = ds["text"], ["POSITIVE" if l == 1 else "NEGATIVE" for l in ds["label"]]

# VADER (binary: compound >= 0 -> positive)
t0 = time.time()
vader_preds = ["POSITIVE" if sia.polarity_scores(t)["compound"] >= 0 else "NEGATIVE" for t in texts]
vader_time = time.time() - t0

# Transformer
t0 = time.time()
hf_preds = hf_predict_batch(texts)
hf_time = time.time() - t0

print(f"VADER       acc={accuracy_score(labels, vader_preds):.3f}  time={vader_time:.1f}s")
print(f"Transformer acc={accuracy_score(labels, hf_preds):.3f}  time={hf_time:.1f}s\n")
print(classification_report(labels, hf_preds))