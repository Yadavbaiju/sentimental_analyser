from transformers import pipeline

_clf = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
)

def hf_predict(text: str) -> str:
    # truncation avoids errors on texts longer than 512 tokens
    return _clf(text, truncation=True, max_length=512)[0]["label"]

def hf_predict_batch(texts: list[str], batch_size: int = 32) -> list[str]:
    out = _clf(texts, batch_size=batch_size, truncation=True, max_length=512)
    return [o["label"] for o in out]

if __name__ == "__main__":
    print(hf_predict("I love this phone!"))