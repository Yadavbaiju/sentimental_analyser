import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download("vader_lexicon", quiet=True)
_sia = SentimentIntensityAnalyzer()

def vader_predict(text: str) -> str:
    score = _sia.polarity_scores(text)["compound"]  # -1 to +1
    if score >= 0.05:
        return "POSITIVE"
    if score <= -0.05:
        return "NEGATIVE"
    return "NEUTRAL"

if __name__ == "__main__":
    for t in ["I love this phone!", "This is the worst service ever.", "It arrived on Tuesday."]:
        print(t, "->", vader_predict(t))