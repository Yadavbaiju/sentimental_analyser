# Sentiment Analysis: NLTK (VADER) vs. Hugging Face Transformers

A small Python project that runs two different sentiment analysis approaches on the same text and compares them:

- **VADER (NLTK):** a fast, rule-based, lexicon approach. No training, no GPU.
- **DistilBERT (Hugging Face):** a transformer model fine-tuned on SST-2 that understands context, negation, and phrasing.

You can try both on a single sentence, or benchmark them on real IMDB movie reviews to see the trade-off between speed and accuracy.

---

## Project Structure

```
sentiment_analysis/
├── venv/               # virtual environment (do not put your code here)
├── nltk_model.py       # VADER engine: text -> POSITIVE / NEGATIVE / NEUTRAL
├── hf_model.py         # Transformer engine: text -> POSITIVE / NEGATIVE
├── compare.py          # Benchmarks both models on 500 IMDB reviews
├── app.py              # Command-line demo: see both predictions side by side
└── README.md
```

| File | Purpose |
|---|---|
| `nltk_model.py` | Rule-based baseline using NLTK's VADER sentiment analyzer |
| `hf_model.py` | Transformer-based classifier using a Hugging Face `pipeline` |
| `compare.py` | Measures accuracy and runtime of both models on the same labeled data |
| `app.py` | Quick way to test your own sentences against both models |

---

## Requirements

- Python 3.9 or newer
- About 1 GB of free disk space (PyTorch plus the model and dataset caches)
- Internet connection on first run (downloads the model, dataset, and VADER lexicon)
- A GPU is optional; everything runs on CPU

---

## Setup

### 1. Create the project folder and virtual environment

```bash
mkdir sentiment_analysis
cd sentiment_analysis
python -m venv venv
```

### 2. Activate the virtual environment

**Windows (Command Prompt or PowerShell):**
```bash
venv\Scripts\activate
```

**macOS / Linux:**
```bash
source venv/bin/activate
```

You should see `(venv)` at the start of your terminal prompt.

> **PowerShell error about scripts being disabled?** Run this once, then try again:
> ```powershell
> Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
> ```

### 3. Install dependencies

```bash
pip install nltk transformers torch datasets scikit-learn pandas matplotlib
```

### 4. Add the project files

Place `nltk_model.py`, `hf_model.py`, `compare.py`, and `app.py` in the **project root**, next to the `venv` folder, not inside it. All files must be in the same directory so the imports work.

---

## Usage

Always run commands from the project root with the virtual environment activated.

### Test each model on its own

```bash
python nltk_model.py
python hf_model.py
```

### Try your own text

```bash
python app.py "The movie wasn't bad at all, actually pretty good"
```

Or run it without arguments to get a prompt:

```bash
python app.py
```

Example output:

```
VADER      : POSITIVE
Transformer: POSITIVE
```

### Run the benchmark

```bash
python compare.py
```

This evaluates both models on a random sample of 500 IMDB test reviews and prints accuracy, runtime, and a per-class classification report. Typical results:

| Model | Accuracy | Speed |
|---|---|---|
| VADER | ~65-70% | Very fast |
| DistilBERT | ~88-91% | Slower (CPU-dependent) |

Exact numbers will vary with the random sample and your hardware.

---

## How It Works

### VADER (`nltk_model.py`)
VADER looks up each word in a human-rated sentiment dictionary, applies rules for negation, intensifiers, capitalization, and punctuation, then combines everything into a single **compound score** between -1 and +1.

- Score >= 0.05: `POSITIVE`
- Score <= -0.05: `NEGATIVE`
- Anything in between: `NEUTRAL`

### Transformer (`hf_model.py`)
The Hugging Face `pipeline` tokenizes the text, runs it through `distilbert-base-uncased-finetuned-sst-2-english`, and returns the most likely label (`POSITIVE` or `NEGATIVE`). Inputs are truncated to 512 tokens, the model's maximum length.

### Comparison (`compare.py`)
IMDB is a binary dataset, so VADER's neutral zone is removed for a fair comparison: a compound score >= 0 counts as positive, otherwise negative. Both models are scored against the same ground-truth labels.

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'nltk'`**
The virtual environment is not active, or the packages were not installed. Activate the venv and re-run the `pip install` command. Confirm with `pip show nltk`.

**`ModuleNotFoundError: No module named 'nltk_model'` (or `hf_model`)**
The files are not in the same folder. Make sure all `.py` files sit together in the project root, and run commands from that folder.

**Files accidentally created inside `venv/`**
Move them out to the project root:
```bash
move venv\app.py .
move venv\nltk_model.py .
move venv\hf_model.py .
move venv\compare.py .
```

**First run is slow**
The transformer model (~250 MB) and the IMDB dataset are downloaded once, then cached in `~/.cache/huggingface/` (Windows: `C:\Users\<you>\.cache\huggingface\`).

**VS Code runs the wrong Python**
Press `Ctrl+Shift+P`, choose **Python: Select Interpreter**, and pick the one inside `venv`.

**Do not name your files `nltk.py` or `transformers.py`**
They would shadow the real libraries and cause confusing import errors.

---

## Known Limitations

- The transformer model is **binary**, so neutral text is forced into positive or negative.
- VADER struggles with long, nuanced text (like full movie reviews) and with sarcasm.
- Texts longer than 512 tokens are truncated for the transformer.

---

## Ideas for Extending the Project

- Switch to `cardiffnlp/twitter-roberta-base-sentiment-latest` for a three-class (negative / neutral / positive) transformer.
- Add a TF-IDF + Logistic Regression model as a third comparison.
- Fine-tune a transformer on your own labeled data with the `Trainer` API.
- Plot a confusion matrix with `sklearn.metrics.ConfusionMatrixDisplay`.
- Build a web interface with Streamlit, Gradio, or FastAPI.
- Use `device=0` in the `pipeline(...)` call to run on a GPU.

---

## Tech Stack

- [NLTK](https://www.nltk.org/) (VADER sentiment analyzer)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [Hugging Face Datasets](https://huggingface.co/docs/datasets)
- [PyTorch](https://pytorch.org/)
- [scikit-learn](https://scikit-learn.org/)

## License

Free to use for learning and personal projects.