"""
scripts/gpt2_sentiment/label_vocab_sentiment.py

Label each token in the GPT-2 vocabulary as positive, negative, or neutral
using VADER sentiment scores. Saves to data/gpt2_sentiment_vocab.json.

Usage
-----
    uv run python scripts/gpt2_sentiment/label_vocab_sentiment.py

Extra dependencies (install once):
    pip install transformers vaderSentiment
"""
import json
from pathlib import Path

from transformers import GPT2Tokenizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

OUTPUT_PATH = Path("data/gpt2_sentiment_vocab.json")

POSITIVE_THRESHOLD = 0.1
NEGATIVE_THRESHOLD = -0.1


def label_vocabulary(
    tokenizer: GPT2Tokenizer,
    analyzer: SentimentIntensityAnalyzer,
) -> dict[int, str]:
    """
    Score every token in the GPT-2 vocab with VADER and return a
    {token_id: label} dict where label is "positive", "negative", or "neutral".
    """
    vocab = tokenizer.get_vocab()  # {token_str: token_id}
    labels: dict[int, str] = {}

    for _, token_id in vocab.items():
        decoded = tokenizer.decode([token_id]).strip()
        if not decoded:
            labels[token_id] = "neutral"
            continue
        compound = analyzer.polarity_scores(decoded)["compound"]
        if compound > POSITIVE_THRESHOLD:
            labels[token_id] = "positive"
        elif compound < NEGATIVE_THRESHOLD:
            labels[token_id] = "negative"
        else:
            labels[token_id] = "neutral"

    return labels


def main() -> None:
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    analyzer = SentimentIntensityAnalyzer()

    print("Scoring GPT-2 vocabulary with VADER...")
    labels = label_vocabulary(tokenizer, analyzer)

    counts = {"positive": 0, "negative": 0, "neutral": 0}
    for label in labels.values():
        counts[label] += 1

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Keys must be strings in JSON; convert int keys to str
    with open(OUTPUT_PATH, "w") as f:
        json.dump({str(token_id): label for token_id, label in labels.items()}, f)

    print(f"Saved {len(labels):,} token labels to {OUTPUT_PATH}")
    print(f"  positive: {counts['positive']:,}")
    print(f"  negative: {counts['negative']:,}")
    print(f"  neutral:  {counts['neutral']:,}")

    # Show a few examples
    print("\nSample positive tokens:")
    positive_examples = [
        tokenizer.decode([token_id])
        for token_id, label in labels.items()
        if label == "positive"
    ][:10]
    print("  " + ", ".join(repr(t) for t in positive_examples))

    print("Sample negative tokens:")
    negative_examples = [
        tokenizer.decode([token_id])
        for token_id, label in labels.items()
        if label == "negative"
    ][:10]
    print("  " + ", ".join(repr(t) for t in negative_examples))


if __name__ == "__main__":
    main()
