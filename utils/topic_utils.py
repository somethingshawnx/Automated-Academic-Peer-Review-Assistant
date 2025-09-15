import re

def normalize_topic(text: str) -> str:
    """
    Normalize a topic string so FAISS index and pipeline use consistent naming.
    Example: "A Longitudinal Sentiment Analysis!!!"
    → "a_longitudinal_sentiment_analysis"
    """
    if not text:
        return "general"
    # Lowercase
    text = text.lower()
    # Remove non-alphanumeric except spaces
    text = re.sub(r"[^a-z0-9\s]", "", text)
    # Collapse whitespace → underscore
    text = re.sub(r"\s+", "_", text.strip())
    return text or "general"
