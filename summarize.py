import re
from transformers import pipeline

# Use a specific local path or the model name
MODEL_PATH = "facebook/bart-large-cnn"

# Initialize the summarizer pipeline once
summarizer = pipeline(
    "summarization",
    model=MODEL_PATH,
    tokenizer=MODEL_PATH
)

def clean_text(text):
    """
    Basic text cleaning to remove multiple spaces and special characters.
    """
    text = re.sub(r'[\"\'`]', '', text)
    text = re.sub(r'[^\w\s\.\,]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def summarize_text(text):
    """
    Summarizes English text using BART. 
    Handles long text by chunking.
    """
    text = clean_text(text)

    # 🔴 Safety 1: very small text → no summary
    if not text or len(text.split()) < 30:
        return text

    # Chunking for BART (which has a 1024 token limit)
    max_chunk = 700   # roughly 700 characters to be safe
    chunks = [text[i:i + max_chunk] for i in range(0, len(text), max_chunk)]

    summary_parts = []

    for chunk in chunks:
        # 🔴 Safety 2: skip weak chunks
        if not chunk.strip() or len(chunk.split()) < 10:
            continue

        try:
            result = summarizer(
                chunk,
                max_length=150,
                min_length=40,
                do_sample=False
            )

            if result and isinstance(result, list) and "summary_text" in result[0]:
                summary_parts.append(result[0]["summary_text"])

        except Exception:
            # 🔴 Safety 3: skip failing chunk
            continue

        # 🔴 Safety 4: limit processing to avoid too long summary
        if len(summary_parts) >= 5:
            break

    # 🔴 Safety 5: fallback
    if not summary_parts:
        return text[:500]

    return " ".join(summary_parts)
