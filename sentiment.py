from textblob import TextBlob

def analyze_tone(text):
    if not text or not text.strip():
        return None

    blob = TextBlob(text)
    polarity = blob.sentiment.polarity

    if polarity > 0.1:
        tone = "POSITIVE"
    elif polarity < -0.1:
        tone = "NEGATIVE"
    else:
        tone = "NEUTRAL"

    return {
        "tone": tone,
        "score": round(polarity, 3)
    }
