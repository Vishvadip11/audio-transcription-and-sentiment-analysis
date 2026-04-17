import re

def filter_hallucinations(text):
    """
    Removes common Whisper hallucinations especially during silence.
    """
    hallucinations = [
        r"thank you for watching",
        r"thanks for watching",
        r"please subscribe",
        r"subscribe to our channel",
        r"subtitles by",
        r"amara.org",
        r"watching for more",
        r"see you in the next one",
        r"i'll see you in the next",
        r"thank you very much for joining us",
        r"be sure to like and subscribe",
        r"sharing this video"
    ]
    
    for pattern in hallucinations:
        # Match as whole phrases, case-insensitive
        text = re.sub(pattern, "", text, flags=re.IGNORECASE).strip()
    
    # 2. Remove orphan punctuation that often remains after hallucination removal
    # e.g., "Thank you for watching. Hello" -> ". Hello" -> "Hello"
    text = re.sub(r'^\s*[,.!।?-]+\s*', '', text) # Remove leading punctuation
    text = re.sub(r'\s*[,.!।?-]+\s*$', '', text) # Remove trailing punctuation
    
    # 3. Remove any stray "..." or "---" often left behind
    text = re.sub(r'[\.\-]{2,}', '', text)
    return text.strip()

def normalize_sentences(text):
    """
    Fixes casing, double spaces, and basic punctuation.
    """
    if not text: return ""
    
    # 1. Normalize spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 2. Fix Punctuation (e.g., "word , word" -> "word, word")
    # Added Urdu comma (،) to normalization
    text = re.sub(r'\s+([,.?!।۔،])', r'\1', text)
    
    # 3. Ensure sentence Case ONLY for Latin scripts or safe starters
    # Added Urdu full stop (۔) to sentence split
    sentences = re.split(r'(?<=[\.\!\?।۔])\s+', text)
    cleaned_sentences = []
    for s in sentences:
        if s:
            s = s.strip()
            # Capitalize first letter if it's a letter (safe for Latin, no-op for others)
            if s and s[0].isalpha():
                s = s[0].upper() + s[1:]
            cleaned_sentences.append(s)
    
    return " ".join(cleaned_sentences).strip()

def deduplicate_text(text, max_repeat=2):
    """
    Detects and removes repeating sequences (single words or phrases).
    Increased max_repeat=2 to prevent over-aggressive removal of valid repeats.
    """
    words = text.split()
    if not words: return ""
    
    # Check for repeating n-grams (from 1 word up to 12 words)
    for n in range(1, 13):
        i = 0
        while i <= len(words) - n * 2:
            chunk = " ".join(words[i:i+n]).lower()
            # Check if next segments match the current chunk
            j = i + n
            count = 1
            while j <= len(words) - n and " ".join(words[j:j+n]).lower() == chunk:
                count += 1
                j += n
            
            if count > max_repeat:
                # Keep up to 2 instances if they exist
                words = words[:i+n*max_repeat] + words[j:]
            else:
                i += 1
    return " ".join(words)

def clean_text(text):
    if not text: return ""
    # 1. Remove whisper special tokens
    text = re.sub(r'<\|.*?\|>', '', text)
    # 2. Remove hallucinations
    text = filter_hallucinations(text)
    # 3. Deduplicate (Slightly relaxed to preserve valid repeated emphasis)
    text = deduplicate_text(text, max_repeat=2)
    # 4. Normalize casing/punctuation
    text = normalize_sentences(text)
    return text

def split_text(text, max_chars=180):
    """
    Splits text into chunks at sentence boundaries (., ?, !, ।, ۔) to maintain context.
    Reduced max_chars to 180 to ensure better quality for NLLB models.
    """
    if not text: return []
    # Added Urdu full stop (۔) to split regex
    sentences = re.split(r'(?<=[.?!।۔])\s+', text)
    
    chunks, current = [], []
    current_len = 0
    
    for s in sentences:
        # If a single sentence is still too long, split it by commas as fallback
        # Added Urdu comma (،) to sub-split regex
        if len(s) > max_chars:
            sub_parts = re.split(r'(?<=[,،])\s+', s)
            for sp in sub_parts:
                if current_len + len(sp) > max_chars and current:
                    chunks.append(" ".join(current).strip())
                    current = []
                    current_len = 0
                current.append(sp)
                current_len += len(sp)
        else:
            if current_len + len(s) > max_chars and current:
                chunks.append(" ".join(current).strip())
                current = []
                current_len = 0
            current.append(s)
            current_len += len(s)
        
    if current:
        chunks.append(" ".join(current).strip())
    return chunks
