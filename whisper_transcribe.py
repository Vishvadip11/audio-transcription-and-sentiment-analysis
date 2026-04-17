import whisper
import numpy as np
from text_utils import clean_text

# Load a robust model for both detection and transcription once
model = whisper.load_model("small")  # multilingual model

# Native language prompts to steer Whisper toward correct scripts/style
LANGUAGE_PROMPTS = {
    "gu": "આ એક ગુજરાતી ટ્રાન્સક્રિપ્શન છે. તે સ્વચ્છ અને સચોટ છે.",
    "hi": "यह एक हिंदी प्रतिलेखन है। यह स्पष्ट और सटीक है।",
    "ta": "இது ஒரு தமிழ் டிரான்ஸ்கிரிப்ஷன் ஆகும். இது ஒரு தெளிவான ஆவணமாகும்.",
    "mr": "हा एक मराठी उतारा आहे. तो स्पष्ट आणि अचूक आहे.",
    "bn": "এটি একটি বাংলা ট্রান্সক্রিপশন। এটি স্পষ্ট এবং সঠিক।",
    "te": "ఇది ఒక తెలుగు ట్రాన్స్‌క్రిప్షన్. ఇది స్పష్టంగా ఉంది.",
    "kn": "ఇదు కన్నడ ప్రతిలిಪಿ. ఇదు స్పష్టವಾಗಿದೆ.",
    "ml": "ഇതൊരു മലയാളം ട്രാൻസ്ക്രിപ്ഷൻ ആണ്. ഇത് വ്യക്തമാണ്.",
    "ur": "یہ ایک اردو ٹرانسکرپشن ہے۔ یہ صاف اور درست ہے", # Added Urdu prompt
    "en": "This is a clean English transcription without special tags or symbols.",
    "fr": "Ceci est une transcription en français propre.",
    "es": "Esta is una transcripción en español limpia."
}

def whisper_detect_language(audio_path):
    audio = whisper.load_audio(audio_path)
    
    # Use 5 segments to "vote" on the language (start, 25%, 50%, 75%, end)
    total_len = len(audio)
    # Points to sample (ensure we don't go out of bounds)
    points = [0, total_len // 4, total_len // 2, (3 * total_len) // 4, max(0, total_len - 480000)]
    
    results = [] # Store (best_lang_code, confidence)
    for p in points:
        if p + 480000 <= total_len or p == 0:
            seg = whisper.pad_or_trim(audio[p:])
            mel = whisper.log_mel_spectrogram(seg).to(model.device)
            _, probs = model.detect_language(mel)
            best_lang = max(probs, key=probs.get)
            results.append((best_lang, probs[best_lang]))
    
    if not results:
        # Fallback to standard check if audio is tiny
        seg = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(seg).to(model.device)
        _, probs = model.detect_language(mel)
        best_lang_code = max(probs, key=probs.get)
        confidence = probs[best_lang_code]
    else:
        # Tally the votes with confidence weighting
        lang_scores = {}
        for code, conf in results:
            lang_scores[code] = lang_scores.get(code, 0) + conf
        
        best_lang_code = max(lang_scores, key=lang_scores.get)
        # Average confidence for the winning language across segments where it appeared
        winner_samples = [conf for code, conf in results if code == best_lang_code]
        confidence = sum(winner_samples) / len(winner_samples)

    # Get the valid language name from whisper
    lang_name = whisper.tokenizer.LANGUAGES.get(best_lang_code, "english").title()
    
    return best_lang_code, lang_name, confidence

def whisper_transcribe(audio_path, language_code=None):
    audio = whisper.load_audio(audio_path)

    if audio is None or len(audio) < 16000:
        return ""

    # Dynamic Prompt based on language (English fallback)
    prompt = LANGUAGE_PROMPTS.get(language_code, LANGUAGE_PROMPTS["en"])

    # Using multiple temperatures helps Whisper fallback gracefully if the audio is noisy
    result = model.transcribe(
        audio,
        task="transcribe",
        language=language_code,  # Force language routing for higher accuracy
        fp16=False,
        beam_size=5, # Higher intelligence mode for accurate Indian languages
        initial_prompt=prompt,
        temperature=(0.0, 0.2, 0.4, 0.6), 
        logprob_threshold=-1.0,
        no_speech_threshold=0.6,
        compression_ratio_threshold=2.4, # Prevent repetitive loops
        condition_on_previous_text=False
    )

    # Decode and clean
    text = result.get("text", "").strip()
    return clean_text(text)
