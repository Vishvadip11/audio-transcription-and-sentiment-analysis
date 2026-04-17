from whisper_transcribe import model, LANGUAGE_PROMPTS
from text_utils import clean_text
import whisper

def transcribe_audio(audio_path, language=None):
    try:
        # Dynamic Prompt based on language (English fallback)
        prompt = LANGUAGE_PROMPTS.get(language, LANGUAGE_PROMPTS["en"])

        # Detect or Force language
        result = model.transcribe(
            audio_path,
            task="transcribe",
            language=language,
            fp16=False,
            verbose=False,
            beam_size=5,
            initial_prompt=prompt,
            temperature=(0.0, 0.2, 0.4, 0.6),
            no_speech_threshold=0.6,
            logprob_threshold=-1.0,
            compression_ratio_threshold=2.4, 
            condition_on_previous_text=False
        )

        text = result["text"].strip()
        text = clean_text(text)

        if text == "":
            return "No speech detected in audio.", "English"

        # Get detected/confirmed language name
        src_code = result.get("language", "en")
        lang_name = whisper.tokenizer.LANGUAGES.get(src_code, "english").title()

        return text, lang_name

    except Exception as e:
        return f"Error while processing audio: {str(e)}", "English"
