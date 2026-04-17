from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = "facebook/nllb-200-distilled-600M"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

LANG_MAP = {
    "English": "eng_Latn",
    "Hindi": "hin_Deva",
    "Gujarati": "guj_Gujr"
}

def split_text(text, max_length=600):
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start+max_length])
        start += max_length
    return chunks


def translate_text(text, target_language):
    target_code = LANG_MAP[target_language]
    tokenizer.src_lang = "eng_Latn"   # Whisper gives English

    chunks = split_text(text, max_length=600)
    final_translation = ""

    for chunk in chunks:
        inputs = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True)

        tokens = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids(target_code),
            max_length=512,
            no_repeat_ngram_size=3,
            repetition_penalty=1.2,
            temperature=0.7
        )

        translated_chunk = tokenizer.decode(tokens[0], skip_special_tokens=True)
        final_translation += translated_chunk + " "

    return final_translation.strip()

