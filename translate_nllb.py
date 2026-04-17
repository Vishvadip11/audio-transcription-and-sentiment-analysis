import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load once (global)
MODEL_NAME = "facebook/nllb-200-distilled-600M"

# language codes (Supported NLLB Mapping)
LANG_MAP = {
    "english": "eng_Latn",
    "hindi": "hin_Deva",
    "gujarati": "guj_Gujr",
    "tamil": "tam_Taml",
    "telugu": "tel_Telu",
    "marathi": "mar_Deva",
    "bengali": "ben_Beng",
    "punjabi": "pan_Guru",
    "kannada": "kan_Knda",
    "malayalam": "mal_Mlym",
    "urdu": "urd_Arab",      
    "french": "fra_Latn",
    "spanish": "spa_Latn",
    "german": "deu_Latn",
}

# Determine device (CPU or GPU)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load model and tokenizer once
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(DEVICE)

def translate_text(text, source_lang, target_langs):
    # Pre-check for noise/empty text
    if not text or len(text.strip()) < 2:
        return {tgt_lang.lower(): "" for tgt_lang in target_langs}

    results = {}
    # Get the NLLB code for the source language (fallback to English)
    src_code = LANG_MAP.get(source_lang.lower(), "eng_Latn")

    for tgt_lang in target_langs:
        tgt_code = LANG_MAP.get(tgt_lang.lower(), "eng_Latn")

        # --- OPTIMIZATION: Same Language Bypass ---
        if src_code == tgt_code:
            results[tgt_lang.lower()] = text.strip()
            continue

        try:
            # Set-up for source language
            tokenizer.src_lang = src_code
            
            # Tokenize
            inputs = tokenizer(text, return_tensors="pt").to(DEVICE)

            # Generate with premium parameters to prevent truncation and loops
            translated_tokens = model.generate(
                **inputs,
                forced_bos_token_id=tokenizer.lang_code_to_id[tgt_code],
                max_new_tokens=256,   # Allow enough expansion for each chunk
                num_beams=5,          # High intelligence mode
                early_stopping=True,  # Stop once we have a clear conclusion
                no_repeat_ngram_size=3,
                repetition_penalty=1.1, # Discourage repetitive phrases
                length_penalty=1.0    # Encourage neutral length
            )

            # Decode
            output = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
            results[tgt_lang.lower()] = output.strip()
        
        except Exception as e:
            results[tgt_lang.lower()] = f"[Translation Error: {str(e)}]"

    return results