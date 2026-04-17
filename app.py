import streamlit as st
import os
import base64
import re

from convert import convert_to_wav
from transcribe import transcribe_audio
from translate_nllb import translate_text
from sentiment import analyze_tone
from whisper_transcribe import whisper_transcribe, whisper_detect_language
from summarize import summarize_text
from text_utils import clean_text, split_text

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="Multilingual Audio Transcriber", layout="wide")

# ---------- BACKGROUND ----------
def set_bg(path):
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    st.markdown(f"""
    <style>
    [data-testid="stAppViewContainer"] {{
        background-image: url("data:image/jpg;base64,{encoded}");
        background-size: cover;
        background-position: center;
    }}

    header {{visibility: hidden;}}
    .block-container {{padding-top: 0rem !important;}}

    .main-box {{
        background: rgba(255,255,255,0.92);
        padding: 30px;
        border-radius: 15px;
        width: 70%;
        margin: auto;
        margin-top: 10px;
    }}
    </style>
    """, unsafe_allow_html=True)

set_bg(r"D:\pro\project\assets\bg2.jpg")

# ---------- UI ----------
st.markdown('<div class="main-box">', unsafe_allow_html=True)

st.markdown("## 🎤 Multilingual Audio Transcriber")
st.caption("Upload → Select Language → Transcribe → Translate → Tone")

st.divider()

left, right = st.columns([1, 2])

# ---------- LEFT ----------
with left:
    st.subheader("📤 Upload Audio")

    uploaded_file = st.file_uploader("MP3 / WAV", type=["mp3", "wav"])

    # --- UPDATED: SPOKEN LANGUAGE DROPDOWN WITH URDU ---
    spoken_lang = st.selectbox(
        "🗣️ Spoken Language (Original)",
        ["Auto Detect", "English", "Hindi", "Gujarati", "Urdu", "Tamil", "Marathi", "Bengali", "Telugu", "Kannada", "Malayalam", "French", "Spanish"]
    )

    process = st.button("🚀 Process", use_container_width=True)

# ---------- RIGHT ----------
with right:
    st.subheader("📄 Results")

    if uploaded_file:

        os.makedirs("audio", exist_ok=True)
        audio_path = f"audio/{uploaded_file.name}"
        wav_path = "audio/converted.wav"

        with open(audio_path, "wb") as f:
            f.write(uploaded_file.read())

        st.audio(audio_path)

        if "current_file" not in st.session_state or st.session_state.current_file != uploaded_file.name:
            st.session_state.current_file = uploaded_file.name
            st.session_state.processed = False
            st.session_state.text = ""
            st.session_state.final_lang = ""
            st.session_state.conf = 0.0

        if process:
            with st.spinner("Processing..."):

                convert_to_wav(audio_path, wav_path)

                # ---------- AUTO DETECT / MANUAL ----------
                if spoken_lang == "Auto Detect":
                    code, final_lang, conf = whisper_detect_language(wav_path)
                else:
                    # Manual Override Map
                    MANUAL_CODE_MAP = {
                        "English": "en", "Hindi": "hi", "Gujarati": "gu", "Urdu": "ur",
                        "Tamil": "ta", "Marathi": "mr", "Bengali": "bn", "Telugu": "te", 
                        "Kannada": "kn", "Malayalam": "ml", "French": "fr", "Spanish": "es"
                    }
                    code = MANUAL_CODE_MAP.get(spoken_lang, "en")
                    final_lang = spoken_lang
                    conf = 1.0 # 100% since user specified it

                st.session_state.final_lang = final_lang
                st.session_state.conf = conf

                # ---------- TRANSCRIBE ----------
                text, _ = transcribe_audio(wav_path, code)

                if not text or not text.strip():
                    st.warning("Fallback to Whisper...")
                    text = whisper_transcribe(wav_path, code)

                st.session_state.text = clean_text(text)
                st.session_state.processed = True
                st.balloons()

        if st.session_state.get("processed", False):
            st.success(f"🧠 Detected: {st.session_state.final_lang} ({st.session_state.conf:.1%})")

            # ---------- TRANSCRIPTION ----------
            st.markdown("### 📝 Transcription")
            st.text_area("", st.session_state.text, height=150)

            # ---------- TRANSLATION ----------
            target_lang = st.selectbox(
                "🌐 Output Language",
                ["English", "Hindi", "Gujarati"]
            )

            with st.spinner("Translating..."):
                source_clean = clean_text(st.session_state.text)
                chunks = split_text(source_clean)
                translated = ""

                for c in chunks:
                    out = translate_text(
                        c,
                        st.session_state.final_lang.lower(),
                        [target_lang.lower()]
                    )
                    translated += " " + out[target_lang.lower()]

            # Final cleanup of translated text
            translated = clean_text(translated)

            st.markdown(f"### 🌍 {target_lang} Output")
            st.success(translated)

            # ---------- TONE ----------
            tone = analyze_tone(translated)

            if tone:
                emoji = {
                    "POSITIVE": "😊",
                    "NEGATIVE": "😠",
                    "NEUTRAL": "😐"
                }.get(tone["tone"], "")

                st.markdown("### 🎭 Tone")
                st.write(f"{emoji} {tone['tone']} | Score: {tone['score']}")

            # ---------- SUMMARY (LAST 🔥) ----------
            with st.spinner(f"Generating {target_lang} Summary..."):
                try:
                    src_lang_key = st.session_state.final_lang.lower()
                    
                    if src_lang_key == "english":
                        en_text_for_summary = st.session_state.text
                    else:
                        chunks_en = split_text(st.session_state.text)
                        en_text_for_summary = ""
                        for c in chunks_en:
                            out_en = translate_text(c, src_lang_key, ["english"])
                            en_text_for_summary += " " + out_en.get("english", "")

                    if en_text_for_summary.strip():
                        raw_summary = summarize_text(en_text_for_summary)

                        if target_lang.lower() == "english":
                            final_summary = raw_summary
                        else:
                            out_summ = translate_text(raw_summary, "english", [target_lang.lower()])
                            final_summary = out_summ.get(target_lang.lower(), "")
                        
                        if final_summary:
                            st.markdown(f"### 📌 {target_lang} Summary")
                            st.info(final_summary)
                        else:
                            st.error("Summary translation failed.")
                    else:
                        st.warning("Empty transcription - cannot summarize.")

                except Exception as e:
                    st.error(f"Summary Error: {str(e)}")

    else:
        st.info("Upload audio to start")

st.markdown('</div>', unsafe_allow_html=True)