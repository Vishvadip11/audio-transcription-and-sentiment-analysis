---
title: Audio Transcription Sentiment
emoji: 🎤
colorFrom: red
colorTo: red
sdk: streamlit
python_version: "3.10"
app_file: app.py
pinned: false
short_description: Multilingual Audio Transcriber and Analyzer
license: mit
---

# Multilingual Audio Transcriber & Analyzer

A Streamlit app that converts audio to text, detects the spoken language, translates the transcript, summarizes it, and performs tone analysis.

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Transformers-orange?style=for-the-badge)

## Features

- Upload MP3 or WAV audio files.
- Detect the spoken language automatically with Whisper.
- Transcribe multilingual audio.
- Translate output into English, Hindi, or Gujarati.
- Generate an English summary and translate it to the selected output language.
- Analyze transcript tone with TextBlob.
- Run in Streamlit locally, on Streamlit Community Cloud, or in Docker.

## Tech Stack

- Speech-to-text: [OpenAI Whisper](https://github.com/openai/whisper)
- Translation: [NLLB-200 Distilled 600M](https://huggingface.co/facebook/nllb-200-distilled-600M)
- Summarization: [BART Large CNN](https://huggingface.co/facebook/bart-large-cnn)
- Sentiment: [TextBlob](https://textblob.readthedocs.io/en/dev/)
- UI: [Streamlit](https://streamlit.io/)
- Audio conversion: `ffmpeg`

## Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/Vishvadip11/audio-transcription-and-sentiment-analysis.git
cd audio-transcription-and-sentiment-analysis
```

### 2. Install FFmpeg

- Windows: install from [ffmpeg.org](https://ffmpeg.org/) or `choco install ffmpeg`
- Linux: `sudo apt update && sudo apt install ffmpeg`
- macOS: `brew install ffmpeg`

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
streamlit run app.py
```

## Deployment

### Option 1: Streamlit Community Cloud

This repo is ready for Streamlit Cloud deployment:

- `app.py` is the entrypoint
- `requirements.txt` installs Python packages
- `packages.txt` installs `ffmpeg`
- `.streamlit/config.toml` enables cloud-friendly startup

Steps:

1. Push this project to GitHub.
2. Open [share.streamlit.io](https://share.streamlit.io/).
3. Create a new app and select this repository.
4. Set the main file path to `app.py`.
5. Click deploy.

Notes:

- The first startup can take time because Whisper, NLLB, and BART models may download on the server.
- Free-tier machines can feel slow for longer audio because several ML models run in sequence.

### Option 2: Docker

This repo also includes a `Dockerfile` for platforms like Render, Railway, Azure, or any VPS with Docker support.

Build and run locally:

```bash
docker build -t multilingual-audio-transcriber .
docker run -p 8501:8501 multilingual-audio-transcriber
```

Open `http://localhost:8501`

## Project Structure

```text
app.py
transcribe.py
whisper_transcribe.py
translate_nllb.py
summarize.py
sentiment.py
text_utils.py
convert.py
assets/
requirements.txt
packages.txt
Dockerfile
```

## Notes

- Heavy models are loaded lazily and cached to reduce startup failures during deployment.
- The app currently stores uploaded audio in the local `audio/` folder while a session is active.

## License

Distributed under the MIT License.
