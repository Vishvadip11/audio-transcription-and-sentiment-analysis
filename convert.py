import shutil
import subprocess


def convert_to_wav(input_path, output_path):
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        raise RuntimeError(
            "FFmpeg is not installed or not available in PATH. "
            "Install FFmpeg to process uploaded audio files."
        )

    command = [
        ffmpeg_path,
        "-y",
        "-i",
        input_path,
        "-ac",
        "1",
        "-ar",
        "16000",
        output_path,
    ]

    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        error_message = result.stderr.strip() or "Unknown FFmpeg conversion error."
        raise RuntimeError(f"Audio conversion failed: {error_message}")
