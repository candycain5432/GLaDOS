"""
net_check.py - find out why OpenAI calls fail with "Connection error".
Run:  python net_check.py
It tries a chat call, then a speech-to-text call with a test tone, and prints
the underlying cause of any failure.
"""

import io
import wave

import numpy as np
import openai
from openai import OpenAI

client = OpenAI()


def show(err):
    """Print an error and every error that caused it, since the top one hides the reason."""
    print(f"  FAILED: {type(err).__name__}: {err}")
    cause = err.__cause__
    while cause is not None:
        print(f"    caused by: {type(cause).__name__}: {cause}")
        cause = cause.__cause__


def make_tone_wav():
    """One second of a 440 Hz tone as an in-memory WAV file."""
    t = np.arange(16000) / 16000
    pcm = (0.3 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(pcm.tobytes())
    return buf.getvalue()


print("1. Chat call (the brain uses this)...")
try:
    reply = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": "Say ok."}],
        max_completion_tokens=5,
    )
    print("  OK:", reply.choices[0].message.content)
except openai.OpenAIError as err:
    show(err)

wav_bytes = make_tone_wav()
for model in ["gpt-4o-mini-transcribe", "whisper-1"]:
    print(f"2. Speech-to-text call with {model} (a tone, so the text may be empty)...")
    try:
        result = client.audio.transcriptions.create(
            model=model,
            file=("tone.wav", wav_bytes, "audio/wav"),
        )
        print("  OK:", repr(result.text))
    except openai.OpenAIError as err:
        show(err)
