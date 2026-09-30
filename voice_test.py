"""Voice check: fetch one sentence of speech and play it. No brain involved."""

import sounddevice as sd
from openai import OpenAI

client = OpenAI()

response = client.audio.speech.create(
    model="gpt-4o-mini-tts",
    voice="shimmer",
    input="Voice check. If you can hear this, your speakers work, which is more than I can say for your judgment.",
    instructions=(
        "Speak in a flat, calm, slightly robotic tone, like a polite machine that finds the "
        "listener disappointing. Measured pace, precise enunciation, faint cheerfulness, "
        "no emotion in the delivery."
    ),
    response_format="pcm",  # raw 24 kHz, 16-bit, mono audio
)

with sd.RawOutputStream(samplerate=24000, channels=1, dtype="int16") as stream:
    stream.write(response.content)
