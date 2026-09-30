"""
ears.py - GLaDOS Stage 3: the ears (OpenAI speech-to-text as a placeholder).

Pipeline:  microphone -> loudness check (local) -> record one utterance
           -> OpenAI speech-to-text -> wake word check on the transcript

Run this file directly to test the ears on their own:   python ears.py
"""

import collections
import io
import re
import time
import wave

import numpy as np
import openai
import sounddevice as sd
from openai import OpenAI

# The client reads OPENAI_API_KEY from your environment automatically.
client = OpenAI()

# --- Speech-to-text settings --------------------------------------------------

STT_MODEL = "gpt-4o-mini-transcribe"  # if this errors on your account, try "whisper-1"
STT_LANGUAGE = "en"
STT_PROMPT = "GLaDOS"  # a spelling hint so the wake word is written correctly

# --- Microphone settings ------------------------------------------------------

SAMPLE_RATE = 16000                       # 16 kHz mono is plenty for speech
BLOCK_MS = 30                             # we measure loudness in 30 ms blocks
BLOCK = SAMPLE_RATE * BLOCK_MS // 1000    # samples per block (480)

NOISE_MULTIPLIER = 3.0   # speech threshold = room noise x this...
MIN_THRESHOLD = 0.012    # ...but never below this (RMS on a 0 to 1 scale)

START_BLOCKS = 3         # loud blocks in a row before we decide speech has begun
MIN_LOUD_BLOCKS = 6      # fewer loud blocks than this = a click or cough, discard it
PREROLL_BLOCKS = 10      # keep 0.3 s from before speech began so the first syllable survives
END_SILENCE_BLOCKS = 27  # about 0.8 s of quiet ends the utterance
KEEP_TAIL_BLOCKS = 7     # trim the trailing silence down to about 0.2 s before uploading
MAX_BLOCKS = 500         # hard cap: 15 seconds per utterance

# Set by calibrate(). Until then, the floor value is used.
threshold = MIN_THRESHOLD

# --- Wake word ---------------------------------------------------------------

# Speech-to-text hears "GLaDOS" many ways: Glados, Gladys, Glad Os, Glay dos...
WAKE_RE = re.compile(
    r"\bglad(?:ys|is|us|ous)\b|\bglad[\s-]?d?os\b|\bglay[\s-]?d?os\b",
    re.IGNORECASE,
)
FILLER_RE = re.compile(r"^(?:hey|hi|hello|okay|ok)\b[\s,.:;!?-]*", re.IGNORECASE)
PUNCT = " ,.:;!?-"

# Phrases speech-to-text tends to invent out of noise.
HALLUCINATIONS = {
    "you",
    "bye",
    "thanks for watching",
    "thank you for watching",
    "subtitles by the amara.org community",
}


def _level(block):
    """Loudness of one block: root-mean-square, 0 (silent) to 1 (full scale)."""
    return float(np.sqrt(np.mean(block ** 2)))


def calibrate(seconds=1.0):
    """Measure the room's background noise and set the speech threshold from it.

    Returns (noise_level, threshold). Stay quiet while this runs.
    """
    global threshold
    blocks = int(seconds * 1000 / BLOCK_MS)
    levels = []
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", blocksize=BLOCK) as stream:
        for _ in range(blocks):
            data, _ = stream.read(BLOCK)
            levels.append(_level(data[:, 0]))

    # The 90th percentile ignores a single stray blip during calibration.
    noise = float(np.percentile(levels, 90))
    threshold = max(noise * NOISE_MULTIPLIER, MIN_THRESHOLD)

    if noise == 0.0:
        print(
            "WARNING: the microphone is returning pure silence. On a Mac, allow your\n"
            "terminal app under Privacy settings > Microphone, then restart the terminal."
        )
    return noise, threshold


def record_utterance(timeout=None, meter=False):
    """Block until the human says something, and return it as a float32 array.

    timeout: seconds to wait for speech to START. None waits forever.
             Returns None if nothing started in time.
    meter:   print a live loudness bar (used by the test mode).
    """
    preroll = collections.deque(maxlen=PREROLL_BLOCKS)  # rolling window of recent quiet
    speech = []
    speaking = False
    loud_run = 0      # consecutive loud blocks
    loud_total = 0    # all loud blocks in this utterance
    silent_blocks = 0
    started = time.time()
    tick = 0

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", blocksize=BLOCK) as stream:
        while True:
            data, _ = stream.read(BLOCK)
            block = data[:, 0]
            level = _level(block)
            loud = level > threshold

            if meter:
                tick += 1
                if tick % 3 == 0:
                    bar = "#" * min(40, int(level * 400))
                    print(f"\r  level {level:.3f} |{bar:<40}|", end="", flush=True)

            if not speaking:
                preroll.append(block)
                loud_run = loud_run + 1 if loud else 0
                if loud_run >= START_BLOCKS:
                    speaking = True
                    speech = list(preroll)  # lead-in plus the blocks that triggered us
                    loud_total = loud_run
                    silent_blocks = 0
                elif timeout is not None and time.time() - started > timeout:
                    if meter:
                        print()
                    return None
            else:
                speech.append(block)
                if loud:
                    loud_total += 1
                    silent_blocks = 0
                else:
                    silent_blocks += 1

                if silent_blocks >= END_SILENCE_BLOCKS or len(speech) >= MAX_BLOCKS:
                    if loud_total < MIN_LOUD_BLOCKS:
                        # A click or cough, not speech. Reset and keep listening.
                        speaking = False
                        speech = []
                        preroll.clear()
                        loud_run = 0
                        loud_total = 0
                        silent_blocks = 0
                        continue
                    trim = max(0, silent_blocks - KEEP_TAIL_BLOCKS)
                    if trim:
                        speech = speech[:-trim]
                    break

    if meter:
        print()
    return np.concatenate(speech)


def _to_wav_bytes(audio):
    """Float audio -> an in-memory WAV file, which is what the API accepts."""
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)  # 16-bit
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(pcm.tobytes())
    return buf.getvalue()


def transcribe(audio):
    """Send one utterance to OpenAI and return the text ("" if it failed)."""
    try:
        response = client.audio.transcriptions.create(
            model=STT_MODEL,
            file=("speech.wav", _to_wav_bytes(audio), "audio/wav"),
            language=STT_LANGUAGE,
            prompt=STT_PROMPT,
        )
        return response.text.strip()
    except openai.OpenAIError as err:
        print(f"\n[ears fault: {err}]")
        return ""


def looks_like_noise(text):
    """True for empty transcripts and the usual phrases invented out of silence."""
    cleaned = text.lower().strip(PUNCT)
    return len(cleaned) < 2 or cleaned in HALLUCINATIONS


def split_wake_word(text):
    """Find the wake word in a transcript.

    Returns (woke, command). command is everything said besides the wake word,
    so "GLaDOS, what time is it" and "what time is it, GLaDOS" both give
    "what time is it". If there is no wake word, command is the whole text.
    """
    match = WAKE_RE.search(text)
    if not match:
        return False, text.strip()

    before = text[: match.start()].strip(PUNCT)
    before = FILLER_RE.sub("", before).strip(PUNCT)
    after = text[match.end():].strip(PUNCT)
    return True, f"{before} {after}".strip()


# --- Stand-alone test: python ears.py ---------------------------------------

if __name__ == "__main__":
    print("Ears test. Stay quiet for one second while the room is measured...")
    noise, thr = calibrate()
    print(f"Room noise {noise:.4f}, speech threshold {thr:.4f}")
    print("Speak when ready. The bar shows your loudness. Ctrl+C to stop.\n")

    try:
        while True:
            audio = record_utterance(meter=True)
            print(f"  captured {len(audio) / SAMPLE_RATE:.1f} s, transcribing...")
            text = transcribe(audio)

            if looks_like_noise(text):
                print("  heard: (nothing usable)\n")
                continue

            woke, command = split_wake_word(text)
            print(f'  heard:     "{text}"')
            if woke:
                print(f'  wake word: YES   command: "{command}"\n')
            else:
                print("  wake word: no\n")
    except KeyboardInterrupt:
        print("\nEars off.")
