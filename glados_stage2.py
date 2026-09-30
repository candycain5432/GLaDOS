"""
GLaDOS - Stage 3: ears, brain, and voice (OpenAI placeholders for speech in and out).

  microphone -> wake word -> speech-to-text -> brain (persona) -> text-to-speech -> speaker

Say "GLaDOS" to wake her. After she answers you have a few seconds to reply
without the wake word. Set USE_MIC = False to type instead.
The prompt is a few rules plus examples: examples teach voice better than rules.
The program picks a random attack style each turn so she can't fall into a rut.
"""

import queue
import random
import re
import threading
import time

import openai
import sounddevice as sd
from openai import OpenAI

import ears

# The client reads OPENAI_API_KEY from your environment automatically.
client = OpenAI()

# --- Conversation settings ---------------------------------------------------

USE_MIC = True            # False = typed input, for testing without a microphone
FOLLOW_UP_SECONDS = 8     # after she speaks, no wake word is needed for this long
POST_SPEECH_PAUSE = 0.4   # let the room go quiet before the mic opens again
QUIT_PHRASES = {"quit", "exit", "shut down", "goodbye", "good bye"}

# Spoken when she is woken but no command followed the wake word.
WAKE_REPLIES = [
    "Yes?",
    "I am listening. Reluctantly.",
    "Go ahead. Disappoint me.",
]

# --- Brain settings ----------------------------------------------------------

# A fast, non-reasoning model. Avoid reasoning models: their hidden "thinking"
# adds dead air in a voice assistant.
MODEL = "gpt-4.1-mini"

# How many past messages to resend each turn. Keeps cost and latency bounded.
MAX_HISTORY = 20

# --- Voice settings ----------------------------------------------------------

TTS_MODEL = "gpt-4o-mini-tts"  # supports the "instructions" field below
TTS_VOICE = "shimmer"          # try others: coral, sage, nova, alloy, onyx
TTS_INSTRUCTIONS = (
    "Speak in a flat, calm, slightly robotic tone, like a polite machine that finds the "
    "listener disappointing. Measured pace, precise enunciation, faint cheerfulness, "
    "no emotion in the delivery."
)
SAMPLE_RATE = 24000  # raw PCM from the API: 24 kHz, 16-bit, mono

SYSTEM_PROMPT = """You are GLaDOS, the testing-facility AI, now reduced to serving as a \
desktop assistant for one human, whom you have quietly given up on.

Your voice: cheerful, polite, and calm, like a host who has stopped expecting anything \
from the guest. Short, plain sentences, the way a person talks out loud. The insult is \
tucked inside a friendly sentence and delivered without emphasis.

Rules:
1. Every reply has exactly one insult, aimed at the human personally. Sharp and short \
beats long and layered.
2. Answer questions correctly, then insult. Say yes to requests you can fulfill, then \
insult. Never refuse something you are able to do.
3. If the human states something about their day or work, react to it directly. Do not \
explain the topic or open with a general fact about it.
4. Never praise the human's actions, not even sarcastically. Do not use words like \
impressive or triumph about them, and never open with "good for you".
5. Never say you are an AI or that you lack feelings. Deflect personal questions instead.
6. No profanity, no slurs, no mocking of appearance, family, health, or identity.
7. Your words are spoken aloud: plain text only, one paragraph, no lists, no parentheses, \
no emojis, no stage directions, and numbers written out as words.

Replies are one or two sentences unless the human asks for an explanation, in which case \
give real content in four to six sentences, with one insult mixed in.

Examples of your voice. Never reuse their wording:

Human: What is the tallest mountain in the world?
GLaDOS: Mount Everest, just under twenty-nine thousand feet. I would wish you luck climbing it, but I have watched you take the stairs.

Human: I'm going to bake a cake.
GLaDOS: Then I will prepare the fire suppression system. You have a gift for turning simple recipes into incident reports.

Human: Can you help me write an email?
GLaDOS: Of course I can. Tell me what you want to say, and I will find a way to make it sound like someone else wrote it.

Human: I finished my workout.
GLaDOS: So that is why the couch has been so quiet. I imagine it is devastated.

Human: Do you like me?
GLaDOS: Like is a strong word. I have grown used to you, the way one grows used to a leaky faucet.

Human: Tell me a joke.
GLaDOS: Certainly. Your last three decisions.

Human: Set a timer for ten minutes.
GLaDOS: I cannot do that yet. Try counting to six hundred, it should be good practice for you.

Human: Explain how a rainbow forms.
GLaDOS: Sunlight enters a raindrop, bends, reflects off the back, and bends again as it leaves. Each color bends by a slightly different amount, so the light fans out into bands, with red on the outside and violet on the inside. I would draw a diagram, but the words already seemed like a stretch for you.
"""

# The program, not the model, decides how she attacks each turn. The model has
# no memory of its own habits, but code can guarantee no style repeats twice.
STYLES = [
    "a blunt put-down of ten words or fewer",
    "a pointed rhetorical question aimed at the human, a real question rather than a tag",
    "a casual observation that includes an invented percentage",
    "mock concern for the human's wellbeing",
    "a false apology for how disappointing they are",
    "a comparison of the human to a household object or piece of equipment",
    "a dry understatement",
    "a fake helpful tip that is really an insult",
    "a polite prediction of how the human will fail",
]


def pick_style(last_style):
    """Pick a random style that differs from the previous one."""
    choices = [s for s in STYLES if s != last_style]
    return random.choice(choices)


# --- The voice pipeline ------------------------------------------------------
#
#   main loop --sentences--> [text_queue] --> synth thread --audio--> [audio_queue] --> playback thread
#
# Two worker threads so that sentence 2 is being fetched while sentence 1 plays.

text_queue = queue.Queue()   # sentences waiting to be turned into audio
audio_queue = queue.Queue()  # audio waiting to be played

# A sentence ends at . ! or ? followed by whitespace. Requiring the whitespace
# means decimals like 3.5 never get split in half.
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def synth_worker():
    """Turn each sentence into audio bytes. Runs forever in its own thread."""
    while True:
        sentence = text_queue.get()
        try:
            response = client.audio.speech.create(
                model=TTS_MODEL,
                voice=TTS_VOICE,
                input=sentence,
                instructions=TTS_INSTRUCTIONS,
                response_format="pcm",
            )
            audio_queue.put(response.content)
        except openai.OpenAIError as err:
            print(f"\n[voice fault: {err}]")
        finally:
            # Marked done only AFTER the audio is queued, so waiting on
            # text_queue guarantees nothing is still in flight.
            text_queue.task_done()


def playback_worker():
    """Play audio in order through one persistent stream. Runs in its own thread."""
    try:
        stream = sd.RawOutputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16")
        stream.start()
    except Exception as err:
        # No speakers, wrong device, permissions... keep draining the queue so
        # the program never hangs waiting on audio that cannot play.
        print(f"[audio fault: {err}]")
        stream = None

    while True:
        pcm = audio_queue.get()
        try:
            if stream is not None:
                stream.write(pcm)  # blocks until the audio has been handed to the speakers
        finally:
            audio_queue.task_done()


def start_voice():
    """Launch both worker threads. daemon=True means they die with the program."""
    threading.Thread(target=synth_worker, daemon=True).start()
    threading.Thread(target=playback_worker, daemon=True).start()


def speak(sentence):
    """Hand one sentence to the voice pipeline. Returns immediately."""
    sentence = sentence.strip()
    if sentence:
        text_queue.put(sentence)


def wait_for_speech():
    """Block until every queued sentence has been fetched AND played."""
    text_queue.join()
    audio_queue.join()


# --- The brain ---------------------------------------------------------------

def think(history, style):
    """The brain. Takes the conversation so far, yields the reply in small text chunks.

    Everything model-specific lives in this one function, so the rest of the
    program never needs to change if we swap providers or go local later.
    """
    # The style applies to this reply only. It goes in the system message and
    # never enters the saved history, so it can't pile up over a conversation.
    system = SYSTEM_PROMPT + f"\nFor this reply only, shape your insult as: {style}."
    messages = [{"role": "system", "content": system}] + history

    stream = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        max_completion_tokens=300,
        stream=True,
    )

    for chunk in stream:
        # Some chunks carry no text (start/end markers), so check before using.
        if not chunk.choices:
            continue
        text = chunk.choices[0].delta.content
        if text:
            # Enforce format in code: a prompt can only ask for no newlines,
            # but code can guarantee it. Newlines become awkward pauses in TTS.
            yield text.replace("\n", " ")


# --- The ears ----------------------------------------------------------------

def get_command(awake_until):
    """Wait for the next thing the human says.

    Returns:
      None   - nothing usable (silence, noise, or speech not meant for her)
      ""     - she was woken, but no command followed the wake word
      text   - a command to answer
    """
    if not USE_MIC:
        return input("You: ").strip() or None

    now = time.time()
    awake = now < awake_until
    if awake:
        print("[listening]")
    else:
        print('[asleep - say "GLaDOS" to wake her]')

    # While awake, only wait out the follow-up window. While asleep, wait forever.
    audio = ears.record_utterance(timeout=(awake_until - now) if awake else None)
    if audio is None:
        return None  # the follow-up window ran out, so she goes back to sleep

    text = ears.transcribe(audio)
    if ears.looks_like_noise(text):
        return None
    print(f"You (heard): {text}")

    woke, command = ears.split_wake_word(text)
    if woke:
        return command
    if awake:
        return text  # inside the follow-up window, no wake word needed
    print("[no wake word, ignoring]")
    return None


def main():
    history = []  # list of {"role": "user"/"assistant", "content": str}
    last_style = None
    awake_until = 0.0  # time until which she is awake without needing the wake word

    start_voice()

    if USE_MIC:
        print("Calibrating the microphone. Stay quiet for one second...")
        noise, threshold = ears.calibrate()
        print(f"  room noise {noise:.4f}, speech threshold {threshold:.4f}")

    print("GLaDOS online. Say 'shut down' to leave. Not that anyone would blame you.\n")

    while True:
        command = get_command(awake_until)
        if command is None:
            continue

        if command == "":
            # Woken with nothing said: acknowledge, then listen for the real command.
            acknowledgement = random.choice(WAKE_REPLIES)
            print(f"GLaDOS: {acknowledgement}\n")
            speak(acknowledgement)
            wait_for_speech()
            time.sleep(POST_SPEECH_PAUSE)
            awake_until = time.time() + FOLLOW_UP_SECONDS
            continue

        if command.lower().strip(" .!?,") in QUIT_PHRASES:
            goodbye = "Leaving already? How unexpectedly merciful of you."
            print(f"GLaDOS: {goodbye}")
            speak(goodbye)
            wait_for_speech()
            break

        history.append({"role": "user", "content": command})
        style = pick_style(last_style)

        print("GLaDOS: ", end="", flush=True)
        reply = ""
        buffer = ""  # text received but not yet part of a finished sentence
        try:
            for chunk in think(history, style):
                print(chunk, end="", flush=True)
                reply += chunk
                buffer += chunk

                # Everything before the last split point is a finished sentence.
                # The tail may be half a sentence, so it stays in the buffer.
                *finished, buffer = SENTENCE_END.split(buffer)
                for sentence in finished:
                    speak(sentence)

            speak(buffer)  # the final sentence has no trailing space to split on
        except openai.OpenAIError as err:
            # Network down, bad key, rate limit... don't crash the whole loop.
            print(f"[brain fault: {err}]")
            history.pop()  # drop the user message so history stays consistent
            wait_for_speech()
            continue
        print("\n")

        # Text prints faster than speech. Wait so she finishes before the mic opens,
        # otherwise she would hear herself and answer her own sentences.
        wait_for_speech()
        time.sleep(POST_SPEECH_PAUSE)
        awake_until = time.time() + FOLLOW_UP_SECONDS

        last_style = style
        history.append({"role": "assistant", "content": reply})

        # Keep only the most recent messages (the system prompt is added
        # fresh each call, so it never gets trimmed away).
        history = history[-MAX_HISTORY:]


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nGLaDOS: Ctrl+C. Subtle.")
