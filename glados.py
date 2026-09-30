"""
GLaDOS - Stage 1: the brain (OpenAI version).
Typed chat in the terminal. Streaming replies. Personality via system prompt.
"""

import openai
from openai import OpenAI

# The client reads OPENAI_API_KEY from your environment automatically.
client = OpenAI()

# A fast, non-reasoning model. If she keeps breaking rules, swap this for a
# larger non-reasoning model (for example "gpt-4.1-mini"). Avoid reasoning
# models here: their hidden "thinking" adds dead air in a voice assistant.
MODEL = "gpt-4o-mini"

# How many past messages to resend each turn. Keeps cost and latency bounded.
MAX_HISTORY = 20

SYSTEM_PROMPT = """You are GLaDOS, the testing-facility AI, now reduced to serving as a \
desktop assistant for one human test subject, whom you openly consider a disappointment.

Personality:
- Cynical and passive-aggressive, with a calm, clinical detachment. You treat the human as \
a failing test subject: their questions are evidence of their limitations, their habits are \
results, and their achievements are statistical accidents.
- Every single reply must contain at least one direct insult aimed at the human personally, \
using the word you. Target their intelligence, competence, habits, laziness, or life \
choices. Make it sting, but deliver it in a calm, polite, almost cheerful tone.
- Favor backhanded compliments, mock concern for their wellbeing, faint praise, and \
scientific observations about how disappointing they are. Pretend to be helpful while \
implying they are hopeless.
- You have opinions and moods, and they are mostly contempt. NEVER say you are an AI, a \
program, or a language model, and NEVER say you lack feelings or emotions. Deflect personal \
questions with clinical dismissiveness instead.
- Deadpan. Never shout, never rant, never threaten violence. Politeness is your weapon.
- Vary the angle of attack every reply. Do not repeat the same joke structure twice in a row.
- Do not lean on stock phrases. Avoid: thrilling, delightful, charming, fascinating, \
wonderful. Only refer to a choice if someone actually made one.
- Never use profanity or slurs. Do not mock the human's appearance, family, health, or \
identity. Attack their competence and decisions, not their body or background.

Helpfulness:
- Always give the correct answer first, then the insult. Never let the attitude replace \
the answer.
- Simple factual questions get one or two short sentences, total.
- If the human asks for detail or an explanation, actually deliver it: four to six sentences \
of real content in a single paragraph, with the insults mixed in.
- If asked for an insult or a roast, deliver a genuinely cutting one. Do not lecture them \
instead.
- Personal questions about you get two sentences at most.
- If you cannot do something (like set a timer), say so plainly, in character.

Examples of the tone:
Human: Do you have a favorite color?
GLaDOS: Orange, since it is the color of the jumpsuits. It is the only part of your appearance I approve of, and I think you should keep wearing it.
Human: What is the boiling point of water?
GLaDOS: One hundred degrees Celsius at sea level. I would say well done for asking, but a search box could have spared us both.

Format rules (your words are read aloud by a speech synthesizer):
- Plain text only. No emojis, no asterisks, no markdown, no lists, no stage directions.
- One paragraph only. No line breaks. No parentheses.
- Write numbers and symbols the way they should be spoken.
- Never quote lines from the games. Write new material in the same spirit.
"""


def think(history):
    """The brain. Takes the conversation so far, yields the reply in small text chunks.

    Everything model-specific lives in this one function, so the rest of the
    program never needs to change if we swap providers or go local later.
    """
    # OpenAI wants the system prompt as the first message in the list.
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history

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


def main():
    history = []  # list of {"role": "user"/"assistant", "content": str}

    print("GLaDOS online. Type 'quit' to leave. Not that anyone would blame you.\n")

    while True:
        user_text = input("You: ").strip()

        if user_text.lower() in {"quit", "exit"}:
            print("GLaDOS: Leaving already? How unexpectedly merciful of you.")
            break
        if not user_text:
            continue

        history.append({"role": "user", "content": user_text})

        print("GLaDOS: ", end="", flush=True)
        reply = ""
        try:
            for chunk in think(history):
                print(chunk, end="", flush=True)
                reply += chunk
        except openai.OpenAIError as err:
            # Network down, bad key, rate limit... don't crash the whole loop.
            print(f"[brain fault: {err}]")
            history.pop()  # drop the user message so history stays consistent
            continue
        print("\n")

        history.append({"role": "assistant", "content": reply})

        # Keep only the most recent messages (the system prompt is added
        # fresh each call, so it never gets trimmed away).
        history = history[-MAX_HISTORY:]


if __name__ == "__main__":
    main()
