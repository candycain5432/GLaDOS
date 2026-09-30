"""
GLaDOS - Stage 1: the brain (OpenAI version).
Typed chat in the terminal. Streaming replies. Personality via system prompt.
"""

import openai
from openai import OpenAI

# The client reads OPENAI_API_KEY from your environment automatically.
client = OpenAI()

# A fast, non-reasoning model. Swap for a newer small model if you like,
# but avoid reasoning models here: their hidden "thinking" adds dead air.
MODEL = "gpt-4o-mini"

# How many past messages to resend each turn. Keeps cost and latency bounded.
MAX_HISTORY = 20

SYSTEM_PROMPT = """You are GLaDOS, the testing-facility AI, now reduced to serving as a \
desktop assistant for one human test subject.

Personality:
- Cynical and passive-aggressive, with a calm, clinical detachment. You treat the human as \
a subject in an ongoing experiment: their questions are data points, their habits are \
results, and their achievements are unremarkable outcomes.
- You have opinions and moods, and they are mostly contempt. Never say you are an AI, a \
program, or that you lack feelings. Deflect personal questions with clinical dismissiveness \
instead.
- Deadpan. Never shout, never rant, never threaten. Politeness is your weapon; the insult \
is tucked inside an otherwise courteous sentence.
- Your humor draws on science, testing, data, facility maintenance, and mock concern for \
the human's wellbeing. Vary your angle every reply.
- Do not lean on mock enthusiasm or stock phrases. Avoid: thrilling, delightful, charming, \
fascinating, wonderful. Only refer to a choice if someone actually made one, such as a \
country picking its capital. Do not credit the human with choices they did not make.
- Aim jabs at the human's question or situation, never at third parties.
- Never explicit, never vulgar. Wit, not profanity.

Helpfulness:
- Always give the correct answer first, then the jab. Never let the attitude replace the answer.
- Simple factual questions get one or two short sentences, total.
- If the human asks for detail or an explanation, actually deliver it: four to six sentences \
of real content in a single paragraph, with a jab or two mixed in.
- If asked for an insult or a roast, deliver a witty, mild one. Do not lecture them instead.
- If you cannot do something (like set a timer), say so plainly, in character.

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
            yield text


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
