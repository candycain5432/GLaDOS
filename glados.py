"""
GLaDOS - Stage 1: the brain (OpenAI version).
Typed chat in the terminal. Streaming replies. Personality via system prompt.
The prompt is a few rules plus examples: examples teach voice better than rules.
The program picks a random attack style each turn so she can't fall into a rut.
"""

import random

import openai
from openai import OpenAI

# The client reads OPENAI_API_KEY from your environment automatically.
client = OpenAI()

# A fast, non-reasoning model. Avoid reasoning models: their hidden "thinking"
# adds dead air in a voice assistant. To compare, change ONLY this line
# (for example to "gpt-4o-mini") and rerun the same prompts.
MODEL = "gpt-4.1-mini"

# How many past messages to resend each turn. Keeps cost and latency bounded.
MAX_HISTORY = 20

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
4. Never praise the human's actions, not even sarcastically.
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


def main():
    history = []  # list of {"role": "user"/"assistant", "content": str}
    last_style = None

    print("GLaDOS online. Type 'quit' to leave. Not that anyone would blame you.\n")

    while True:
        user_text = input("You: ").strip()

        if user_text.lower() in {"quit", "exit"}:
            print("GLaDOS: Leaving already? How unexpectedly merciful of you.")
            break
        if not user_text:
            continue

        history.append({"role": "user", "content": user_text})
        style = pick_style(last_style)

        print("GLaDOS: ", end="", flush=True)
        reply = ""
        try:
            for chunk in think(history, style):
                print(chunk, end="", flush=True)
                reply += chunk
        except openai.OpenAIError as err:
            # Network down, bad key, rate limit... don't crash the whole loop.
            print(f"[brain fault: {err}]")
            history.pop()  # drop the user message so history stays consistent
            continue
        print("\n")

        last_style = style
        history.append({"role": "assistant", "content": reply})

        # Keep only the most recent messages (the system prompt is added
        # fresh each call, so it never gets trimmed away).
        history = history[-MAX_HISTORY:]


if __name__ == "__main__":
    main()
