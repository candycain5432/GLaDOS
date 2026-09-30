# GLaDOS

A robotic, voice-controlled desktop assistant with the personality of GLaDOS from the *Portal* series: cheerful, polite, and quietly contemptuous of you.

The end goal is a physical build: a Raspberry Pi running the AI pipeline, an animated eye on a small round screen, servo-driven head movement, and a 3D-printed shell. It is built from the ground up in stages, and each stage is tested on its own before the next one starts.

## Status

| Stage | Goal | Status |
|---|---|---|
| 1 | Brain: typed chat with the GLaDOS personality | Done |
| 2 | Voice: she speaks her replies (TTS) | Next |
| 3 | Ears: wake word and speech-to-text | Planned |
| 4 | Move everything onto the Raspberry Pi | Planned |
| 5 | Assistant tools: timers, weather, and so on | Planned |
| 6 | Eye: animated on a round screen | Planned |
| 7 | Motion: head pan/tilt with servos | Planned |
| 8 | Shell: design and 3D print | Planned |

## How it works

The whole system is a loop of four independent jobs that connect only through plain text, so each one can be built and debugged separately:

```
mic -> wake word -> speech-to-text -> LLM (persona) -> text-to-speech -> speaker
```

Stage 1 covers the middle piece: the LLM and the persona, driven by typed input.

### Design decisions

- **The model is stateless.** Every request resends the recent conversation as a list of messages. "Memory" is just that list, trimmed to the last `MAX_HISTORY` messages.
- **The whole model layer lives in one function, `think()`.** Swapping the cloud API for a local model later means changing that function only.
- **Replies stream in chunks.** This becomes essential in Stage 2, when each finished sentence is sent to the voice while the model is still writing the next one.
- **The personality is a system prompt built from a few rules plus examples.** Early versions used about thirty rules and got stiff, report-like replies. Examples teach voice far better than rules, and fewer rules stopped them fighting each other.
- **Variety is enforced in code, not in the prompt.** The model cannot remember its own habits, so the program picks a random attack style each turn (`STYLES`) and never repeats the previous one. The style is added to that one request only and is never saved in the history.
- **Format is enforced in code too.** Newlines are stripped from the model output, because a prompt can only ask for a single paragraph and stray line breaks become awkward pauses in text-to-speech.
- **A non-reasoning model is used on purpose.** Reasoning models think silently before answering, and that pause is dead air in a voice assistant.

## Requirements

- Python 3.9 or newer (3.11+ recommended)
- An OpenAI API key with credit on the account. API billing is separate from a ChatGPT subscription.

## Setup

```bash
git clone <your-repo-url>
cd glados

python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
python -m pip install --upgrade pip
pip install openai

export OPENAI_API_KEY="your-key-here"     # Windows PowerShell: $env:OPENAI_API_KEY="your-key-here"
python glados.py
```

Type `quit` to exit.

The venv and the `export` only last for the current terminal window. In a new window, reactivate the venv and set the key again.

### Keeping your key safe

Never put the API key in the code or commit it. Use the environment variable, and add a `.gitignore` containing at least:

```
venv/
__pycache__/
.env
```

Set a monthly spending cap on the OpenAI billing page. An always-listening voice assistant is exactly the kind of program that can loop by accident.

### Troubleshooting

| Problem | Cause and fix |
|---|---|
| `Could not find a version that satisfies the requirement jiter` | Python is too old (3.8 or lower) or pip is outdated. Install a newer Python and rebuild the venv. |
| `python` shows 2.7 on macOS | The system default is old. Use `python3` outside a venv. Inside an active venv, `python` is safe. |
| Error 429, `insufficient_quota` | No API credit on the account. Add credit under OpenAI billing. |
| Authentication error | The key is not set in this terminal window. Run the `export` line again. |
| The last pasted line never gets a reply | The final line of a multi-line paste is waiting for Enter. Press Enter once. |

## Configuration

At the top of `glados.py`:

- `MODEL`: currently `gpt-4.1-mini`. `gpt-4o-mini` is a cheaper option that also works. Avoid reasoning models.
- `MAX_HISTORY`: how many past messages are resent each turn.
- `SYSTEM_PROMPT`: the entire personality. Tune it by changing one thing at a time and rerunning the same test prompts.
- `STYLES`: the list of attack styles the program rotates through.

## Testing the personality

Paste these into the prompt one after another, then press Enter once more so the last line is sent:

```
I'm going to go for a run today.
I think I'm pretty smart.
Can you help me with my homework?
What's 15 times 12?
I made dinner tonight.
I finished my project!
What's the largest planet?
Good morning.
```

What to look for:

- Statements get a direct reaction, not a lecture about the topic.
- Requests get a plain yes plus an insult, never a refusal.
- Questions get the correct answer first.
- No markdown, stage directions, or line breaks (the replies will be spoken aloud).
- No copying of the wording in the prompt's examples.

## Planned hardware

- Raspberry Pi 5 (4GB) with the official active cooler and 27W power supply
- Round LCD (about 1.28 inch, GC9A01 type) or a NeoPixel ring for the eye
- PCA9685 servo driver and 2-4 micro servos for head movement, on a separate 5-6V supply (never power servos from the Pi)
- USB microphone or speakerphone, and a small speaker (the Pi 5 has no headphone jack)
- 3D-printed shell, printed in several parts, with vents for the Pi and mounts sized to the real servo dimensions

## Roadmap notes

- **Stage 2 (voice):** text-to-speech with sentence-by-sentence streaming and a speaking thread with a queue. The first version can use OpenAI TTS (no install), with Piper (free and offline) as the target for the Pi. Both plug into the same `speak(sentence)` function.
- **Stage 3 (ears):** a local wake word so the microphone is not constantly streaming audio, then speech-to-text.
- **Stage 5 (tools):** timers, time and date, weather, and other functions the model can call, with the results wrapped in sarcasm.
- **Later:** an option to run the brain on a local model, either on a stronger PC on the home network or on the Pi, by changing only `think()`.

## Known limitations

- Small models do not reliably follow long lists of banned words. A word list turns into a game of whack-a-mole, so the prompt relies on examples instead.
- A leaked positive adjective (for example "impressive") can still appear occasionally. A code-level token filter is available as a fallback if it ever matters.
- The character never quotes lines from the games. All material is written new in the same spirit.

## Disclaimer

This is a fan project and is not affiliated with Valve. GLaDOS and *Portal* belong to their respective owners.
