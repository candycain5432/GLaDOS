# GLaDOS

A robotic, voice-controlled desktop assistant with the personality of GLaDOS from the *Portal* series: cheerful, polite, and quietly contemptuous of you.

The end goal is a physical build: a Raspberry Pi running the AI pipeline, an animated eye on a small round screen, servo-driven head movement, and a 3D-printed shell. It is built from the ground up in stages, and each stage is tested on its own before the next one starts.

## Status

| Stage | Goal | Status |
|---|---|---|
| 1 | Brain: chat with the GLaDOS personality | Done |
| 2 | Voice: she speaks her replies (TTS) | Done, using an OpenAI voice as a placeholder |
| 3 | Ears: wake word and speech-to-text | Done, using OpenAI as a placeholder |
| 4 | Swap in a real GLaDOS voice and a local wake word | Next |
| 5 | Move everything onto the Raspberry Pi | Planned |
| 6 | Assistant tools: timers, weather, and so on | Planned |
| 7 | Eye: animated on a round screen | Planned |
| 8 | Motion: head pan/tilt with servos | Planned |
| 9 | Shell: design and 3D print | Planned |

## How it works

The whole system is a loop of independent jobs that connect only through plain text, so each one can be built and debugged separately:

```
mic -> wake word -> speech-to-text -> LLM (persona) -> text-to-speech -> speaker
```

Say "GLaDOS" and ask something. She answers out loud, and for a few seconds afterwards you can reply without saying her name.

### Files

| File | Purpose |
|---|---|
| `glados.py` | The main program: brain, voice pipeline, and conversation loop |
| `ears.py` | Microphone capture, speech detection, speech-to-text, and wake word. Run it directly (`python ears.py`) to test the ears alone |
| `net_check.py` | Diagnostic that tests the OpenAI connection and prints the real cause of any failure |

### Design decisions

**Brain**
- The model is stateless. Every request resends the recent conversation as a list of messages, trimmed to the last `MAX_HISTORY`.
- All model-specific code lives in one function, `think()`, so swapping the cloud API for a local model means changing that function only.
- The personality is a system prompt built from a few rules plus examples. Early versions used about thirty rules and got stiff, report-like replies. Examples teach voice far better than rules.
- Variety is enforced in code. The model cannot remember its own habits, so the program picks a random attack style each turn (`STYLES`) and never repeats the previous one. The style is added to that one request and never saved in the history.
- A non-reasoning model is used on purpose. Reasoning models think silently before answering, and that pause is dead air in a voice assistant.

**Voice**
- Replies stream in chunks. Each finished sentence goes to the voice while the model is still writing the next one, so she starts talking about a second after the reply begins.
- Two worker threads connected by queues: one fetches audio, one plays it, so sentence 2 is fetched while sentence 1 plays.
- Format is enforced in code too. Newlines are stripped from the model output, because a prompt can only ask for one paragraph and stray line breaks become awkward pauses in speech.

**Ears**
- Speech detection is a local loudness check (RMS over 30 ms blocks). At startup the program listens to one second of silence to learn the room's noise floor and sets the speech threshold at about three times that.
- An utterance starts after three loud blocks in a row and ends after about 0.8 seconds of quiet. A short pre-roll keeps the first syllable from being clipped, and short clicks are discarded.
- The wake word is found in the speech-to-text transcript. The spelling is nudged with a prompt hint, and a regex covers common mishearings (Glados, Gladys, Glad Os, and so on).
- She does not listen while she is speaking, so she never answers her own voice. The cost is that you cannot interrupt her.
- After she speaks there is a follow-up window (`FOLLOW_UP_SECONDS`) in which no wake word is needed.

## Requirements

- Python 3.9 or newer (3.11+ recommended)
- An OpenAI API key with credit on the account. API billing is separate from a ChatGPT subscription.
- A microphone and speakers

## Setup

```bash
git clone <your-repo-url>
cd glados

python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
python -m pip install --upgrade pip
pip install openai sounddevice numpy

export OPENAI_API_KEY="your-key-here"     # Windows PowerShell: $env:OPENAI_API_KEY="your-key-here"
python ears.py        # test the ears on their own first
python glados.py      # then the full assistant
```

On Linux, `sounddevice` needs PortAudio first (`sudo pacman -S portaudio` on Arch, `sudo apt install libportaudio2` on Debian, Ubuntu, and Raspberry Pi OS).

On a Mac, allow your terminal app under Privacy settings, then Microphone, the first time it asks.

The venv and the `export` only last for the current terminal window. In a new window, reactivate the venv and set the key again. Set `USE_MIC = False` in `glados.py` to type instead of speak.

### Keeping your key safe

Never put the API key in the code, commit it, or paste it anywhere public. Error messages and logs can contain it, so check before sharing output. If a key is ever exposed, delete it on the OpenAI API keys page and create a new one. Add a `.gitignore` containing at least:

```
venv/
__pycache__/
.env
```

Set a monthly spending cap on the OpenAI billing page. An always-listening voice assistant is exactly the kind of program that can loop by accident.

### Troubleshooting

| Problem | Cause and fix |
|---|---|
| `Connection error` with a cause of `Illegal header value ... Bearer sk-...` | The key has a stray newline on the end. Set it again on a single line and check with `echo "[$OPENAI_API_KEY]"`. The closing bracket should sit right after the key. `net_check.py` shows the underlying cause. |
| Error 429, `insufficient_quota` | No API credit on the account. Add credit under OpenAI billing. |
| Authentication error | The key is not set in this terminal window. Run the `export` line again. |
| `Could not find a version that satisfies the requirement jiter` | Python is too old (3.8 or lower) or pip is outdated. Install a newer Python and rebuild the venv. |
| `python` shows 2.7 on macOS | The system default is old. Use `python3` outside a venv. Inside an active venv, `python` is safe. |
| Microphone warning about pure silence | The terminal app lacks microphone permission. Enable it in the system Privacy settings and restart the terminal. |
| She cuts you off mid-sentence | Raise `END_SILENCE_BLOCKS` in `ears.py` (each block is 30 ms). |
| She never triggers, or triggers on room noise | Adjust `MIN_THRESHOLD` or `NOISE_MULTIPLIER` in `ears.py`, using the level bar in `python ears.py`. |

## Configuration

In `glados.py`:

- `MODEL`: the brain, currently `gpt-4.1-mini`. Avoid reasoning models.
- `TTS_MODEL`, `TTS_VOICE`, `TTS_INSTRUCTIONS`: the placeholder voice.
- `SYSTEM_PROMPT` and `STYLES`: the entire personality. Tune one thing at a time and rerun the same test prompts.
- `FOLLOW_UP_SECONDS`, `POST_SPEECH_PAUSE`, `USE_MIC`: conversation behavior.

In `ears.py`: `STT_MODEL` (try `whisper-1` if the default errors, though it invents text out of noise more readily) and the tuning constants at the top.

## Planned hardware

- Raspberry Pi 5 (4GB) with the official active cooler and 27W power supply
- Round LCD (about 1.28 inch, GC9A01 type) or a NeoPixel ring for the eye
- PCA9685 servo driver and 2-4 micro servos for head movement, on a separate 5-6V supply (never power servos from the Pi)
- USB microphone or speakerphone, and a small speaker (the Pi 5 has no headphone jack)
- 3D-printed shell, printed in several parts, with vents for the Pi and mounts sized to the real servo dimensions

## Roadmap notes

- **Real GLaDOS voice.** The OpenAI voice cannot reproduce her synthetic timbre, which comes from processing rather than delivery. The plan is to run [glados-tts](https://github.com/nerdaxic/glados-tts) (a neural voice trained on her voice lines, forked from R2D2FISH's original) on a Linux machine and have `speak()` fetch audio from it over the network. Only that one function changes. It needs PyTorch, which does not install on Intel Macs, so this is tested on a Linux machine. Its README warns that newer versions of the related voice assistant do not run on a Raspberry Pi, so the plan is to keep the synthesis on a stronger machine and let the Pi handle the eye, servos, mic, and speaker.
- **Local wake word.** The current wake word uploads every utterance to OpenAI, including speech not meant for her. A local detector (such as openWakeWord) would send audio only after hearing her name.
- **Assistant tools.** Timers, time and date, weather, and other functions the model can call, with results wrapped in sarcasm.
- **Phrase cache.** Generate stock lines once and replay them from disk.
- **Local brain.** Run the model on a stronger machine on the home network by changing only `think()`.

## Known limitations

- Every spoken utterance near the microphone is uploaded to OpenAI while the program runs. This is temporary, pending a local wake word.
- Speech-to-text and voice calls are billed per use.
- She cannot be interrupted while speaking.
- Small models do not reliably follow long lists of banned words, so the prompt relies on examples. An occasional stray word can still slip through.
- The character never quotes lines from the games. All material is written new in the same spirit.

## Credits

The voice and hardware plans draw on two public projects: [glados-tts](https://github.com/nerdaxic/glados-tts) and [glados-voice-assistant](https://github.com/nerdaxic/glados-voice-assistant) by nerdaxic, with the original TTS model by R2D2FISH. Check their licenses before copying any code or model files, and link to them instead of redistributing.

## Disclaimer

This is a fan project and is not affiliated with Valve. GLaDOS and *Portal* belong to their respective owners.
