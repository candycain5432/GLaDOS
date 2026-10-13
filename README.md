# GLaDOS

A robotic, voice-controlled desktop assistant with the personality of GLaDOS from the *Portal* series: cheerful, polite, and quietly contemptuous of you.

The end goal is a physical build: a Raspberry Pi running the voice pipeline, a ceiling-mounted 3D-printed body with a servo-driven arm and head, an LED ring and an orange LED eye. It is built in stages, and each stage is tested on its own before the next one starts.

## Status

| Stage | Goal | Status |
|---|---|---|
| 1 | Brain: chat with the GLaDOS personality | Done |
| 2 | Voice: she speaks her replies | Done, now with the real GLaDOS voice running locally |
| 3 | Ears: wake word and speech-to-text | Done, using OpenAI speech-to-text, wake word read from the transcript |
| 4 | Local voice merged into the main program, plus a hardware on/off switch | Done on the Arch box. Switch is written and tested with fake GPIO, not yet on a real Pi |
| 5 | Move everything onto the Raspberry Pi | Next |
| 6 | Assistant tools: timers, time and date, weather | Planned |
| 7 | Eye: orange LED, brightness animation from the Pi | Planned |
| 8 | Motion: arm and head servos, LED ring control | Planned |
| 9 | Body: print, assemble, wire | Planned (print list below) |

Dropped: a local wake word (openWakeWord). Training it in Colab failed repeatedly, so the wake word is found in the speech-to-text transcript instead. See Known limitations for what that costs.

## How it works

The whole system is a loop of independent jobs that connect only through plain text, so each one can be built and debugged separately:

```
mic -> speech detection -> OpenAI speech-to-text -> wake word check -> LLM (persona) -> local voice -> speaker
```

Say "GLaDOS" and ask something. She answers out loud, and for a few seconds afterwards you can reply without saying her name.

### Files

| File | Purpose |
|---|---|
| `glados.py` | The main program: brain, local voice, switch, and conversation loop |
| `ears.py` | Microphone capture, speech detection, speech-to-text, and wake word. Run it directly (`python ears.py`) to test the ears alone |
| `net_check.py` | Diagnostic that tests the OpenAI connection and prints the real cause of any failure |

### Design decisions

**Brain**
- The model is stateless. Every request resends the recent conversation as a list of messages, trimmed to the last `MAX_HISTORY`.
- All model-specific code lives in one function, `think()`, so swapping the cloud API for a local model means changing that function only.
- The personality is a system prompt built from a few rules plus examples. Early versions used about thirty rules and got stiff, report-like replies. Examples teach voice far better than rules.
- Variety is enforced in code. The model cannot remember its own habits, so the program picks a random attack style each turn (`STYLES`). The style is added to that one request and never saved in the history.
- A non-reasoning model is used on purpose. Reasoning models think silently before answering, and that pause is dead air in a voice assistant.

**Voice**
- The voice is [glados-tts](https://github.com/nerdaxic/glados-tts): a Forward Tacotron model that turns text into a spectrogram, and a HiFiGAN vocoder that turns the spectrogram into audio. It runs inside `glados.py`, with no separate server.
- Replies stream in chunks. Each finished sentence goes to the voice while the model is still writing the next one.
- Two worker threads connected by queues: one makes audio, one plays it, so sentence 2 is generated while sentence 1 plays.
- The lq (low quality) vocoder is used on purpose. Measured on a 4-core Arch machine: about 0.5 s for a short line and 1 s for a long one. The hq vocoder took about 6 s on a long line, which is too slow for conversation.
- Repeated lines are cached (`lru_cache`), and a few warm-up calls at startup avoid a slow first reply.
- Format is enforced in code too. Newlines are stripped from the model output, because stray line breaks become awkward pauses in speech.

**Ears**
- Speech detection is a local loudness check (RMS over 30 ms blocks). At startup the program listens to one second of silence to learn the room's noise floor and sets the threshold at about three times that.
- An utterance starts after three loud blocks in a row and ends after about 0.8 seconds of quiet. A short pre-roll keeps the first syllable from being clipped, and short clicks are discarded.
- The wake word is found in the transcript. The spelling is nudged with a prompt hint, and a regex covers common mishearings (Glados, Gladys, Glad Os, and so on).
- She does not listen while she is speaking, so she never answers her own voice. The cost is that you cannot interrupt her.
- After she speaks there is a follow-up window (`FOLLOW_UP_SECONDS`) in which no wake word is needed.

**On/off switch**
- Set `SWITCH_PIN` in `glados.py` to the GPIO (BCM) number. Wire a switch between that pin and ground. The program uses the Pi's internal pull-up, so closed means on.
- When the switch is off, the microphone is never opened. Flipping it off mid-recording throws the recording away. Flipping it back on starts a fresh conversation.
- This is a software gate. If you want a guarantee that cannot be bypassed by a bug, cut the microphone's power or signal with the switch itself, which is a hardware decision still to be made.

## Requirements

- An OpenAI API key with credit on the account. API billing is separate from a ChatGPT subscription.
- A microphone and speakers.
- For the local voice: Python 3.10, because the CPU vocoder file needs the old PyTorch 1.13.1, which needs Python 3.10 and `numpy<2`. Newer PyTorch cannot load it (it fails on `xnnpack.Conv2dOpContext`).
- The glados-tts repository and its model files, in `~/glados-tts` (or set `GLADOS_TTS_DIR`). Check its license before copying anything, and link to it instead of redistributing.

## Setup (Linux)

```bash
# system packages (Arch shown)
sudo pacman -S espeak-ng portaudio alsa-utils
# Debian, Ubuntu and Raspberry Pi OS: sudo apt install espeak-ng libportaudio2 alsa-utils

git clone https://github.com/nerdaxic/glados-tts ~/glados-tts     # then follow its README to get models/

mkdir -p ~/glados && cd ~/glados
python3.10 -m venv venv-local
source venv-local/bin/activate
python -m pip install --upgrade pip
pip install torch==1.13.1 "numpy<2" scipy phonemizer unidecode inflect openai sounddevice
# on a Raspberry Pi also: pip install gpiozero

export OPENAI_API_KEY="your-key-here"     # one line, no trailing newline
python ears.py        # test the ears on their own first
python glados.py      # then the full assistant
```

Set `USE_MIC = False` in `glados.py` to type instead of speak. Set `TTS_BACKEND = "openai"` to use the OpenAI voice instead of the local one (this does not need torch).

On a Mac, allow your terminal app under Privacy settings, then Microphone. Intel Macs cannot run the local voice because PyTorch no longer ships for them.

The venv and the `export` only last for the current terminal window. In a new window, reactivate the venv and set the key again.

### Keeping your key safe

Never put the API key in the code, commit it, or paste it anywhere public. Error messages and logs can contain it, so check before sharing output. If a key is ever exposed, delete it on the OpenAI API keys page and create a new one. Add a `.gitignore` containing at least:

```
venv*/
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
| `ModuleNotFoundError: phonemizer` or an espeak error | Install `espeak-ng` with your package manager, then `pip install phonemizer unidecode inflect`. |
| `FileNotFoundError: aplay` | Install `alsa-utils`. |
| `xnnpack.Conv2dOpContext` error when loading the vocoder | PyTorch is too new for the CPU vocoder. Use the Python 3.10 environment with `torch==1.13.1` and `numpy<2`. |
| Voice replies take several seconds | `LOCAL_VOCODER` is set to an hq or gpu file. Use `vocoder-cpu-lq.pt`. |
| `Could not find a version that satisfies the requirement jiter` | Python is too old (3.8 or lower) or pip is outdated. Install a newer Python and rebuild the venv. |
| Microphone warning about pure silence | The terminal app lacks microphone permission. Enable it in the system Privacy settings and restart the terminal. |
| She cuts you off mid-sentence | Raise `END_SILENCE_BLOCKS` in `ears.py` (each block is 30 ms). |
| She never triggers, or triggers on room noise | Adjust `MIN_THRESHOLD` or `NOISE_MULTIPLIER` in `ears.py`, using the level bar in `python ears.py`. |

## Configuration

In `glados.py`:

- `MODEL`: the brain, currently `gpt-4.1-mini`. Avoid reasoning models.
- `TTS_BACKEND`: `"local"` or `"openai"`. `LOCAL_VOCODER` and `GLADOS_TTS_DIR` set the local voice files.
- `SWITCH_PIN`: the GPIO pin for the listening switch, or `None` for no switch.
- `SYSTEM_PROMPT` and `STYLES`: the entire personality. Tune one thing at a time and rerun the same test prompts.
- `FOLLOW_UP_SECONDS`, `POST_SPEECH_PAUSE`, `USE_MIC`, `WAKE_REPLIES`, `QUIT_PHRASES`: conversation behavior.

In `ears.py`: `STT_MODEL` (try `whisper-1` if the default errors, though it invents text out of noise more readily) and the tuning constants at the top.

## Hardware plan

The body is a ceiling-mounted, 3D-printed model made by someone else, with 28 main-unit files (the optional hand controller is not used). Its listing gives the electronics. Check the listing's license before printing, and do not redistribute the files. Measured from the files, everything is in millimeters at 100% scale.

**Sizes (100% scale):** base 130 mm across and 71 mm tall, mounting plate 130 mm across and 24 mm thick, main pivot about 152 × 68 × 110, moving arm 100 × 115 × 102, head 57 × 83 × 42. Hanging down it should be roughly 25 to 30 cm (estimate, not yet assembled).

**Print list:** one base, one mounting plate (wires version), one top cover ring and one matching base ring (use the "open" variant with no fans first), the main pivot parts (back, internals, tube), moving arm, rotation, rotation junction box, head, gear brace, servo clamp, pushrods, the spur gear pair, and all wire blocks. Print the gears, servo clamp and bearing parts first as a fit test.

**Electronics:**
- Raspberry Pi 5 (4 GB) with the active cooler, microSD, and the official 27 W supply
- 12 V supply and an LED driver sized for the LEDs
- 8 high-power LEDs (1 to 3 W, warm white) with heatsinks for the ring
- 1 orange 1 to 3 W LED with a heatsink for the eye, driven by PWM from the Pi through a MOSFET
- 3 digital micro servos and 1 digital standard servo rated 6 to 8 kg·cm, on their own 5 to 6 V supply (about 5 A), never from the Pi
- PCA9685 servo driver
- 3 × 608 bearings and 1 × 624 bearing
- 2 × 40 mm fans (optional, only with the fan variants)
- USB microphone, small speaker and I2S amp (the Pi 5 has no headphone jack)
- A lot of wire

**Fit:** the Pi 5 fits in the open upper part of the base on a printed tray (it does not fit beside the central column lower down). The 12 V supply and LED driver should live outside the base, in a small box on the ceiling side, with one cable to the unit. The ring parts are thin curved walls and hold nothing.

The round-display eye from the first plan was dropped: the head only takes a single LED.

## Roadmap notes

- **Run on the Pi.** Confirm that PyTorch 1.13.1 and the glados-tts voice run on a Pi 5 (64-bit OS needed; ARM wheels for that exact PyTorch and Python 3.10 are not confirmed). Voice timing is expected to be about three times slower than the Arch box, which is not measured.
- **Assistant tools.** Timers, time and date, weather, and other functions the model can call, with results wrapped in sarcasm.
- **Eye and ring.** PWM brightness patterns for listening, thinking and talking.
- **Motion.** Arm and head servo control from the Pi, with idle motion and looking toward you.
- **Pi tray.** A printable tray with four M2.5 standoffs (58 × 49 mm hole pattern) for the base.
- **Phrase cache.** Generate stock lines once and replay them from disk.
- **Local brain.** Run the model on a stronger machine on the home network by changing only `think()`.
- **Press-to-talk.** An optional alternative to the wake word that sends no audio until a button is pressed.

## Known limitations

- Every spoken utterance near the microphone is uploaded to OpenAI for speech-to-text while the program runs, including speech not meant for her. The switch stops that when it is off. Cost is about $0.003 per minute of audio.
- Speech-to-text and the brain are billed per use.
- She cannot be interrupted while speaking.
- Small models do not reliably follow long lists of banned words, so the prompt relies on examples. An occasional stray word can still slip through.
- The character never quotes lines from the games. All material is written new in the same spirit.
- Running on a real Raspberry Pi, the real GPIO switch, and the physical build are untested.

## Credits

The voice and hardware plans draw on two public projects: [glados-tts](https://github.com/nerdaxic/glados-tts) and [glados-voice-assistant](https://github.com/nerdaxic/glados-voice-assistant) by nerdaxic, with the original TTS model by R2D2FISH. The printed body is a fan-made model by its original designer. Check each license before copying any code, model or print files, and link to them instead of redistributing.

## Disclaimer

This is a fan project and is not affiliated with Valve. GLaDOS and *Portal* belong to their respective owners.
