# Qwen3-TTS for Mac - Run AI Text-to-Speech Locally on Apple Silicon

Run **Qwen3-TTS** text-to-speech AI locally on your MacBook with Apple Silicon (M1, M2, M3, M4). No cloud, no API keys, completely offline.

**Keywords:** Qwen TTS Mac, Qwen3 TTS Apple Silicon, MLX text to speech, local TTS Mac, voice cloning Mac, AI voice generator MacBook

---

## Features

- **Voice Cloning** - Clone any voice from a 5-second audio sample
- **Voice Design** - Create new voices by describing them ("deep narrator", "excited child")
- **Custom Voices** - 9 built-in voices with emotion and speed control
- **100% Local** - Runs entirely on your Mac, no internet required
- **Optimized for M-Series** - Uses Apple's MLX framework for fast GPU inference

---

## Why MLX Models?

MLX models are specifically optimized for Apple Silicon. Compared to running standard PyTorch models:

| Metric | Standard Model | MLX Model |
|--------|----------------|-----------|
| **RAM Usage** | 10+ GB | 2-3 GB |
| **CPU Temperature** | 80-90°C | 40-50°C |

*Tested on M4 MacBook Air (fanless) with 1.7B models*

MLX runs natively on the Apple Neural Engine and GPU, meaning better performance with less heat and battery drain.

---

## Quick Start (5 Minutes)

### 1. Clone and setup

```bash
git clone https://github.com/kapi2800/qwen3-tts-apple-silicon.git
cd qwen3-tts-apple-silicon
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
brew install ffmpeg
```

### 2. Download models

Pick the models you need from the table below. Click the link, then click "Download" on HuggingFace.

**Pro Models (1.7B) - Best Quality**

| Model | Use Case | Download |
|-------|----------|----------|
| CustomVoice | Preset voices + emotion control | [Download](https://huggingface.co/mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit) |
| VoiceDesign | Create voices from text description | [Download](https://huggingface.co/mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit) |
| Base | Voice cloning from audio | [Download](https://huggingface.co/mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit) |

**Lite Models (0.6B) - Faster, Less RAM**

| Model | Use Case | Download |
|-------|----------|----------|
| CustomVoice | Preset voices + emotion control | [Download](https://huggingface.co/mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit) |
| VoiceDesign | Create voices from text description | [Download](https://huggingface.co/mlx-community/Qwen3-TTS-12Hz-0.6B-VoiceDesign-8bit) |
| Base | Voice cloning from audio | [Download](https://huggingface.co/mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit) |

Put downloaded folders in `models/`:
```
models/
├── Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit/
├── Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit/
└── Qwen3-TTS-12Hz-1.7B-Base-8bit/
```

### 3. Run

```bash
source .venv/bin/activate
python main.py
```

---

## Usage

```
========================================
 Qwen3-TTS Manager
========================================

  Pro Models (1.7B - Best Quality)
  ---------------------------------
  1. Custom Voice
  2. Voice Design
  3. Voice Cloning

  Lite Models (0.6B - Faster)
  ---------------------------
  4. Custom Voice
  5. Voice Design
  6. Voice Cloning

  q. Exit

Select: 
```

- **Custom Voice**: Pick from preset speakers, set emotion and speed
- **Voice Design**: Describe a voice (e.g., "calm British narrator")
- **Voice Cloning**: Provide a reference audio clip to clone

---

## Tips

- Drag `.txt` files directly into the terminal for long text
- Multi-line text is generated as separate segments and joined into a single file
- Voice cloning works best with clean 5-10 second audio clips
- Speed options: Normal (1.0x), Fast (1.3x), Slow (0.8x) - applied as pitch-preserving time-stretching
- Type `q` or `exit` anytime to go back

---

## Audio Post-Processing

Every generated file is cleaned up before it is saved to `outputs/`:

0. **Vocoder fix** - MLX's `ConvTranspose1d` silently returns wrong results once
   its output grows past 32768 samples, which blanked out large stretches of the
   vocoder output (see *Known upstream bug* below). `main.py` replaces it with an
   exact chunked equivalent, so this happens before anything else.
1. **Segment joining** - all generated segments are concatenated into one file
2. **Band-limiting** - a 50 Hz high-pass removes the strong DC / sub-bass offset
   produced by the 8-bit codec decoder (audible as rumble and distortion), and an
   11 kHz low-pass removes its tone at Nyquist
3. **Defect repair** - a safety net for the rare remaining artifact. Silence longer
   than 0.8 s is collapsed to 0.45 s and sustained pure tones are cut out, each
   splice crossfaded over 5 ms. Normal speech is left untouched.
4. **Retry** - if a generation still contains a sustained tone artifact, it is
   regenerated (up to 3 attempts) and the cleanest take is kept
5. **Speed adjustment** - via `ffmpeg atempo`, falling back to `librosa` when
   `ffmpeg` is not installed
6. **Peak normalization** to -1 dBFS
7. **10 ms fade in/out** to remove clicks at the edges

These settings can be tuned at the top of `main.py`.

## Known upstream bug (MLX `ConvTranspose1d`)

On `mlx==0.30.3`, `nn.ConvTranspose1d` returns incorrect results once the output
exceeds 32768 samples. It is reproducible in isolation, with no TTS involved:

```python
import mlx.core as mx, mlx.nn as nn, numpy as np
conv = nn.ConvTranspose1d(384, 192, 8, stride=4)
x = mx.array(np.random.randn(1, 9000, 384).astype(np.float32))
# conv(x) differs from a float64 reference by ~1.5 (the signal's own magnitude),
# while any length below the threshold matches to ~1e-3.
```

The speech tokenizer's vocoder upsamples through several of these layers, so for
anything longer than a few words the audio was wrong: roughly 40-60 % of each file
came out as flat, near-silent DC instead of speech. That was heard as the speech
being chopped up by long interruptions, and it also made generations sound as if
words were missing - the model was producing them, but the vocoder was erasing
them.

`main.py` works around it by splitting the transposed convolution along time and
summing the overlapping tails, which is mathematically identical and keeps every
call inside the range where MLX is correct. Verified exact (within float32 noise)
against a float64 reference at all lengths. Remove `_patch_mlx_conv_transpose()`
once MLX fixes this upstream.

## Language

Qwen3-TTS conditions on full language names (`english`, `russian`, `chinese`,
`japanese`, `korean`, `german`, `french`, `spanish`, `italian`, `portuguese`).
The language is detected automatically from the script of your text, so Cyrillic
input is rendered with Russian phonetics instead of falling back to unconditioned
generation.

---

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.10+
- RAM: ~3GB for Lite models, ~6GB for Pro models

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `mlx_audio not found` | Run `source .venv/bin/activate` first |
| `Model not found` | Check model folder names match exactly |
| Audio won't play | Check macOS sound output settings |
| Rumble or distortion in output | Ensure `scipy` is installed - the band-limiting filter requires it |
| Beeps or long gaps in output | Repaired automatically; tune `TONE_*` / `*_GAP_SECONDS` in `main.py` |
| Speech chopped up, or words missing | Caused by the MLX `ConvTranspose1d` bug above. Make sure `_patch_mlx_conv_transpose()` still runs at startup (it prints nothing when it works) |

---

## Star History

[![Star History Chart](https://api.star-history.com/chart?repos=kapi2800/qwen3-tts-apple-silicon&type=date&legend=top-left)](https://www.star-history.com/?repos=kapi2800%2Fqwen3-tts-apple-silicon&type=date&legend=top-left)


---

## Related Projects

- [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) - Original Qwen3-TTS by Alibaba
- [MLX Audio](https://github.com/Blaizzy/mlx-audio) - MLX framework for audio models
- [MLX Community](https://huggingface.co/mlx-community) - Pre-converted MLX models


---

**If this project helped you, please give it a ⭐ star!**
