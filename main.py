import os
import sys
import glob
import shutil
import time
import wave
import gc
import re
import subprocess
import warnings
from datetime import datetime

import numpy as np
from scipy.signal import butter, sosfiltfilt

# Suppress harmless library warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    import mlx.core as mx
    from mlx_audio.tts.utils import load_model
    from mlx_audio.tts.generate import generate_audio
except ImportError:
    print("Error: 'mlx_audio' library not found.")
    print("Run: source .venv/bin/activate")
    sys.exit(1)

# MLX's ConvTranspose1d returns wrong results once the output exceeds 2^15
# samples, which silences large stretches of the vocoder output. Chunk below it.
MLX_CONVT_MAX_OUT = 16384


def _chunked_conv_transpose(conv, x, stride, kernel):
    """Exact ConvTranspose1d for any length, in NLC layout.

    Transposed convolution is linear and local, so a long input can be split
    along time and the overlapping tails summed. Each call stays inside the
    length range where MLX is correct.
    """
    length = x.shape[1]
    if (length - 1) * stride + kernel <= MLX_CONVT_MAX_OUT:
        return conv(x)

    max_in = max(1, (MLX_CONVT_MAX_OUT - kernel) // stride + 1)
    overlap = kernel - stride
    bias = conv.bias if hasattr(conv, "bias") else None

    pieces, carry = [], None
    for start in range(0, length, max_in):
        chunk = x[:, start : start + max_in, :]
        y = conv(chunk)
        if carry is not None:
            head = y[:, :overlap, :] + carry
            if bias is not None:
                # Bias is applied by both chunks in the overlap; keep one copy.
                head = head - bias
            y = mx.concatenate([head, y[:, overlap:, :]], axis=1)
        body_len = chunk.shape[1] * stride
        if start + max_in < length and overlap > 0:
            pieces.append(y[:, :body_len, :])
            carry = y[:, body_len:, :]
        else:
            pieces.append(y)
            carry = None
    return mx.concatenate(pieces, axis=1)


def _patch_mlx_conv_transpose():
    """Route the vocoder's transposed convolutions through the chunked version."""
    try:
        from mlx_audio.tts.models.qwen3_tts import speech_tokenizer as _st
    except ImportError:
        return

    def call(self, x):
        x = mx.transpose(x, (0, 2, 1))  # NCL -> NLC
        kernel = self.conv.weight.shape[1]
        stride = kernel - self.trim_right
        x = _chunked_conv_transpose(self.conv, x, stride, kernel)
        x = mx.transpose(x, (0, 2, 1))  # NLC -> NCL
        if self.trim_right > 0:
            x = x[..., : -self.trim_right]
        return x

    for name in ("CausalTransposeConv1d", "DecoderBlockUpsample"):
        cls = getattr(_st, name, None)
        if cls is not None:
            cls.__call__ = call


_patch_mlx_conv_transpose()

# Configuration
BASE_OUTPUT_DIR = os.path.join(os.getcwd(), "outputs")
MODELS_DIR = os.path.join(os.getcwd(), "models")
VOICES_DIR = os.path.join(os.getcwd(), "voices")

# Settings
AUTO_PLAY = True
SAMPLE_RATE = 24000
FILENAME_MAX_LEN = 20

# Audio post-processing
# The 8-bit codec decoder emits a strong DC / sub-50Hz component that is heard as
# rumble and distortion, so every generated file is high-pass filtered before saving.
HIGHPASS_HZ = 50
HIGHPASS_ORDER = 2
# The decoder also leaves a tone right at Nyquist (12 kHz) that is heard as a faint whine.
LOWPASS_HZ = 11000
TARGET_PEAK = 0.891  # -1 dBFS
MAX_NORMALIZE_GAIN = 8.0
FADE_MS = 10

# Degenerate output repair
# The talker occasionally falls into a failure mode where it emits codec tokens that
# decode to a constant pure tone (a "beep") padded with blocks of digital silence.
SILENCE_FLOOR = 1e-4       # below this a frame counts as silence
SILENCE_REL = 0.002        # ...or below this share of the loud-speech level
GAP_TRIGGER_SECONDS = 0.8  # only pauses longer than this are considered dead air
MAX_GAP_SECONDS = 0.45     # ...and they are collapsed to this length
TONE_FREQ_MIN = 2000       # a steady tone above this is never speech
TONE_RATIO = 0.6           # share of frame energy concentrated in one narrow band
TONE_MIN_SECONDS = 0.25    # a tone must hold this long to count as an artifact
TONE_DRIFT_HZ = 60         # ...and its pitch must stay this stable throughout
SPLICE_MS = 5              # crossfade applied at every cut
MAX_ATTEMPTS = 3           # regenerate this many times before giving up

# Qwen3-TTS conditions on full language names; "en" is silently ignored by the model.
LANGUAGE_BY_SCRIPT = [
    ("russian", r'[\u0400-\u04FF]'),
    ("japanese", r'[\u3040-\u30FF]'),
    ("korean", r'[\uAC00-\uD7AF]'),
    ("chinese", r'[\u4E00-\u9FFF]'),
]
DEFAULT_LANGUAGE = "english"

# Model Definitions
MODELS = {
    # Pro (1.7B)
    "1": {"name": "Custom Voice", "folder": "Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit", "mode": "custom", "output_subfolder": "CustomVoice"},
    "2": {"name": "Voice Design", "folder": "Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit", "mode": "design", "output_subfolder": "VoiceDesign"},
    "3": {"name": "Voice Cloning", "folder": "Qwen3-TTS-12Hz-1.7B-Base-8bit", "mode": "clone_manager", "output_subfolder": "Clones"},
    # Lite (0.6B)
    "4": {"name": "Custom Voice", "folder": "Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit", "mode": "custom", "output_subfolder": "CustomVoice"},
    "5": {"name": "Voice Design", "folder": "Qwen3-TTS-12Hz-0.6B-VoiceDesign-8bit", "mode": "design", "output_subfolder": "VoiceDesign"},
    "6": {"name": "Voice Cloning", "folder": "Qwen3-TTS-12Hz-0.6B-Base-8bit", "mode": "clone_manager", "output_subfolder": "Clones"},
}

SPEAKER_MAP = {
    "English": ["Ryan", "Aiden", "Ethan", "Chelsie", "Serena", "Vivian"],
    "Chinese": ["Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric"],
    "Japanese": ["Ono_Anna"],
    "Korean": ["Sohee"]
}

EMOTION_EXAMPLES = [
    "Sad and crying, speaking slowly",
    "Excited and happy, speaking very fast",
    "Angry and shouting",
    "Whispering quietly"
]


def flush_input():
    try:
        import termios
        termios.tcflush(sys.stdin, termios.TCIOFLUSH)
    except (ImportError, OSError):
        pass


def clean_memory():
    gc.collect()


def make_temp_dir():
    return f"temp_{int(time.time())}"


def get_smart_path(folder_name):
    full_path = os.path.join(MODELS_DIR, folder_name)
    if not os.path.exists(full_path):
        return None

    snapshots_dir = os.path.join(full_path, "snapshots")
    if os.path.exists(snapshots_dir):
        subfolders = [f for f in os.listdir(snapshots_dir) if not f.startswith('.')]
        if subfolders:
            return os.path.join(snapshots_dir, subfolders[0])

    return full_path


def detect_language(text):
    for language, pattern in LANGUAGE_BY_SCRIPT:
        if re.search(pattern, text):
            return language
    return DEFAULT_LANGUAGE


def _frame_flags(audio, sample_rate, win=2048, hop=512):
    """Classify each analysis frame as silent and/or narrowband, with its peak pitch."""
    starts = np.arange(0, max(len(audio) - win, 0) + 1, hop)
    silent = np.zeros(len(starts), dtype=bool)
    narrow = np.zeros(len(starts), dtype=bool)
    pitch = np.zeros(len(starts), dtype=np.float32)
    if len(starts) == 0:
        return starts, silent, narrow, pitch, hop

    window = np.hanning(win).astype(np.float32)
    freqs = np.fft.rfftfreq(win, 1 / sample_rate)

    levels = np.array([np.sqrt(np.mean(audio[s:s + win] ** 2)) for s in starts])
    floor = max(SILENCE_FLOOR, SILENCE_REL * float(np.percentile(levels, 95)))

    for i, start in enumerate(starts):
        if levels[i] < floor:
            silent[i] = True
            continue

        spectrum = np.abs(np.fft.rfft(audio[start:start + win] * window))
        total = spectrum.sum()
        if total <= 0:
            silent[i] = True
            continue

        peak = int(np.argmax(spectrum))
        if freqs[peak] < TONE_FREQ_MIN:
            continue
        band = np.abs(freqs - freqs[peak]) < 100
        if spectrum[band].sum() / total > TONE_RATIO:
            narrow[i] = True
            pitch[i] = freqs[peak]

    return starts, silent, narrow, pitch, hop


def _runs(flags):
    """Yield (start_index, end_index) for each run of True values."""
    if not len(flags):
        return
    padded = np.r_[False, flags, False]
    edges = np.diff(padded.astype(np.int8))
    for s, e in zip(np.where(edges == 1)[0], np.where(edges == -1)[0]):
        yield s, e


def find_defects(audio, sample_rate):
    """Return (keep_mask, tone_seconds, trimmed_gap_seconds) for a decoded signal."""
    keep = np.ones(len(audio), dtype=bool)
    if len(audio) < 4096:
        return keep, 0.0, 0.0

    starts, silent, narrow, pitch, hop = _frame_flags(audio, sample_rate)
    tone_seconds = 0.0
    gap_seconds = 0.0

    for s, e in _runs(narrow):
        duration = (e - s) * hop / sample_rate
        if duration < TONE_MIN_SECONDS:
            continue
        # Speech is never stationary for this long; a locked pitch means a synthetic beep.
        if np.ptp(pitch[s:e]) > TONE_DRIFT_HZ:
            continue
        lo = int(starts[s])
        hi = int(starts[e - 1] + hop)
        keep[lo:hi] = False
        tone_seconds += duration

    trigger = int(GAP_TRIGGER_SECONDS * sample_rate)
    max_gap = int(MAX_GAP_SECONDS * sample_rate)
    for s, e in _runs(silent):
        lo = int(starts[s])
        hi = int(starts[e - 1] + hop)
        # Leading and trailing silence is trimmed away completely.
        edge = lo == 0 or hi >= len(audio) - hop
        if edge:
            keep[lo:hi] = False
            gap_seconds += (hi - lo) / sample_rate
        elif hi - lo > trigger:
            keep[lo + max_gap:hi] = False
            gap_seconds += (hi - lo - max_gap) / sample_rate

    return keep, tone_seconds, gap_seconds


def repair_audio(audio, sample_rate, verbose=True):
    """Cut out pure-tone bursts and over-long silences, splicing with short crossfades."""
    keep, tone_seconds, gap_seconds = find_defects(audio, sample_rate)
    if keep.all():
        return audio

    splice = max(1, int(sample_rate * SPLICE_MS / 1000))
    ramp = np.linspace(0, 1, splice, dtype=np.float32)

    pieces = []
    for s, e in _runs(keep):
        piece = audio[s:e].copy()
        if len(piece) > 2 * splice:
            if s > 0:
                piece[:splice] *= ramp
            if e < len(audio):
                piece[-splice:] *= ramp[::-1]
        pieces.append(piece)

    if not pieces:
        return audio

    if verbose:
        details = []
        if tone_seconds:
            details.append(f"{tone_seconds:.1f}s of tone artifacts")
        if gap_seconds:
            details.append(f"{gap_seconds:.1f}s of dead air")
        print("Repaired: removed " + " and ".join(details))

    return np.concatenate(pieces)


def analyze_audio(audio, sample_rate):
    """Return (tone_seconds, defect_ratio) for a decoded signal."""
    if audio is None or len(audio) == 0:
        return 0.0, 1.0
    # Analyse the same band-limited signal that repair_audio will see.
    limited = band_limit(np.asarray(audio, dtype=np.float32), sample_rate)
    keep, tone_seconds, _ = find_defects(limited, sample_rate)
    return tone_seconds, 1.0 - float(keep.mean())


def generate_with_retry(generate_fn, temp_dir_factory):
    """Run generation, retrying while the model returns badly degenerate audio.

    Dead air is removed losslessly by repair_audio, so only a sustained pure tone
    (or a near-empty result) is treated as a failed generation worth repeating.
    """
    best = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        temp_dir = temp_dir_factory()
        try:
            generate_fn(temp_dir)
            audio, sample_rate = join_segments(temp_dir)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        if audio is None:
            continue

        tone_seconds, ratio = analyze_audio(audio, sample_rate)
        if best is None or tone_seconds < best[2]:
            best = (audio, sample_rate, tone_seconds)
        if tone_seconds < TONE_MIN_SECONDS and ratio < 0.8:
            break

        if attempt < MAX_ATTEMPTS:
            print(f"Detected {tone_seconds:.1f}s of tone artifacts. Regenerating "
                  f"(attempt {attempt + 1}/{MAX_ATTEMPTS})...")

    if best is None:
        return None, None
    return best[0], best[1]


def _has_ffmpeg():
    return shutil.which("ffmpeg") is not None


def read_wav_mono(path):
    with wave.open(path, 'rb') as f:
        sample_rate = f.getframerate()
        channels = f.getnchannels()
        frames = f.readframes(f.getnframes())

    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return audio, sample_rate


def write_wav_mono(path, audio, sample_rate):
    samples = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(path, 'wb') as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(samples.tobytes())


def join_segments(temp_folder):
    parts = sorted(glob.glob(os.path.join(temp_folder, "audio_*.wav")))
    if not parts:
        return None, None

    chunks, sample_rate = [], None
    for part in parts:
        try:
            audio, rate = read_wav_mono(part)
        except (wave.Error, EOFError):
            continue
        if sample_rate is None:
            sample_rate = rate
        if rate != sample_rate:
            print(f"Warning: skipping '{os.path.basename(part)}' (sample rate mismatch).")
            continue
        chunks.append(audio)

    if not chunks:
        return None, None
    if len(chunks) > 1:
        print(f"Joining {len(chunks)} segments...")
    return np.concatenate(chunks), sample_rate


def band_limit(audio, sample_rate):
    """Strip the decoder's DC/sub-bass offset and its tone at Nyquist."""
    # sosfiltfilt pads the signal by 3 * (2 * order), so skip very short buffers.
    if len(audio) <= 6 * HIGHPASS_ORDER * 3:
        return audio - float(np.mean(audio))

    nyquist = sample_rate / 2
    if LOWPASS_HZ < nyquist:
        sos = butter(HIGHPASS_ORDER, [HIGHPASS_HZ, LOWPASS_HZ], btype='bandpass',
                     fs=sample_rate, output='sos')
    else:
        sos = butter(HIGHPASS_ORDER, HIGHPASS_HZ, btype='highpass',
                     fs=sample_rate, output='sos')
    return sosfiltfilt(sos, audio).astype(np.float32)


def change_speed(audio, sample_rate, speed):
    if abs(speed - 1.0) < 0.01 or len(audio) == 0:
        return audio

    if _has_ffmpeg():
        stretched = _change_speed_ffmpeg(audio, sample_rate, speed)
        if stretched is not None:
            return stretched

    try:
        import librosa
        return librosa.effects.time_stretch(audio, rate=speed).astype(np.float32)
    except Exception as e:
        print(f"Warning: speed adjustment unavailable ({e}). Using original speed.")
        return audio


def _change_speed_ffmpeg(audio, sample_rate, speed):
    stamp = int(time.time() * 1000)
    src = os.path.join(os.getcwd(), f"temp_speed_in_{stamp}.wav")
    dst = os.path.join(os.getcwd(), f"temp_speed_out_{stamp}.wav")
    try:
        write_wav_mono(src, audio, sample_rate)
        cmd = ["ffmpeg", "-y", "-v", "error", "-i", src,
               "-filter:a", f"atempo={speed}", "-ac", "1", "-c:a", "pcm_s16le", dst]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        result, _ = read_wav_mono(dst)
        return result
    except (subprocess.CalledProcessError, FileNotFoundError, wave.Error, OSError):
        return None
    finally:
        for path in (src, dst):
            if os.path.exists(path):
                os.remove(path)


def normalize_peak(audio):
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    if peak <= 1e-6:
        return audio
    return audio * min(TARGET_PEAK / peak, MAX_NORMALIZE_GAIN)


def apply_fades(audio, sample_rate):
    fade_len = int(sample_rate * FADE_MS / 1000)
    if fade_len < 2 or len(audio) < 2 * fade_len:
        return audio

    ramp = 0.5 * (1 - np.cos(np.linspace(0, np.pi, fade_len, dtype=np.float32)))
    audio = audio.copy()
    audio[:fade_len] *= ramp
    audio[-fade_len:] *= ramp[::-1]
    return audio


def postprocess_audio(audio, sample_rate, speed=1.0):
    audio = np.asarray(audio, dtype=np.float32)
    if len(audio) == 0:
        return audio

    audio = band_limit(audio, sample_rate)
    audio = repair_audio(audio, sample_rate)
    audio = change_speed(audio, sample_rate, speed)
    audio = normalize_peak(audio)
    return apply_fades(audio, sample_rate)


def save_audio_file(audio, sample_rate, subfolder, text_snippet, speed=1.0):
    if audio is None or len(audio) == 0:
        print("Nothing to save: the model returned no audio.")
        return

    save_path = os.path.join(BASE_OUTPUT_DIR, subfolder)
    os.makedirs(save_path, exist_ok=True)

    timestamp = datetime.now().strftime("%H-%M-%S")
    flat_text = " ".join(text_snippet.split())
    clean_text = re.sub(r'[^\w\s-]', '', flat_text)[:FILENAME_MAX_LEN].strip().replace(' ', '_') or "audio"
    filename = f"{timestamp}_{clean_text}.wav"
    final_path = os.path.join(save_path, filename)

    audio = postprocess_audio(audio, sample_rate, speed)
    write_wav_mono(final_path, audio, sample_rate)
    print(f"Saved: outputs/{subfolder}/{filename}")

    if AUTO_PLAY:
        print("Playing...")
        try:
            subprocess.run(["afplay", final_path], check=False, 
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except FileNotFoundError:
            pass


def clean_path(user_input):
    path = user_input.strip()
    if len(path) > 1 and path[0] in ["'", '"'] and path[-1] == path[0]:
        path = path[1:-1]
    return path.replace("\\ ", " ")


def get_safe_input(prompt="\nEnter text (or drag .txt file): "):
    try:
        raw_input = input(prompt).strip()
        if raw_input.lower() in ['exit', 'quit', 'q']:
            return None

        clean_p = clean_path(raw_input)
        if os.path.exists(clean_p) and clean_p.endswith(".txt"):
            print(f"Reading from: {os.path.basename(clean_p)}")
            try:
                with open(clean_p, 'r', encoding='utf-8') as f:
                    return f.read().strip()
            except IOError as e:
                print(f"Error reading file: {e}")
                return None

        return raw_input
    except KeyboardInterrupt:
        flush_input()
        return None


def convert_audio_if_needed(input_path):
    if not os.path.exists(input_path):
        return None

    filename = os.path.basename(input_path)
    name, ext = os.path.splitext(filename)

    if ext.lower() == ".wav":
        try:
            with wave.open(input_path, 'rb') as f:
                if f.getnchannels() > 0:
                    return input_path
        except wave.Error:
            pass

    temp_wav = os.path.join(os.getcwd(), f"temp_convert_{int(time.time())}.wav")
    print(f"Converting '{ext}' to WAV...")

    cmd = ["ffmpeg", "-y", "-v", "error", "-i", input_path, 
           "-ar", str(SAMPLE_RATE), "-ac", "1", "-c:a", "pcm_s16le", temp_wav]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        return temp_wav
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: Could not convert audio. Is ffmpeg installed?")
        return None


def get_saved_voices():
    if not os.path.exists(VOICES_DIR):
        return []
    voices = [f.replace(".wav", "") for f in os.listdir(VOICES_DIR) if f.endswith(".wav")]
    return sorted(voices)


def enroll_new_voice():
    print("\n--- Enroll New Voice ---")
    flush_input()

    name = input("1. Voice name (e.g. Boss, Mom): ").strip()
    if not name:
        return

    safe_name = re.sub(r'[^\w\s-]', '', name).strip().replace(' ', '_')

    ref_input = input("2. Drag & Drop Reference File: ").strip()
    raw_path = clean_path(ref_input)

    if len(raw_path) > 300 or "\n" in raw_path:
        print("Error: Input too long.")
        flush_input()
        return

    clean_wav_path = convert_audio_if_needed(raw_path)
    if not clean_wav_path:
        return

    print("3. Transcript (important for quality):")
    ref_text = input("   Type EXACTLY what the audio says: ").strip()

    if not os.path.exists(VOICES_DIR):
        os.makedirs(VOICES_DIR)

    target_wav = os.path.join(VOICES_DIR, f"{safe_name}.wav")
    target_txt = os.path.join(VOICES_DIR, f"{safe_name}.txt")

    shutil.copy(clean_wav_path, target_wav)
    with open(target_txt, "w", encoding='utf-8') as f:
        f.write(ref_text)

    if clean_wav_path != raw_path and os.path.exists(clean_wav_path):
        os.remove(clean_wav_path)

    print(f"Voice saved as '{safe_name}'")


def run_custom_session(model_key):
    info = MODELS[model_key]
    model_path = get_smart_path(info["folder"])
    if not model_path:
        print("Error: Model not found.")
        return

    print(f"\nLoading {info['name']}...")
    try:
        model = load_model(model_path)
    except Exception as e:
        print(f"Load failed: {e}")
        return

    print(f"\n--- {info['name']} ---")
    speaker = "Vivian"
    all_speakers = [n for names in SPEAKER_MAP.values() for n in names]
    print("Available Speakers: " + ", ".join(all_speakers))

    user_choice = input("\nSelect Speaker (Name): ").strip()
    for lang, names in SPEAKER_MAP.items():
        if user_choice in names:
            speaker = user_choice
            break
    print(f"Using: {speaker}")

    print("\nEmotion Examples:")
    for ex in EMOTION_EXAMPLES:
        print(f"  - {ex}")
    base_instruct = input("Emotion Instruction: ").strip() or "Normal tone"

    print("\nSpeed:")
    print("  1. Normal (1.0x)")
    print("  2. Fast (1.3x)")
    print("  3. Slow (0.8x)")
    sp = input("Choice (1-3): ").strip()
    speed = 1.0
    if sp == "2":
        speed = 1.3
    elif sp == "3":
        speed = 0.8

    while True:
        text = get_safe_input()
        if text is None:
            break
        print("Generating...")
        language = detect_language(text)
        try:
            audio, sample_rate = generate_with_retry(
                lambda temp_dir: generate_audio(
                    model=model, text=text, voice=speaker, instruct=base_instruct,
                    lang_code=language, output_path=temp_dir),
                make_temp_dir)
            save_audio_file(audio, sample_rate, info["output_subfolder"], text, speed=speed)
        except Exception as e:
            print(f"Error: {e}")
    clean_memory()


def run_design_session(model_key):
    info = MODELS[model_key]
    model_path = get_smart_path(info["folder"])
    if not model_path:
        print("Error: Model not found.")
        return

    print(f"\nLoading {info['name']}...")
    try:
        model = load_model(model_path)
    except Exception as e:
        print(f"Load failed: {e}")
        return

    print(f"\n--- {info['name']} ---")
    instruct = input("Describe the voice: ").strip()
    if not instruct:
        return

    while True:
        text = get_safe_input()
        if text is None:
            break
        print("Generating...")
        language = detect_language(text)
        try:
            audio, sample_rate = generate_with_retry(
                lambda temp_dir: generate_audio(
                    model=model, text=text, instruct=instruct,
                    lang_code=language, output_path=temp_dir),
                make_temp_dir)
            save_audio_file(audio, sample_rate, info["output_subfolder"], text)
        except Exception as e:
            print(f"Error: {e}")
    clean_memory()


def run_clone_manager(model_key):
    print("\n--- Voice Cloning Manager ---")
    print("  1. Pick from Saved Voices")
    print("  2. Enroll New Voice")
    print("  3. Quick Clone")
    print("  4. Back")

    sub_choice = input("\nChoice: ").strip()
    if sub_choice == "2":
        enroll_new_voice()
        return
    if sub_choice == "4":
        return

    info = MODELS[model_key]
    model_path = get_smart_path(info["folder"])
    if not model_path:
        print("Error: Model not found.")
        return

    print("\nLoading Base Model...")
    try:
        model = load_model(model_path)
    except Exception as e:
        print(f"Load failed: {e}")
        return

    ref_audio, ref_text = None, None

    if sub_choice == "1":
        saved = get_saved_voices()
        if not saved:
            print("No saved voices found.")
            return
        print("\nSaved Voices:")
        for i, v in enumerate(saved):
            print(f"  {i+1}. {v}")
        try:
            idx = int(input("\nPick Number: ")) - 1
            if idx < 0 or idx >= len(saved):
                print("Invalid selection.")
                return
            name = saved[idx]
            ref_audio = os.path.join(VOICES_DIR, f"{name}.wav")
            txt_path = os.path.join(VOICES_DIR, f"{name}.txt")
            if os.path.exists(txt_path):
                with open(txt_path, 'r', encoding='utf-8') as f:
                    ref_text = f.read().strip()
            print(f"Loaded: {name}")
        except (ValueError, IndexError):
            print("Invalid selection.")
            return

    elif sub_choice == "3":
        ref_input = input("\nDrag Reference Audio: ").strip()
        raw_path = clean_path(ref_input)
        ref_audio = convert_audio_if_needed(raw_path)
        if not ref_audio:
            return
        ref_text = input("   Transcript (Optional): ").strip() or "."

    else:
        return

    while True:
        text = get_safe_input(f"\nText for '{os.path.basename(str(ref_audio))}' (or 'exit'): ")
        if text is None:
            break
        print("Cloning...")
        language = detect_language(text)
        try:
            audio, sample_rate = generate_with_retry(
                lambda temp_dir: generate_audio(
                    model=model, text=text, ref_audio=ref_audio, ref_text=ref_text,
                    lang_code=language, output_path=temp_dir),
                make_temp_dir)
            save_audio_file(audio, sample_rate, info["output_subfolder"], text)
        except Exception as e:
            print(f"Error: {e}")
    clean_memory()


def main_menu():
    print("\n" + "=" * 40)
    print(" Qwen3-TTS Manager")
    print("=" * 40)
    
    print("\n  Pro Models (1.7B - Best Quality)")
    print("  ---------------------------------")
    print("  1. Custom Voice")
    print("  2. Voice Design")
    print("  3. Voice Cloning")
    
    print("\n  Lite Models (0.6B - Faster)")
    print("  ---------------------------")
    print("  4. Custom Voice")
    print("  5. Voice Design")
    print("  6. Voice Cloning")
    
    print("\n  q. Exit")

    choice = input("\nSelect: ").strip().lower()

    if choice == "q":
        sys.exit()

    if choice not in MODELS:
        print("Invalid selection.")
        flush_input()
        return

    mode = MODELS[choice]["mode"]

    if mode == "custom":
        run_custom_session(choice)
    elif mode == "design":
        run_design_session(choice)
    elif mode == "clone_manager":
        run_clone_manager(choice)


if __name__ == "__main__":
    try:
        os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
        while True:
            main_menu()
    except KeyboardInterrupt:
        print("\nExiting...")
