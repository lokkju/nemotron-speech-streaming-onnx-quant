"""End-to-end inference quality test for quantized models.

Runs audio through the full Nemotron streaming pipeline:
  WAV -> mel spectrogram -> encoder (with cache) -> decoder_joint (RNN-T greedy) -> tokens -> text

Compares transcription output between fp32, int8, and int4 models.
"""

import argparse
import os
import time

import numpy as np
import onnxruntime as ort
import sentencepiece as spm
import soundfile as sf

# --- Model constants (from parakeet-rs/src/nemotron.rs) ---

SAMPLE_RATE = 16000
N_FFT = 512
WIN_LENGTH = 400      # 25ms
HOP_LENGTH = 160      # 10ms
N_MELS = 128
PREEMPH = 0.97
LOG_ZERO_GUARD = 5.960_464_5e-8  # 2^(-24)
FMAX = 8000.0
FMIN = 0.0

# Encoder streaming
NUM_ENCODER_LAYERS = 24
HIDDEN_DIM = 1024
LEFT_CONTEXT = 70
CONV_CONTEXT = 8
CHUNK_SIZE = 56          # mel frames per encoder chunk
PRE_ENCODE_CACHE = 9     # frames of context prepended before each chunk
CHUNK_SAMPLES = CHUNK_SIZE * HOP_LENGTH  # 8960 samples = 560ms

# Decoder
DECODER_LSTM_DIM = 640
DECODER_LSTM_LAYERS = 2
VOCAB_SIZE = 1024
BLANK_ID = 1024
MAX_SYMBOLS_PER_STEP = 10

SOURCE_DIR = "source_model"
OUTPUT_DIR = "output"


# --- Mel spectrogram ---

def _hann_window(length: int) -> np.ndarray:
    return 0.5 * (1 - np.cos(2 * np.pi * np.arange(length) / length))


def _mel_filterbank(sr: int, n_fft: int, n_mels: int, fmin: float, fmax: float) -> np.ndarray:
    """Slaney mel filterbank with Slaney normalization."""

    def hz_to_mel_slaney(hz):
        if hz < 1000.0:
            return hz * 3.0 / 200.0
        return 15.0 + 27.0 * np.log(hz / 1000.0) / np.log(6.4)

    def mel_to_hz_slaney(mel):
        if mel < 15.0:
            return mel * 200.0 / 3.0
        return 1000.0 * np.exp((mel - 15.0) * np.log(6.4) / 27.0)

    mel_min = hz_to_mel_slaney(fmin)
    mel_max = hz_to_mel_slaney(fmax)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = np.array([mel_to_hz_slaney(m) for m in mel_points])

    bin_freqs = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    n_freqs = n_fft // 2 + 1
    fb = np.zeros((n_mels, n_freqs), dtype=np.float32)

    for i in range(n_mels):
        low, center, high = bin_freqs[i], bin_freqs[i + 1], bin_freqs[i + 2]
        for j in range(low, center):
            if center > low:
                fb[i, j] = (j - low) / (center - low)
        for j in range(center, high):
            if high > center:
                fb[i, j] = (high - j) / (high - center)
        # Slaney normalization: 2 / bandwidth
        bandwidth = hz_points[i + 2] - hz_points[i]
        if bandwidth > 0:
            fb[i] *= 2.0 / bandwidth

    return fb


def compute_mel(audio: np.ndarray) -> np.ndarray:
    """Compute log-mel spectrogram matching NeMo/parakeet-rs Nemotron config.

    Returns shape [n_mels, n_frames].
    """
    # Preemphasis
    preemph_audio = np.empty_like(audio)
    preemph_audio[0] = audio[0]
    preemph_audio[1:] = audio[1:] - PREEMPH * audio[:-1]

    # Center pad
    pad = N_FFT // 2
    padded = np.pad(preemph_audio, (pad, pad), mode='constant')

    # STFT
    window = _hann_window(WIN_LENGTH)
    # Zero-pad window to N_FFT
    if WIN_LENGTH < N_FFT:
        win_padded = np.zeros(N_FFT)
        offset = (N_FFT - WIN_LENGTH) // 2
        win_padded[offset:offset + WIN_LENGTH] = window
        window = win_padded

    n_frames = 1 + (len(padded) - N_FFT) // HOP_LENGTH
    n_freqs = N_FFT // 2 + 1
    power = np.zeros((n_freqs, n_frames), dtype=np.float32)

    for t in range(n_frames):
        start = t * HOP_LENGTH
        frame = padded[start:start + N_FFT] * window
        spectrum = np.fft.rfft(frame, n=N_FFT)
        power[:, t] = np.real(spectrum * np.conj(spectrum)).astype(np.float32)

    # Mel filterbank
    fb = _mel_filterbank(SAMPLE_RATE, N_FFT, N_MELS, FMIN, FMAX)
    mel = fb @ power  # [n_mels, n_frames]

    # Log with guard
    mel = np.log(np.maximum(mel, 0.0) + LOG_ZERO_GUARD)

    return mel


# --- Encoder ---

def init_encoder_cache() -> dict:
    """Initialize zero cache state for the encoder."""
    return {
        "cache_last_channel": np.zeros(
            (NUM_ENCODER_LAYERS, 1, LEFT_CONTEXT, HIDDEN_DIM), dtype=np.float32
        ),
        "cache_last_time": np.zeros(
            (NUM_ENCODER_LAYERS, 1, HIDDEN_DIM, CONV_CONTEXT), dtype=np.float32
        ),
        "cache_last_channel_len": np.zeros((1,), dtype=np.int64),
    }


def run_encoder(sess: ort.InferenceSession, mel_chunk: np.ndarray, cache: dict):
    """Run encoder on a single chunk. Returns (encoder_out, new_cache)."""
    # mel_chunk shape: [1, n_mels, PRE_ENCODE_CACHE + CHUNK_SIZE]
    length = np.array([mel_chunk.shape[2]], dtype=np.int64)

    feed = {
        "audio_signal": mel_chunk,
        "length": length,
        "cache_last_channel": cache["cache_last_channel"],
        "cache_last_time": cache["cache_last_time"],
        "cache_last_channel_len": cache["cache_last_channel_len"],
    }
    outputs = sess.run(None, feed)
    # outputs: [encoder_out, encoded_lengths, new_cache_channel, new_cache_time, new_cache_channel_len]
    new_cache = {
        "cache_last_channel": outputs[2],
        "cache_last_time": outputs[3],
        "cache_last_channel_len": outputs[4],
    }
    return outputs[0], outputs[1], new_cache


# --- Decoder (RNN-T greedy) ---

def init_decoder_state() -> tuple:
    """Initialize zero LSTM state for decoder."""
    state_1 = np.zeros((DECODER_LSTM_LAYERS, 1, DECODER_LSTM_DIM), dtype=np.float32)
    state_2 = np.zeros((DECODER_LSTM_LAYERS, 1, DECODER_LSTM_DIM), dtype=np.float32)
    last_token = 0
    return last_token, state_1, state_2


def run_decoder_step(
    sess: ort.InferenceSession,
    encoder_frame: np.ndarray,
    last_token: int,
    state_1: np.ndarray,
    state_2: np.ndarray,
):
    """Run one decoder_joint step. Returns (logits, new_state_1, new_state_2)."""
    feed = {
        "encoder_outputs": encoder_frame,  # [1, hidden_dim, 1]
        "targets": np.array([[last_token]], dtype=np.int32),
        "target_length": np.array([1], dtype=np.int32),
        "input_states_1": state_1,
        "input_states_2": state_2,
    }
    outputs = sess.run(None, feed)
    return outputs[0], outputs[1], outputs[2]


def greedy_decode_chunk(
    decoder_sess: ort.InferenceSession,
    encoder_out: np.ndarray,
    enc_frames: int,
    last_token: int,
    state_1: np.ndarray,
    state_2: np.ndarray,
) -> tuple[list[int], int, np.ndarray, np.ndarray]:
    """Greedy RNN-T decode over encoder output frames."""
    tokens = []
    hidden_dim = encoder_out.shape[1]

    for t in range(enc_frames):
        frame = encoder_out[0, :, t].reshape(1, hidden_dim, 1)

        for _ in range(MAX_SYMBOLS_PER_STEP):
            logits, new_s1, new_s2 = run_decoder_step(
                decoder_sess, frame, last_token, state_1, state_2
            )
            max_idx = int(np.argmax(logits.flatten()))

            if max_idx == BLANK_ID:
                break

            tokens.append(max_idx)
            last_token = max_idx
            state_1 = new_s1
            state_2 = new_s2

    return tokens, last_token, state_1, state_2


# --- Full pipeline ---

def transcribe(
    encoder_sess: ort.InferenceSession,
    decoder_sess: ort.InferenceSession,
    tokenizer: spm.SentencePieceProcessor,
    audio: np.ndarray,
) -> tuple[str, float]:
    """Run full streaming transcription. Returns (text, total_inference_ms)."""
    mel = compute_mel(audio)  # [n_mels, n_frames]
    n_frames = mel.shape[1]

    # Initialize state
    cache = init_encoder_cache()
    last_token, state_1, state_2 = init_decoder_state()
    pre_encode_buffer = np.zeros((N_MELS, PRE_ENCODE_CACHE), dtype=np.float32)

    all_tokens = []
    total_ms = 0.0

    # Process in chunks
    pos = 0
    while pos < n_frames:
        end = min(pos + CHUNK_SIZE, n_frames)
        chunk_frames = mel[:, pos:end]

        # Pad if last chunk is short
        if chunk_frames.shape[1] < CHUNK_SIZE:
            pad_width = CHUNK_SIZE - chunk_frames.shape[1]
            chunk_frames = np.pad(chunk_frames, ((0, 0), (0, pad_width)), mode='constant')

        # Prepend pre-encode cache
        full_chunk = np.concatenate([pre_encode_buffer, chunk_frames], axis=1)
        full_chunk = full_chunk[np.newaxis, :, :]  # [1, n_mels, PRE_ENCODE_CACHE + CHUNK_SIZE]

        start_t = time.perf_counter()

        # Encoder
        encoder_out, enc_lengths, cache = run_encoder(encoder_sess, full_chunk, cache)
        enc_frames = int(enc_lengths[0])

        # Decoder
        tokens, last_token, state_1, state_2 = greedy_decode_chunk(
            decoder_sess, encoder_out, enc_frames, last_token, state_1, state_2
        )

        elapsed = (time.perf_counter() - start_t) * 1000
        total_ms += elapsed
        all_tokens.extend(tokens)

        # Update pre-encode buffer from the end of current chunk
        pre_encode_buffer = mel[:, max(0, end - PRE_ENCODE_CACHE):end]
        if pre_encode_buffer.shape[1] < PRE_ENCODE_CACHE:
            pre_encode_buffer = np.pad(
                pre_encode_buffer,
                ((0, 0), (PRE_ENCODE_CACHE - pre_encode_buffer.shape[1], 0)),
                mode='constant',
            )

        pos = end

    text = tokenizer.decode(all_tokens)
    return text, total_ms


def load_model_set(model_dir: str, quant_suffix: str = ""):
    """Load encoder and decoder sessions from a directory."""
    if quant_suffix:
        enc_name = f"encoder.{quant_suffix}.onnx"
        dec_name = f"decoder_joint.{quant_suffix}.onnx"
    else:
        enc_name = "encoder.onnx"
        dec_name = "decoder_joint.onnx"

    enc_path = os.path.join(model_dir, enc_name)
    dec_path = os.path.join(model_dir, dec_name)

    if not os.path.exists(enc_path) or not os.path.exists(dec_path):
        return None, None

    enc_sess = ort.InferenceSession(enc_path, providers=["CPUExecutionProvider"])
    dec_sess = ort.InferenceSession(dec_path, providers=["CPUExecutionProvider"])
    return enc_sess, dec_sess


def main():
    parser = argparse.ArgumentParser(description="Inference quality test for quantized models")
    parser.add_argument("--audio", required=True, help="Path to WAV file (16kHz mono)")
    parser.add_argument("--models", nargs="*", default=["fp32", "int8", "int4"],
                        help="Model variants to test (default: fp32 int8 int4)")
    args = parser.parse_args()

    # Load audio
    audio, sr = sf.read(args.audio, dtype="float32")
    if sr != SAMPLE_RATE:
        print(f"ERROR: Expected {SAMPLE_RATE} Hz, got {sr} Hz. Resample first.")
        return
    if audio.ndim > 1:
        audio = audio[:, 0]  # mono

    duration_s = len(audio) / SAMPLE_RATE
    print(f"Audio: {args.audio} ({duration_s:.1f}s, {len(audio)} samples)")

    # Load tokenizer
    tok_path = os.path.join(SOURCE_DIR, "tokenizer.model")
    if not os.path.exists(tok_path):
        print(f"ERROR: Tokenizer not found at {tok_path}. Run download.py first.")
        return
    tokenizer = spm.SentencePieceProcessor(model_file=tok_path)
    print(f"Tokenizer: {tokenizer.get_piece_size()} tokens")

    # Run each model variant
    results = {}
    for variant in args.models:
        if variant == "fp32":
            model_dir = SOURCE_DIR
            suffix = ""
        else:
            model_dir = os.path.join(OUTPUT_DIR, variant)
            suffix = variant

        enc_sess, dec_sess = load_model_set(model_dir, suffix)
        if enc_sess is None:
            print(f"\n--- {variant} --- SKIPPED (model files not found)")
            continue

        print(f"\n--- {variant} ---")
        text, total_ms = transcribe(enc_sess, dec_sess, tokenizer, audio)
        ms_per_chunk = total_ms / max(1, int(np.ceil(duration_s / 0.56)))

        print(f"  Text: {text}")
        print(f"  Total inference: {total_ms:.0f} ms")
        print(f"  Per chunk (~560ms audio): {ms_per_chunk:.0f} ms")
        results[variant] = text

    # Compare
    if "fp32" in results and len(results) > 1:
        print("\n--- Comparison ---")
        ref = results["fp32"]
        for variant, text in results.items():
            if variant == "fp32":
                continue
            if text == ref:
                print(f"  {variant}: EXACT MATCH")
            else:
                # Simple character-level similarity
                matches = sum(a == b for a, b in zip(ref, text))
                max_len = max(len(ref), len(text))
                sim = matches / max_len if max_len > 0 else 1.0
                print(f"  {variant}: {sim*100:.1f}% character similarity")
                print(f"    fp32: {ref[:200]}")
                print(f"    {variant}: {text[:200]}")


if __name__ == "__main__":
    main()
