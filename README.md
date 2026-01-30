# Nemotron Speech Streaming ONNX Quantization

Quantize the [Nemotron Speech Streaming EN 0.6B](https://huggingface.co/altunenes/parakeet-rs)
ONNX model to **int8** and **int4** for faster CPU inference.

The source model is an RNN-T (Conformer encoder + LSTM decoder/joint) exported from
NVIDIA NeMo with cache-aware stateful streaming. This project produces smaller quantized
variants suitable for real-time CPU inference in tools like
[parakeet-rs](https://huggingface.co/altunenes/parakeet-rs).

## Quantization Methods

### int8 — Dynamic Quantization

Uses `onnxruntime.quantization.quantize_dynamic` with `QInt8` weight type. No calibration
data is needed. Weights are stored as int8, activations are quantized dynamically at runtime.
Expected output size: ~700 MB (down from ~2.7 GB).

### int4 — Weight-Only Quantization

Uses `onnxruntime.quantization.MatMul4BitsQuantizer` for 4-bit weight-only quantization.
Only MatMul weight tensors are quantized to 4 bits; other ops remain in fp32.
Expected output size: ~350–400 MB.

## Setup

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --no-install-project
```

## Usage

Run the scripts in order:

### 1. Download source model

```bash
uv run download.py
```

Downloads `encoder.onnx`, `encoder.onnx.data` (2.6 GB), `decoder_joint.onnx`, and
`tokenizer.model` from [altunenes/parakeet-rs](https://huggingface.co/altunenes/parakeet-rs)
into `source_model/`.

### 2. Quantize

```bash
uv run quantize.py
```

Produces int8 and int4 variants of both the encoder and decoder_joint in `output/int8/`
and `output/int4/`. Copies `tokenizer.model` unchanged. Verifies that I/O signatures
(including the encoder's cache tensors) are preserved after quantization.

### 3. Validate

```bash
uv run validate.py
```

Compares quantized models against the fp32 source:
- Verifies input/output tensor names match (critical for the encoder's stateful cache
  tensors: `cache_last_channel`, `cache_last_time`, `cache_last_channel_len`)
- Runs inference with dummy input and measures numerical divergence (cosine similarity,
  max/mean absolute error)
- Benchmarks inference latency (target: <560 ms per chunk on CPU)

### 4. Inference quality test

```bash
uv run inference_test.py --audio test.wav
```

Runs audio through the full streaming pipeline (mel spectrogram → encoder with cache
state → decoder/joint → SentencePiece decode) for fp32, int8, and int4 models. Compares
the actual transcription output to assess real-world quality impact of quantization.

### 5. Upload to HuggingFace

```bash
uv run upload.py
```

Publishes quantized models to:
- [lokkju/nemotron-speech-streaming-en-0.6b-int8](https://huggingface.co/lokkju/nemotron-speech-streaming-en-0.6b-int8)
- [lokkju/nemotron-speech-streaming-en-0.6b-int4](https://huggingface.co/lokkju/nemotron-speech-streaming-en-0.6b-int4)

Each repo includes the quantized ONNX files, tokenizer, and a model card with source
attribution.

## Source Model Details

| File | Size | Description |
|------|------|-------------|
| `encoder.onnx` | 44 MB | Stateful streaming encoder (graph only) |
| `encoder.onnx.data` | 2.6 GB | External weights for encoder |
| `decoder_joint.onnx` | 37 MB | Decoder + joint network |
| `tokenizer.model` | 257 KB | SentencePiece tokenizer |

The encoder is cache-aware with streaming state tensors for real-time chunk-by-chunk
inference. The export process is documented in
[this HuggingFace discussion](https://huggingface.co/altunenes/parakeet-rs/discussions/1).

## License

[PolyForm Shield 1.0.0](https://polyformproject.org/licenses/shield/1.0.0/)
