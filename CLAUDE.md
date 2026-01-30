# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Quantize the **Nemotron Speech Streaming EN 0.6B ONNX** model to int8 (dynamic quantization) and int4 (weight-only quantization) for faster CPU inference. Source model lives at `altunenes/parakeet-rs` on HuggingFace.

## Build & Run

This is a `uv`-managed Python project (Python 3.10+).

```bash
uv sync                          # install dependencies
uv run download.py               # download source model from HuggingFace
uv run quantize.py               # run quantization (outputs to output/int8/ and output/int4/)
uv run validate.py               # validate quantized models against fp32 baseline
uv run upload.py                 # upload quantized models to HuggingFace
```

## Architecture

The project is a linear pipeline of standalone scripts:

- **download.py** — Downloads source ONNX files (`encoder.onnx`, `encoder.onnx.data`, `decoder_joint.onnx`, `tokenizer.model`) from HuggingFace
- **quantize.py** — Applies int8 dynamic quantization (`onnxruntime.quantization.quantize_dynamic`) and int4 weight-only quantization (`MatMulNBitsQuantizer`) to encoder and decoder_joint models
- **validate.py** — Loads quantized models, compares transcripts against fp32 baseline, measures inference time (target: <560ms per chunk on CPU)
- **upload.py** — Publishes quantized models to HuggingFace repos (`lokkju/nemotron-speech-streaming-en-0.6b-int8` and `-int4`)

## Key Technical Details

- The encoder is **stateful** (cache-aware streaming) with cache inputs (`cache_last_channel`, `cache_last_time`, `cache_last_channel_len`) and corresponding cache outputs. Quantization must preserve these I/O signatures.
- The encoder has a large external weights file (`encoder.onnx.data`, 2.6 GB). Load with `onnx.load(..., load_external_data=True)`. Quantization must handle external data correctly.
- `tokenizer.model` (SentencePiece) is copied unchanged — never quantized.
- int4 `MatMulNBitsQuantizer` may not support all ops; fall back to GPTQ or RTN block-wise quantization if needed.
- Output files use naming convention: `encoder.int8.onnx`, `decoder_joint.int4.onnx`, etc.
- The `output/` directory is gitignored and contains the quantized model files.
- Source model was exported from NeMo using `cache_aware_stream_step()` with dynamic axes on batch and time dimensions. The export script is documented in the [HuggingFace discussion](https://huggingface.co/altunenes/parakeet-rs/discussions/1).

## Downstream Integration

Quantized models will be integrated into the **voxtype** project by adding entries to `PARAKEET_MODELS` in `src/setup/model.rs` and updating `detect_model_type()` to recognize int8/int4 file naming.
