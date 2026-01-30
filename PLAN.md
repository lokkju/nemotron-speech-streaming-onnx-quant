# Nemotron Speech Streaming ONNX Quantization

Quantize the Nemotron Speech Streaming EN 0.6B ONNX model to int8 and int4 for faster CPU inference.

## Source Model

- **Name**: nemotron-speech-streaming-en-0.6b
- **Source**: `altunenes/parakeet-rs` on HuggingFace (subdirectory `nemotron-speech-streaming-en-0.6b/`)
- **Files**:
  - `encoder.onnx` (44 MB)
  - `encoder.onnx.data` (2.6 GB) — external weights for encoder
  - `decoder_joint.onnx` (37 MB)
  - `tokenizer.model` (257 KB) — SentencePiece tokenizer (copy as-is)

## Output Models

### int8 (dynamic quantization)
- `encoder.onnx` (quantized, replaces fp32)
- `decoder_joint.onnx` (quantized, replaces fp32)
- `tokenizer.model` (unchanged)
- Expected size: ~700 MB total (similar ratio to TDT int8: 2600 MB → 670 MB)

### int4 (weight-only quantization)
- `encoder.onnx` (quantized, replaces fp32)
- `decoder_joint.onnx` (quantized, replaces fp32)
- `tokenizer.model` (unchanged)
- Expected size: ~350-400 MB total

**Important:** Output files must be named `encoder.onnx` and `decoder_joint.onnx` (not `encoder.int8.onnx` etc.) because parakeet-rs `Nemotron::from_pretrained()` hardcodes these filenames. Each quantization level is published as a separate HuggingFace repo, so the directory name distinguishes the variant.

## Approach

### int8: Dynamic Quantization
No calibration data needed. Weights are quantized to int8, activations quantized dynamically at runtime.

```python
from onnxruntime.quantization import quantize_dynamic, QuantType

quantize_dynamic("encoder.onnx", "encoder.int8.onnx", weight_type=QuantType.QInt8)
quantize_dynamic("decoder_joint.onnx", "decoder_joint.int8.onnx", weight_type=QuantType.QInt8)
```

### int4: Weight-Only Quantization
Uses ONNX Runtime's MatMul4BitsQuantizer for 4-bit weight quantization.

```python
from onnxruntime.quantization import matmul_4bits_quantizer

quantizer = matmul_4bits_quantizer.MatMul4BitsQuantizer(model)
quantizer.process()
quantizer.model.save("encoder.int4.onnx")
```

## Project Structure

```
nemotron-speech-streaming-onnx-quant/
├── PLAN.md
├── LICENSE              # Polyform Shield 1.0.0
├── pyproject.toml       # uv project, deps: onnxruntime, onnx
├── quantize.py          # Main script
├── validate.py          # Compare output quality vs fp32
├── download.py          # Download source model from HuggingFace
├── upload.py            # Upload quantized models to HuggingFace
└── output/              # gitignored, quantized model files
    ├── int8/
    └── int4/
```

## Validation

After quantization, validate that streaming inference still works:
1. Load quantized model in parakeet-rs
2. Feed known audio through streaming pipeline
3. Compare transcript against fp32 baseline
4. Measure inference time per chunk (target: <560ms per chunk on CPU)

## HuggingFace Publishing

Publish as separate model repos under the lokkju namespace:
- `lokkju/nemotron-speech-streaming-en-0.6b-int8`
- `lokkju/nemotron-speech-streaming-en-0.6b-int4`

Each repo contains the quantized ONNX files + tokenizer.model + a model card explaining the source and quantization method.

## Voxtype Integration

After publishing, add the int8/int4 models to voxtype's `PARAKEET_MODELS` in `src/setup/model.rs` so they appear in `voxtype setup model`. Also update `detect_model_type()` to recognize int8/int4 Nemotron file naming (encoder.int8.onnx etc).

## Dependencies

- Python 3.10+
- onnxruntime (for quantization APIs)
- onnx (for model loading/saving)
- huggingface-hub (for download/upload)

## Notes

- The encoder is **stateful** (cache-aware streaming) with cache I/O tensors (`cache_last_channel`, `cache_last_time`, `cache_last_channel_len`). Quantization must preserve these signatures. See [HF discussion](https://huggingface.co/altunenes/parakeet-rs/discussions/1).
- The encoder has external data (`encoder.onnx.data`). Load with `onnx.load(..., load_external_data=True)`. Verify the output is self-contained or has its own external data file.
- int4 quantization via MatMul4BitsQuantizer may not support all ops in the model. If it fails, try GPTQ or RTN block-wise quantization.
- SentencePiece tokenizer (`tokenizer.model`) is not quantized — copy it unchanged.
