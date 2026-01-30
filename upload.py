"""Upload quantized models to HuggingFace."""

import os

from huggingface_hub import HfApi

OUTPUT_DIR = "output"
NAMESPACE = "lokkju"
BASE_NAME = "nemotron-speech-streaming-en-0.6b"
SOURCE_REPO = "altunenes/parakeet-rs"

QUANT_CONFIGS = {
    "int8": {
        "repo_id": f"{NAMESPACE}/{BASE_NAME}-int8",
        "method": "Dynamic int8 quantization (onnxruntime quantize_dynamic, QInt8 weights)",
    },
    "int4": {
        "repo_id": f"{NAMESPACE}/{BASE_NAME}-int4",
        "method": "Weight-only int4 quantization (onnxruntime MatMul4BitsQuantizer)",
    },
}

MODEL_CARD_TEMPLATE = """---
license: other
license_name: polyform-shield-1.0.0
license_link: https://polyformproject.org/licenses/shield/1.0.0/
tags:
  - onnx
  - asr
  - speech-recognition
  - streaming
  - quantized
  - {quant_type}
base_model: {source_repo}
---

# {base_name}-{quant_type}

Quantized ONNX model for streaming speech recognition, derived from
[{source_repo}](https://huggingface.co/{source_repo}) (nemotron-speech-streaming-en-0.6b).

## Quantization Method

{method}

## Files

| File | Description |
|------|-------------|
| `encoder.{quant_type}.onnx` | Quantized encoder (stateful, cache-aware streaming) |
| `decoder_joint.{quant_type}.onnx` | Quantized decoder + joint network |
| `tokenizer.model` | SentencePiece tokenizer (unchanged from source) |

## Usage

These models are designed for use with [parakeet-rs](https://huggingface.co/{source_repo})
or compatible ONNX Runtime inference pipelines. The encoder is stateful with cache tensors
for streaming inference (`cache_last_channel`, `cache_last_time`, `cache_last_channel_len`).

## Source

Quantized from the ONNX models in [{source_repo}](https://huggingface.co/{source_repo})
subdirectory `nemotron-speech-streaming-en-0.6b/`.
"""


def make_model_card(quant_type: str, method: str) -> str:
    return MODEL_CARD_TEMPLATE.format(
        quant_type=quant_type,
        base_name=BASE_NAME,
        source_repo=SOURCE_REPO,
        method=method,
    )


def main():
    api = HfApi()

    for quant_type, config in QUANT_CONFIGS.items():
        repo_id = config["repo_id"]
        quant_dir = os.path.join(OUTPUT_DIR, quant_type)

        if not os.path.isdir(quant_dir):
            print(f"SKIP: {quant_dir} does not exist. Run quantize.py first.")
            continue

        files = os.listdir(quant_dir)
        if not files:
            print(f"SKIP: {quant_dir} is empty.")
            continue

        print(f"\nUploading {quant_type} to {repo_id}...")

        # Create repo if needed
        api.create_repo(repo_id=repo_id, exist_ok=True, repo_type="model")

        # Upload model card
        card = make_model_card(quant_type, config["method"])
        api.upload_file(
            path_or_fileobj=card.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=repo_id,
            commit_message=f"Add model card for {quant_type} quantized model",
        )

        # Upload model files
        api.upload_folder(
            folder_path=quant_dir,
            repo_id=repo_id,
            commit_message=f"Upload {quant_type} quantized ONNX models",
        )

        print(f"  Uploaded to https://huggingface.co/{repo_id}")

    print("\nDone.")


if __name__ == "__main__":
    main()
