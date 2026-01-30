"""Download source ONNX model files from HuggingFace."""

import os
from huggingface_hub import hf_hub_download

REPO_ID = "altunenes/parakeet-rs"
SUBDIR = "nemotron-speech-streaming-en-0.6b"
OUTPUT_DIR = "source_model"

FILES = [
    "encoder.onnx",
    "encoder.onnx.data",
    "decoder_joint.onnx",
    "tokenizer.model",
]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for filename in FILES:
        remote_path = f"{SUBDIR}/{filename}"
        local_path = os.path.join(OUTPUT_DIR, filename)

        if os.path.exists(local_path):
            print(f"Already exists: {local_path}")
            continue

        print(f"Downloading {remote_path}...")
        hf_hub_download(
            repo_id=REPO_ID,
            filename=remote_path,
            local_dir=OUTPUT_DIR,
            local_dir_use_symlinks=False,
        )

    # hf_hub_download preserves subdirectory structure, so move files up
    subdir_path = os.path.join(OUTPUT_DIR, SUBDIR)
    if os.path.isdir(subdir_path):
        for filename in FILES:
            src = os.path.join(subdir_path, filename)
            dst = os.path.join(OUTPUT_DIR, filename)
            if os.path.exists(src) and not os.path.exists(dst):
                os.rename(src, dst)
        # clean up empty subdirectory
        try:
            os.removedirs(subdir_path)
        except OSError:
            pass

    print(f"Source model files in {OUTPUT_DIR}/")
    for filename in FILES:
        path = os.path.join(OUTPUT_DIR, filename)
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"  {filename}: {size_mb:.1f} MB")
        else:
            print(f"  {filename}: MISSING")


if __name__ == "__main__":
    main()
