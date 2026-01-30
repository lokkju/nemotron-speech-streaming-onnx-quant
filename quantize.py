"""Quantize Nemotron Speech Streaming ONNX models to int8 and int4."""

import os
import shutil

import onnx
from onnxruntime.quantization import QuantType, quantize_dynamic
from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer

SOURCE_DIR = "source_model"
OUTPUT_DIR = "output"

MODELS = ["encoder.onnx", "decoder_joint.onnx"]
COPY_FILES = ["tokenizer.model"]


def quantize_int8(input_path: str, output_path: str):
    """Apply dynamic int8 quantization."""
    print(f"  int8: {input_path} -> {output_path}")
    quantize_dynamic(
        model_input=input_path,
        model_output=output_path,
        weight_type=QuantType.QInt8,
    )


def quantize_int4(input_path: str, output_path: str):
    """Apply weight-only int4 quantization using MatMulNBitsQuantizer."""
    print(f"  int4: {input_path} -> {output_path}")
    model = onnx.load(input_path, load_external_data=True)
    quantizer = MatMulNBitsQuantizer(model)
    quantizer.process()
    onnx.save_model(quantizer.model.model, output_path)


def verify_io_signatures(original_path: str, quantized_path: str, label: str):
    """Verify that quantized model preserves I/O tensor names and counts."""
    import onnxruntime as ort

    orig_sess = ort.InferenceSession(original_path, providers=["CPUExecutionProvider"])
    quant_sess = ort.InferenceSession(quantized_path, providers=["CPUExecutionProvider"])

    orig_inputs = [i.name for i in orig_sess.get_inputs()]
    quant_inputs = [i.name for i in quant_sess.get_inputs()]
    orig_outputs = [o.name for o in orig_sess.get_outputs()]
    quant_outputs = [o.name for o in quant_sess.get_outputs()]

    ok = True
    if orig_inputs != quant_inputs:
        print(f"  WARNING [{label}]: Input mismatch!")
        print(f"    Original: {orig_inputs}")
        print(f"    Quantized: {quant_inputs}")
        ok = False
    if orig_outputs != quant_outputs:
        print(f"  WARNING [{label}]: Output mismatch!")
        print(f"    Original: {orig_outputs}")
        print(f"    Quantized: {quant_outputs}")
        ok = False
    if ok:
        print(f"  OK [{label}]: I/O signatures match")
    return ok


def main():
    for model_file in MODELS:
        src = os.path.join(SOURCE_DIR, model_file)
        if not os.path.exists(src):
            print(f"ERROR: Source model not found: {src}")
            print("Run download.py first.")
            return

    # Create output directories
    int8_dir = os.path.join(OUTPUT_DIR, "int8")
    int4_dir = os.path.join(OUTPUT_DIR, "int4")
    os.makedirs(int8_dir, exist_ok=True)
    os.makedirs(int4_dir, exist_ok=True)

    # Quantize each model
    for model_file in MODELS:
        src = os.path.join(SOURCE_DIR, model_file)
        base = model_file.replace(".onnx", "")

        print(f"\nQuantizing {model_file}...")

        # int8
        int8_out = os.path.join(int8_dir, f"{base}.int8.onnx")
        quantize_int8(src, int8_out)

        # int4
        int4_out = os.path.join(int4_dir, f"{base}.int4.onnx")
        quantize_int4(src, int4_out)

    # Verify I/O signatures
    print("\nVerifying I/O signatures...")
    for model_file in MODELS:
        src = os.path.join(SOURCE_DIR, model_file)
        base = model_file.replace(".onnx", "")

        verify_io_signatures(src, os.path.join(int8_dir, f"{base}.int8.onnx"), f"{base} int8")
        verify_io_signatures(src, os.path.join(int4_dir, f"{base}.int4.onnx"), f"{base} int4")

    # Copy tokenizer and other files
    for copy_file in COPY_FILES:
        src = os.path.join(SOURCE_DIR, copy_file)
        if os.path.exists(src):
            for subdir in [int8_dir, int4_dir]:
                dst = os.path.join(subdir, copy_file)
                shutil.copy2(src, dst)
                print(f"\nCopied {copy_file} -> {dst}")

    # Print output sizes
    print("\n--- Output Summary ---")
    for subdir_name in ["int8", "int4"]:
        subdir = os.path.join(OUTPUT_DIR, subdir_name)
        total = 0
        print(f"\n{subdir_name}/")
        for f in sorted(os.listdir(subdir)):
            path = os.path.join(subdir, f)
            size = os.path.getsize(path)
            total += size
            print(f"  {f}: {size / (1024*1024):.1f} MB")
        print(f"  Total: {total / (1024*1024):.1f} MB")


if __name__ == "__main__":
    main()
