"""Validate quantized models against fp32 baseline.

Compares I/O signatures, runs inference with random input,
measures numerical divergence and inference time.
"""

import os
import time

import numpy as np
import onnxruntime as ort

SOURCE_DIR = "source_model"
OUTPUT_DIR = "output"

# Models to validate
ENCODER = "encoder"
DECODER_JOINT = "decoder_joint"

# Target: <560ms per chunk on CPU
LATENCY_TARGET_MS = 560


def load_session(path: str) -> ort.InferenceSession:
    return ort.InferenceSession(path, providers=["CPUExecutionProvider"])


def get_io_names(sess: ort.InferenceSession):
    inputs = [(i.name, i.shape, i.type) for i in sess.get_inputs()]
    outputs = [(o.name, o.shape, o.type) for o in sess.get_outputs()]
    return inputs, outputs


def make_dummy_input(sess: ort.InferenceSession, batch_size: int = 1) -> dict:
    """Create dummy input tensors based on session input metadata."""
    feed = {}
    for inp in sess.get_inputs():
        shape = []
        for dim in inp.shape:
            if isinstance(dim, str) or dim is None:
                # dynamic dimension: use reasonable defaults
                shape.append(batch_size if "batch" in str(dim) else 64)
            else:
                shape.append(dim)

        if "int32" in inp.type.lower():
            feed[inp.name] = np.ones(shape, dtype=np.int32)
        elif "int" in inp.type.lower():
            feed[inp.name] = np.ones(shape, dtype=np.int64)
        else:
            feed[inp.name] = np.random.randn(*shape).astype(np.float32)

    return feed


def compare_outputs(
    ref_outputs: list[np.ndarray],
    quant_outputs: list[np.ndarray],
    output_names: list[str],
) -> dict:
    """Compare reference and quantized outputs, return error metrics."""
    results = {}
    for name, ref, quant in zip(output_names, ref_outputs, quant_outputs):
        if ref.dtype in (np.float32, np.float64, np.float16):
            abs_diff = np.abs(ref.astype(np.float64) - quant.astype(np.float64))
            max_abs = float(np.max(abs_diff))
            mean_abs = float(np.mean(abs_diff))
            # Cosine similarity
            ref_flat = ref.flatten().astype(np.float64)
            quant_flat = quant.flatten().astype(np.float64)
            norm_ref = np.linalg.norm(ref_flat)
            norm_quant = np.linalg.norm(quant_flat)
            if norm_ref > 0 and norm_quant > 0:
                cosine_sim = float(np.dot(ref_flat, quant_flat) / (norm_ref * norm_quant))
            else:
                cosine_sim = 1.0 if norm_ref == norm_quant == 0 else 0.0
            results[name] = {
                "max_abs_error": max_abs,
                "mean_abs_error": mean_abs,
                "cosine_similarity": cosine_sim,
            }
        else:
            exact_match = np.array_equal(ref, quant)
            results[name] = {"exact_match": exact_match}
    return results


def benchmark_inference(sess: ort.InferenceSession, feed: dict, n_runs: int = 20) -> float:
    """Return median inference time in ms."""
    # warmup
    for _ in range(3):
        sess.run(None, feed)

    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        sess.run(None, feed)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

    return float(np.median(times))


def validate_model(model_name: str, quant_type: str):
    """Validate a single quantized model against its fp32 source."""
    src_path = os.path.join(SOURCE_DIR, f"{model_name}.onnx")
    quant_path = os.path.join(OUTPUT_DIR, quant_type, f"{model_name}.{quant_type}.onnx")

    if not os.path.exists(src_path):
        print(f"  SKIP: source not found ({src_path})")
        return False
    if not os.path.exists(quant_path):
        print(f"  SKIP: quantized not found ({quant_path})")
        return False

    print(f"\n  Loading {model_name} ({quant_type})...")
    ref_sess = load_session(src_path)
    quant_sess = load_session(quant_path)

    # Check I/O signatures
    ref_inputs, ref_outputs = get_io_names(ref_sess)
    quant_inputs, quant_outputs = get_io_names(quant_sess)

    ref_in_names = [i[0] for i in ref_inputs]
    quant_in_names = [i[0] for i in quant_inputs]
    ref_out_names = [o[0] for o in ref_outputs]
    quant_out_names = [o[0] for o in quant_outputs]

    io_ok = True
    if ref_in_names != quant_in_names:
        print(f"  FAIL: Input names differ")
        print(f"    fp32:  {ref_in_names}")
        print(f"    {quant_type}: {quant_in_names}")
        io_ok = False
    else:
        print(f"  OK: Input names match: {ref_in_names}")

    if ref_out_names != quant_out_names:
        print(f"  FAIL: Output names differ")
        print(f"    fp32:  {ref_out_names}")
        print(f"    {quant_type}: {quant_out_names}")
        io_ok = False
    else:
        print(f"  OK: Output names match: {ref_out_names}")

    # Check cache tensors specifically for encoder
    if model_name == ENCODER:
        cache_inputs = [n for n in ref_in_names if "cache" in n]
        cache_outputs = [n for n in ref_out_names if "cache" in n]
        if cache_inputs:
            print(f"  OK: Encoder cache inputs present: {cache_inputs}")
        else:
            print(f"  WARNING: No cache inputs found in encoder")
        if cache_outputs:
            print(f"  OK: Encoder cache outputs present: {cache_outputs}")
        else:
            print(f"  WARNING: No cache outputs found in encoder")

    if not io_ok:
        return False

    # Run inference comparison
    print(f"  Running inference comparison...")
    feed = make_dummy_input(ref_sess)
    ref_results = ref_sess.run(None, feed)
    quant_results = quant_sess.run(None, feed)

    metrics = compare_outputs(ref_results, quant_results, ref_out_names)
    for out_name, m in metrics.items():
        if "cosine_similarity" in m:
            cos = m["cosine_similarity"]
            status = "OK" if cos > 0.95 else "WARNING" if cos > 0.8 else "FAIL"
            print(f"  {status}: {out_name} cosine_sim={cos:.6f} "
                  f"max_err={m['max_abs_error']:.6f} mean_err={m['mean_abs_error']:.6f}")
        else:
            status = "OK" if m["exact_match"] else "FAIL"
            print(f"  {status}: {out_name} exact_match={m['exact_match']}")

    # Benchmark
    print(f"  Benchmarking (20 runs)...")
    ref_ms = benchmark_inference(ref_sess, feed)
    quant_ms = benchmark_inference(quant_sess, feed)
    speedup = ref_ms / quant_ms if quant_ms > 0 else float("inf")

    target_status = "OK" if quant_ms < LATENCY_TARGET_MS else "WARNING"
    print(f"  fp32:  {ref_ms:.1f} ms")
    print(f"  {quant_type}: {quant_ms:.1f} ms ({speedup:.2f}x)")
    print(f"  {target_status}: {'Under' if quant_ms < LATENCY_TARGET_MS else 'Over'} "
          f"{LATENCY_TARGET_MS}ms target")

    return io_ok


def main():
    print("=== Nemotron Speech Streaming ONNX Quantization Validation ===\n")

    all_ok = True
    for model_name in [ENCODER, DECODER_JOINT]:
        for quant_type in ["int8", "int4"]:
            print(f"\n--- {model_name} ({quant_type}) ---")
            ok = validate_model(model_name, quant_type)
            if not ok:
                all_ok = False

    print("\n" + ("=" * 50))
    if all_ok:
        print("All validations passed.")
    else:
        print("Some validations had issues. Review output above.")


if __name__ == "__main__":
    main()
