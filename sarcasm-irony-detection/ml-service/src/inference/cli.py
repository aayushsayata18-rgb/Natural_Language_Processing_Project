"""
Command-Line Interface (CLI) for Sarcasm & Irony Detection.
Supports single text inference, batch CSV scoring, benchmark, and artifact verification.
"""

import argparse
import json
import os
import sys
import time
import pandas as pd

from .predictor import SarcasmIronyPredictor
from .exceptions import InferenceError, InvalidInputError
from .utils import verify_artifact_checksums, get_execution_device


def format_table(data: dict) -> str:
    """Formats prediction dictionary as a clean ASCII block."""
    lines = [
        "=" * 60,
        " SARCASM & IRONY CLASSIFICATION RESULT",
        "=" * 60,
        f"Input Text:       \"{data['text']}\"",
        f"Predicted Class:  {data['predicted_class']}",
        f"Model Confidence: {data['confidence']:.2%}",
        f"Model & Version:  {data['model_name']} ({data['model_version']})",
        f"Latency:          {data.get('latency_ms', 0):.3f} ms",
        "-" * 60,
        " Class Probability Distribution:",
    ]
    for cls_name, prob in data["probabilities"].items():
        bar_len = int(prob * 30)
        bar = "#" * bar_len + "-" * (30 - bar_len)
        lines.append(f"  - {cls_name:<12}: {prob:>6.2%}  [{bar}]")
    lines.append("=" * 60)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="CLI inference tool for Sarcasm & Irony Detection System (Phase 10)"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--text", "-t", type=str, help="Single text input to analyze"
    )
    group.add_argument(
        "--file", "-f", type=str, help="Path to input CSV containing a 'text' column"
    )
    group.add_argument(
        "--verify", "-v", action="store_true", help="Verify integrity and checksums of packaged model artifacts"
    )
    group.add_argument(
        "--benchmark", "-b", action="store_true", help="Run quick latency and throughput benchmark"
    )

    parser.add_argument(
        "--output", "-o", type=str, default="predictions.csv", help="Path to save output CSV in batch mode"
    )
    parser.add_argument(
        "--json", action="store_true", help="Output results in JSON format instead of human-readable text"
    )
    parser.add_argument(
        "--model-dir", type=str, default=None, help="Optional path to custom models/final directory"
    )

    args = parser.parse_args()

    # 1. Verification mode
    if args.verify:
        try:
            predictor = SarcasmIronyPredictor(model_dir=args.model_dir)
            ok, results = verify_artifact_checksums(predictor.bundle.model_dir)
            print("[SUCCESS] All model artifacts verified successfully against checksum manifest.")
            for rel_path, res in results.items():
                print(f"  - {rel_path}: {res['sha256'][:16]}... (OK)")
            sys.exit(0)
        except Exception as e:
            print(f"[ERROR] Artifact verification failed: {e}", file=sys.stderr)
            sys.exit(1)

    # 2. Benchmark mode
    if args.benchmark:
        try:
            print("Initializing predictor for benchmark...")
            t0 = time.perf_counter()
            predictor = SarcasmIronyPredictor(model_dir=args.model_dir)
            load_time_ms = (time.perf_counter() - t0) * 1000
            print(f"Model load time: {load_time_ms:.2f} ms")

            bench_sample = "Fantastic, another three-hour Monday morning meeting."
            # Warm up
            for _ in range(50):
                predictor.predict(bench_sample)

            iterations = 1000
            t_start = time.perf_counter()
            for _ in range(iterations):
                predictor.predict(bench_sample)
            total_duration = time.perf_counter() - t_start
            avg_ms = (total_duration / iterations) * 1000
            throughput = iterations / total_duration

            dev = get_execution_device()
            print("=" * 50)
            print(f"Device:           {dev['device']} ({dev['processor']})")
            print(f"Single Latency:   {avg_ms:.4f} ms per prediction")
            print(f"Throughput:       {throughput:,.1f} predictions/sec")
            print("=" * 50)
            sys.exit(0)
        except Exception as e:
            print(f"[ERROR] Benchmark failed: {e}", file=sys.stderr)
            sys.exit(1)

    # 3. Single text mode
    if args.text is not None:
        try:
            predictor = SarcasmIronyPredictor(model_dir=args.model_dir)
            result = predictor.predict(args.text)
            if args.json:
                print(json.dumps(result.to_dict(), indent=2))
            else:
                print(format_table(result.to_dict()))
            sys.exit(0)
        except InvalidInputError as e:
            print(f"[VALIDATION ERROR] {e}", file=sys.stderr)
            sys.exit(2)
        except Exception as e:
            print(f"[ERROR] Inference failed: {e}", file=sys.stderr)
            sys.exit(1)

    # 4. Batch file mode
    if args.file is not None:
        if not os.path.exists(args.file):
            print(f"[ERROR] Input file not found: {args.file}", file=sys.stderr)
            sys.exit(1)

        try:
            df = pd.read_csv(args.file)
            col_name = "text" if "text" in df.columns else df.columns[0]
            texts = df[col_name].astype(str).tolist()

            predictor = SarcasmIronyPredictor(model_dir=args.model_dir)
            batch_result = predictor.predict_batch(texts)

            records = []
            for res in batch_result.predictions:
                r = {
                    "text": res.text,
                    "predicted_class": res.predicted_class,
                    "confidence": res.confidence,
                    "probability_regular": res.probabilities.get("Regular", 0.0),
                    "probability_sarcasm": res.probabilities.get("Sarcasm", 0.0),
                    "probability_irony": res.probabilities.get("Irony", 0.0),
                    "probability_figurative": res.probabilities.get("Figurative", 0.0),
                }
                records.append(r)

            out_df = pd.DataFrame(records)
            out_df.to_csv(args.output, index=False)
            print(f"[SUCCESS] Scored {len(texts)} samples in {batch_result.batch_latency_ms:.2f} ms.")
            print(f"Results saved to: {args.output}")
            sys.exit(0)
        except Exception as e:
            print(f"[ERROR] Batch inference failed: {e}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
