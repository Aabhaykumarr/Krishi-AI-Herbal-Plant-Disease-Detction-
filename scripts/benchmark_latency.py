"""CLI entrypoint for KrishiAI hardware and latency benchmarking (Section 63/64)."""
import argparse
from pathlib import Path

from krishiai.evaluation.benchmark import run_latency_benchmark
from krishiai.inference.model_manager import ModelManager


def main():
    parser = argparse.ArgumentParser(description="KrishiAI Latency & Throughput Benchmark")
    parser.add_argument("--runs", type=int, default=50, help="Number of measurement runs")
    parser.add_argument("--warmup", type=int, default=10, help="Number of warmup iterations")
    parser.add_argument("--species", type=str, default=None, choices=["basil", "aloe_vera", "mint"], help="Target crop species pipeline")
    parser.add_argument("--det-model", type=str, default=None, help="Path to detection model weights (.pt)")
    parser.add_argument("--cls-model", type=str, default=None, help="Path to classification model weights (.pt)")
    parser.add_argument("--device", type=str, default="auto", help="Compute device (auto, cpu, cuda, mps)")
    parser.add_argument("--output-dir", type=str, default="experiments/benchmarks", help="Output directory for benchmark reports")
    args = parser.parse_args()

    print(f"\n============================================================")
    print(f"KrishiAI Performance & Latency Benchmark")
    print(f"Warmup Iterations: {args.warmup}")
    print(f"Measurement Runs: {args.runs}")
    print(f"Species Pipeline: {args.species or 'Auto-Detect'}")
    print(f"============================================================\n")

    mm = ModelManager.create_default(
        detection_model_path=args.det_model,
        classification_model_path=args.cls_model,
        device=args.device,
    )
    report = run_latency_benchmark(
        model_manager=mm,
        plant_species=args.species,
        num_warmup=args.warmup,
        num_runs=args.runs,
        output_dir=args.output_dir,
    )

    e2e = report["e2e_latency"]
    inf = report["inference_latency"]

    print("\n----------------- BENCHMARK RESULTS -----------------")
    print(f"Device:               {report['device'].upper()}")
    print(f"Throughput:           {report['throughput_fps']} FPS")
    print(f"Mean E2E Latency:     {e2e['mean_ms']} ms ± {e2e['std_ms']} ms")
    print(f"Median (P50) Latency: {e2e['median_p50_ms']} ms")
    print(f"P95 Latency:          {e2e['p95_ms']} ms")
    print(f"P99 Latency:          {e2e['p99_ms']} ms")
    print(f"Pure Inference Mean:  {inf['mean_ms']} ms")
    print("-----------------------------------------------------\n")


if __name__ == "__main__":
    main()
