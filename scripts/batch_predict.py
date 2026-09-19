"""CLI tool for batch inference on image directories (Section 53)."""
import argparse
from pathlib import Path
import sys

from krishiai.inference.batch import run_batch_inference
from krishiai.inference.model_manager import ModelManager


def main():
    parser = argparse.ArgumentParser(description="KrishiAI Batch Leaf Disease Inference")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory containing images")
    parser.add_argument("--output-dir", type=str, default="experiments/batch_predictions", help="Directory to save reports & annotated images")
    parser.add_argument("--species", type=str, default=None, choices=["basil", "aloe_vera", "mint"], help="Explicit target crop species")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--no-annotated", action="store_true", help="Skip saving annotated images")
    args = parser.parse_args()

    input_path = Path(args.input_dir)
    if not input_path.exists():
        print(f"Error: Input directory '{input_path}' does not exist.")
        sys.exit(1)

    print(f"\n============================================================")
    print(f"KrishiAI Batch Inference")
    print(f"Input Directory: {input_path}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Species Filter: {args.species or 'Auto-Detect'}")
    print(f"Confidence Threshold: {args.conf}")
    print(f"============================================================\n")

    mm = ModelManager.create_default()
    summary = run_batch_inference(
        input_dir=input_path,
        output_dir=args.output_dir,
        model_manager=mm,
        plant_species=args.species,
        confidence_threshold=args.conf,
        save_annotated=not args.no_annotated,
    )

    print(f"\nBatch Inference Finished!")
    print(f"Total Processed: {summary.get('total_images', 0)} images")
    print(f"Duration: {summary.get('total_time_seconds', 0)}s")
    print(f"Throughput: {summary.get('throughput_fps', 0)} FPS")
    print(f"Condition Distribution: {summary.get('condition_distribution', {})}")


if __name__ == "__main__":
    main()
