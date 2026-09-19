"""CLI tool for evaluating KrishiAI model robustness under environmental perturbations (Section 65)."""
import argparse
from pathlib import Path
import random
import sys

from krishiai.evaluation.robustness import run_robustness_evaluation
from krishiai.inference.model_manager import ModelManager


def main():
    parser = argparse.ArgumentParser(description="KrishiAI Robustness & Perturbation Suite")
    parser.add_argument("--image-dir", type=str, default="data/final/classification/test", help="Directory of test images")
    parser.add_argument("--max-images", type=int, default=20, help="Maximum number of test images to evaluate")
    parser.add_argument("--det-model", type=str, default=None, help="Path to detection model weights (.pt)")
    parser.add_argument("--cls-model", type=str, default=None, help="Path to classification model weights (.pt)")
    parser.add_argument("--device", type=str, default="auto", help="Compute device (auto, cpu, cuda, mps)")
    parser.add_argument("--output-dir", type=str, default="experiments/robustness", help="Output directory for reports")
    args = parser.parse_args()

    img_dir = Path(args.image_dir)
    if not img_dir.exists():
        print(f"Error: Directory '{img_dir}' not found.")
        sys.exit(1)

    all_images = [p for p in img_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]
    if not all_images:
        print(f"No valid images found in {img_dir}.")
        sys.exit(1)

    selected = random.sample(all_images, min(len(all_images), args.max_images))

    print(f"\n============================================================")
    print(f"KrishiAI Environmental Robustness Suite")
    print(f"Test Set: {len(selected)} images from {img_dir}")
    print(f"Output Directory: {args.output_dir}")
    print(f"============================================================\n")

    mm = ModelManager.create_default(
        detection_model_path=args.det_model,
        classification_model_path=args.cls_model,
        device=args.device,
    )
    report = run_robustness_evaluation(
        images=selected,
        model_manager=mm,
        output_dir=args.output_dir,
    )

    print("\n----------------- ROBUSTNESS REPORT -----------------")
    for category, variants in report["perturbations"].items():
        print(f"\n[Category: {category.upper()}]")
        for var_name, metrics in variants.items():
            print(f"  - {var_name:<24} | Agreement: {metrics['label_agreement_rate']*100:.1f}% | Conf Ret: {metrics['avg_confidence_retention']*100:.1f}% | Uncertainty: {metrics['uncertainty_trigger_rate']*100:.1f}%")
    print("\n-----------------------------------------------------\n")


if __name__ == "__main__":
    main()
