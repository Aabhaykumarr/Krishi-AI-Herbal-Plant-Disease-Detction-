"""Unit tests for benchmarking, robustness, and batch inference (Section 53, 63-65)."""
from pathlib import Path
import pytest
from PIL import Image

from krishiai.evaluation.benchmark import run_latency_benchmark
from krishiai.evaluation.robustness import (
    apply_brightness,
    apply_contrast,
    apply_gaussian_blur,
    apply_gaussian_noise,
    apply_occlusion,
    apply_rotation,
    run_robustness_evaluation,
)
from krishiai.inference.batch import annotate_image_with_prediction, run_batch_inference
from krishiai.inference.model_manager import ModelManager


@pytest.fixture
def sample_image():
    return Image.new("RGB", (128, 128), color=(34, 139, 34))


def test_perturbation_transforms(sample_image):
    b = apply_brightness(sample_image, 1.5)
    assert b.size == sample_image.size

    c = apply_contrast(sample_image, 0.8)
    assert c.size == sample_image.size

    g = apply_gaussian_blur(sample_image, 2.0)
    assert g.size == sample_image.size

    n = apply_gaussian_noise(sample_image, 15.0)
    assert n.size == sample_image.size

    r = apply_rotation(sample_image, 45.0)
    assert r.size == sample_image.size

    o = apply_occlusion(sample_image, 0.3)
    assert o.size == sample_image.size


def test_run_latency_benchmark(sample_image, tmp_path):
    mm = ModelManager(classification_model_path="models/pretrained/yolo26n-cls.pt", device="cpu")
    res = run_latency_benchmark(
        model_manager=mm,
        sample_images=[sample_image],
        num_warmup=2,
        num_runs=3,
        output_dir=tmp_path,
    )
    assert "throughput_fps" in res
    assert "e2e_latency" in res
    assert "mean_ms" in res["e2e_latency"]


def test_run_robustness_evaluation(sample_image, tmp_path):
    mm = ModelManager(classification_model_path="models/pretrained/yolo26n-cls.pt", device="cpu")
    res = run_robustness_evaluation(
        images=[sample_image],
        model_manager=mm,
        output_dir=tmp_path,
    )
    assert "perturbations" in res
    assert "brightness" in res["perturbations"]


def test_batch_inference(tmp_path):
    in_dir = tmp_path / "inputs"
    out_dir = tmp_path / "outputs"
    in_dir.mkdir()

    img = Image.new("RGB", (64, 64), color=(0, 255, 0))
    img.save(in_dir / "leaf1.jpg")
    img.save(in_dir / "leaf2.png")

    mm = ModelManager(classification_model_path="models/pretrained/yolo26n-cls.pt", device="cpu")
    res = run_batch_inference(
        input_dir=in_dir,
        output_dir=out_dir,
        model_manager=mm,
        save_annotated=True,
    )
    assert res["total_images"] == 2
    assert (out_dir / "batch_predictions.json").exists()
    assert (out_dir / "batch_summary.csv").exists()
    assert (out_dir / "annotated" / "annotated_leaf1.jpg").exists()
