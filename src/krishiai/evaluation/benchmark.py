"""Latency, throughput, and hardware benchmarking engine for KrishiAI models (Section 63/64)."""
from __future__ import annotations

import gc
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from PIL import Image

from krishiai.inference.model_manager import ModelManager
from krishiai.utils.logging import get_logger

logger = get_logger(__name__)


def run_latency_benchmark(
    model_manager: Optional[ModelManager] = None,
    image_size: tuple[int, int] = (640, 640),
    sample_images: Optional[List[Union[str, Path, Image.Image]]] = None,
    plant_species: Optional[str] = None,
    num_warmup: int = 10,
    num_runs: int = 50,
    output_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Runs high-precision latency profiling across preprocessing, inference, and postprocessing."""
    if model_manager is None:
        model_manager = ModelManager()

    # Prepare benchmark images
    test_images: List[Image.Image] = []
    if sample_images:
        for item in sample_images:
            if isinstance(item, (str, Path)):
                test_images.append(Image.open(item).convert("RGB"))
            elif isinstance(item, Image.Image):
                test_images.append(item.convert("RGB"))
    else:
        # Generate synthetic leaf RGB image
        test_images = [Image.new("RGB", image_size, color=(34, 139, 34))]

    logger.info(
        f"Starting Latency Benchmark: {num_warmup} warmup iterations, {num_runs} measurement runs | Device: {model_manager.device}"
    )

    # 1. Warmup Phase
    for i in range(num_warmup):
        img = test_images[i % len(test_images)]
        _ = model_manager.predict(img, plant_species=plant_species)

    gc.collect()

    # 2. Measurement Phase
    preproc_times: List[float] = []
    infer_times: List[float] = []
    postproc_times: List[float] = []
    total_times: List[float] = []
    e2e_durations: List[float] = []

    for i in range(num_runs):
        img = test_images[i % len(test_images)]
        t_start = time.perf_counter()
        resp = model_manager.predict(img, plant_species=plant_species)
        t_end = time.perf_counter()

        e2e_durations.append((t_end - t_start) * 1000.0)
        preproc_times.append(resp.latency_ms.preprocessing)
        infer_times.append(resp.latency_ms.inference)
        postproc_times.append(resp.latency_ms.postprocessing)
        total_times.append(resp.latency_ms.total)

    def stats(arr: List[float]) -> Dict[str, float]:
        np_arr = np.array(arr)
        return {
            "mean_ms": round(float(np.mean(np_arr)), 2),
            "std_ms": round(float(np.std(np_arr)), 2),
            "median_p50_ms": round(float(np.percentile(np_arr, 50)), 2),
            "p90_ms": round(float(np.percentile(np_arr, 90)), 2),
            "p95_ms": round(float(np.percentile(np_arr, 95)), 2),
            "p99_ms": round(float(np.percentile(np_arr, 99)), 2),
            "min_ms": round(float(np.min(np_arr)), 2),
            "max_ms": round(float(np.max(np_arr)), 2),
        }

    e2e_stats = stats(e2e_durations)
    avg_total = e2e_stats["mean_ms"]
    throughput_fps = round(1000.0 / avg_total, 2) if avg_total > 0 else 0.0

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device": model_manager.device,
        "num_runs": num_runs,
        "num_warmup": num_warmup,
        "target_species": plant_species or "auto",
        "throughput_fps": throughput_fps,
        "e2e_latency": e2e_stats,
        "preprocessing_latency": stats(preproc_times),
        "inference_latency": stats(infer_times),
        "postprocessing_latency": stats(postproc_times),
        "pipeline_total_latency": stats(total_times),
    }

    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        fname = f"benchmark_{model_manager.device}_{plant_species or 'auto'}_{int(time.time())}.json"
        (out_path / fname).write_text(json.dumps(report, indent=2))
        logger.info(f"Benchmark report saved -> {out_path / fname}")

    return report
