from pathlib import Path
import pytest
from krishiai.training.trainer import KrishiTrainer, get_optimal_device


def test_get_optimal_device():
    dev = get_optimal_device("cpu")
    assert dev == "cpu"
    auto_dev = get_optimal_device("auto")
    assert auto_dev in {"0", "cpu", "mps"}


def test_krishi_trainer_init_with_dict():
    cfg = {
        "model": "models/pretrained/yolo26n.pt",
        "epochs": 1,
        "device": "cpu",
    }
    trainer = KrishiTrainer(config=cfg, task="detect")
    assert trainer.task == "detect"
    assert trainer.device == "cpu"
    assert trainer.model is not None


def test_krishi_trainer_missing_config():
    with pytest.raises(FileNotFoundError):
        KrishiTrainer(config=Path("nonexistent_config.yaml"))
