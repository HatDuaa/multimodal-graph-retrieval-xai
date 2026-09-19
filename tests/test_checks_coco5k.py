"""Pure-array checks for the shared COCO 5K retrieval direction logic."""
import numpy as np
import pytest


def test_evaluate_coco5k_both_directions() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("faiss")
    from scripts.checks.coco5k_zero_shot import evaluate_coco5k

    images = np.eye(2, dtype=np.float32)
    texts = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    caps = [("a", "first", 10), ("b", "second", 10), ("c", "third", 20)]
    result = evaluate_coco5k(images, texts, [10, 20], caps)
    assert result["text_to_image"]["recall@1"] == pytest.approx(1.0)
    assert result["image_to_text"]["recall@1"] == pytest.approx(1.0)
