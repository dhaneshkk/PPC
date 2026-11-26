from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ppc import pipeline

ROOT = Path(__file__).resolve().parents[1]


def _load_demo():
    module_name = "ppc_demo_for_webcam_tests"
    spec = importlib.util.spec_from_file_location(module_name, ROOT / "demo_ppc.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _box(classid: int, x1: float, y1: float, x2: float, y2: float):
    return SimpleNamespace(classid=classid, x1=x1, y1=y1, x2=x2, y2=y2, score=0.9)


@pytest.mark.parametrize(
    "coords",
    [(40, 30, 60, 54), (0, 0, 10, 12), (110, 70, 120, 80), (55.4, 33.7, 61.2, 41.9)],
)
def test_expanded_crop_matches_demo(coords) -> None:
    demo = _load_demo()
    frame = np.random.default_rng(0).integers(0, 255, size=(80, 120, 3), dtype=np.uint8)
    box = _box(pipeline.HAND_CLASS_ID, *coords)
    crop, xyxy = pipeline._extract_expanded_crop(frame, box, expansion=pipeline.DEFAULT_HAND_CROP_EXPANSION)
    expected = demo.crop_image_with_expansion(image=frame, box=box, expansion=demo.PHONE_USAGE_CROP_EXPANSION)
    assert crop is not None and xyxy is not None
    np.testing.assert_array_equal(crop, expected)
    x1, y1, x2, y2 = xyxy
    np.testing.assert_array_equal(frame[y1:y2, x1:x2], crop)


def test_pipeline_hand_constants_match_demo() -> None:
    demo = _load_demo()
    assert pipeline.HAND_CLASS_ID == demo.HAND_CLASS_ID
    assert pipeline.DEFAULT_HAND_CROP_EXPANSION == demo.PHONE_USAGE_CROP_EXPANSION
    assert hasattr(demo, "WholeBodyDetector")


def test_detect_hand_crops_ignores_non_hand_boxes() -> None:
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    boxes = [
        _box(19, 10, 10, 20, 20),  # mouth: previously the only region classified
        _box(pipeline.HAND_CLASS_ID, 40, 40, 50, 50),
        _box(pipeline.HAND_CLASS_ID, 70, 10, 80, 20),
        _box(0, 0, 0, 100, 100),  # body
    ]
    detector = lambda **kwargs: boxes  # noqa: E731
    crops = pipeline._detect_hand_crops(
        detector, frame, hand_class_id=pipeline.HAND_CLASS_ID, expansion=2.5
    )
    assert [coords for _, coords in crops] == [(32, 32, 58, 58), (62, 2, 88, 28)]


def test_tensorrt_provider_keeps_options() -> None:
    providers = pipeline._resolve_onnx_providers("tensorrt")
    assert isinstance(providers[0], tuple) and providers[0][0] == "TensorrtExecutionProvider"
    assert providers[-1] == "CPUExecutionProvider"
