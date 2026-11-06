import importlib.util
from pathlib import Path

import pytest


path = Path(__file__).resolve().parents[1] / "vendor/freetimegs/datasets/camera_split.py"
spec = importlib.util.spec_from_file_location("person4d_camera_split", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_production_split_preserves_every_camera_including_front():
    training, diagnostics, held_out = module.camera_split(48, 0)
    assert training == list(range(48))
    assert diagnostics == [0]
    assert held_out is False


def test_holdout_split_remains_disjoint_and_covers_every_camera():
    training, validation, held_out = module.camera_split(24, 8)
    assert validation == [0, 8, 16]
    assert not set(training) & set(validation)
    assert sorted(training + validation) == list(range(24))
    assert held_out is True
    with pytest.raises(ValueError, match="no training cameras"):
        module.camera_split(6, 1)
