import numpy as np
import pytest
from safetensors.numpy import save_file

from funasr_mlx.checkpoints import convert_tensor, inspect_checkpoint


def test_metadata_inspection_never_loads_mlx(tmp_path):
    path = tmp_path / "model.safetensors"
    save_file({"linear.weight": np.ones((3, 4), np.float16)}, str(path))
    info = inspect_checkpoint(path)
    assert info.tensor_count == 1
    assert info.tensor_bytes == 24
    assert info.largest_tensor_bytes == 24
    assert info.dtypes == {"F16": 24}


@pytest.mark.parametrize(
    "model,name,shape,expected",
    [
        ("sensevoice", "encoder.a.fsmn_block.weight", (4, 1, 3), (4, 3, 1)),
        ("paraformer", "predictor.cif_conv1d.weight", (4, 4, 3), (4, 3, 4)),
        ("ct-punc", "encoder.a.fsmn_block.weight", (4, 1, 3), (4, 3, 1)),
        ("fsmn-vad", "encoder.a.conv_left.weight", (4, 1, 3, 1), (4, 3, 1)),
        ("campplus", "head.conv.weight", (4, 1, 3, 3), (4, 3, 3, 1)),
        ("emotion2vec", "encoder.conv.weight", (4, 4, 3), (4, 3, 4)),
        ("whisper", "encoder.conv1.weight", (4, 4, 3), (4, 3, 4)),
    ],
)
def test_layout_rules(model, name, shape, expected):
    assert convert_tensor(name, np.zeros(shape), model).shape == expected


def test_invalid_fsmn_layout_is_rejected():
    with pytest.raises(ValueError, match="D,1,K,1"):
        convert_tensor("x.conv_left.weight", np.zeros((4, 2, 3, 1)), "fsmn-vad")


def test_duplicate_shard_keys_are_rejected(tmp_path):
    for filename in ("a.safetensors", "b.safetensors"):
        save_file({"duplicate": np.zeros(1)}, str(tmp_path / filename))
    with pytest.raises(ValueError, match="Duplicate"):
        inspect_checkpoint(tmp_path)


def test_truncated_header_is_rejected(tmp_path):
    path = tmp_path / "bad.safetensors"
    path.write_bytes(b"bad")
    with pytest.raises(ValueError, match="Truncated"):
        inspect_checkpoint(path)
