import numpy as np
import pytest
from safetensors.numpy import load_file

from funasr_mlx.checkpoint_stream import write_tensor


@pytest.mark.parametrize("shape,dtype", [((17, 9), "float32"), ((), "int64")])
def test_streamed_tensor_matches_standard_reader(tmp_path, shape, dtype):
    array = np.arange(max(1, int(np.prod(shape)))).reshape(shape).astype(dtype)
    path = tmp_path / "model.safetensors"
    write_tensor(path, "weight", array, dtype=dtype, metadata={"model": "test"}, chunk_bytes=32)
    np.testing.assert_array_equal(load_file(path)["weight"], array)
    assert load_file(path)["weight"].shape == shape


def test_streamed_dtype_conversion_only_changes_dtype(tmp_path):
    array = np.arange(200, dtype=np.float16).reshape(10, 20)
    path = tmp_path / "model.safetensors"
    write_tensor(path, "weight", array, dtype="float32", metadata={}, chunk_bytes=32)
    output = load_file(path)["weight"]
    assert output.dtype == np.float32
    np.testing.assert_array_equal(output, array.astype(np.float32))


def test_streaming_refuses_unbounded_contiguous_copy(tmp_path):
    array = np.ones((20, 20), np.float32).T
    with pytest.raises(ValueError, match="C-contiguous"):
        write_tensor(tmp_path / "model.safetensors", "weight", array, dtype="float32", metadata={})
