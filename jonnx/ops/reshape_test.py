"""Test Reshape ops."""
from absl.testing import absltest
from jax._src import test_util as jtu
from jonnx.utils.test_utils import expect
import numpy as np
import onnx


def reshape_reference_implementation(data: np.ndarray,
                                     shape: np.ndarray,
                                     allowzero: int = 0) -> np.ndarray:
  new_shape = np.copy(shape)
  if allowzero == 0:
    zeros_index = np.where(shape == 0)
    new_shape[zeros_index] = np.array(data.shape)[zeros_index]
  reshaped = np.reshape(data, new_shape)
  return reshaped


class ReshapeTest(jtu.JaxTestCase):

  def test_export_reshape(self):
    original_shape = [2, 3, 4]
    test_cases = {
        "reordered_all_dims": np.array([4, 2, 3], dtype=np.int64),
        "reordered_last_dims": np.array([2, 4, 3], dtype=np.int64),
        "reduced_dims": np.array([2, 12], dtype=np.int64),
        "extended_dims": np.array([2, 3, 2, 2], dtype=np.int64),
        "one_dim": np.array([24], dtype=np.int64),
        "negative_dim": np.array([2, -1, 2], dtype=np.int64),
        "negative_extended_dims": np.array([-1, 2, 3, 4], dtype=np.int64),
    }
    data = np.random.random_sample(original_shape).astype(np.float32)

    for test_name, shape in test_cases.items():
      node = onnx.helper.make_node(
          "Reshape",
          inputs=["data", "shape"],
          outputs=["reshaped"],
      )

      reshaped = reshape_reference_implementation(data, shape)

      expect(
          node,
          inputs=[data, shape],
          outputs=[reshaped],
          name="test_reshape_" + test_name,
      )


if __name__ == "__main__":
  absltest.main()
