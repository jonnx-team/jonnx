"""Test ReShape ops."""
from absl.testing import absltest
import jax
from jax._src import test_util as jtu
from jonnx.utils.test_utils import expect
import numpy as np
import onnx
import onnx.backend.test.case.node as backend_test_case_node
from jonnx.backend import run_model

from jonnx.ops import reshape
from jonnx.utils import registry

# TODO(https://github.com/jonnx-team/jonnx/issues/2): register it manually
Reshape = registry.register_op("Reshape")
Reshape(reshape.Reshape)


def reshape_reference_implementation(data: np.ndarray,
                                     shape: np.ndarray,
                                     allowzero: int = 0) -> np.ndarray:
  # replace zeros with corresponding dim size
  # we need to do this because np.reshape doesn't support 0 by default unless 'allowzero' is set
  new_shape = np.copy(shape)
  if allowzero == 0:
    zeros_index = np.where(shape == 0)
    new_shape[zeros_index] = np.array(data.shape)[zeros_index]
  reshaped = np.reshape(data, new_shape)
  return reshaped


class ReshapeTest(jtu.JaxTestCase):

  def test_reshape(self):
    original_shape = [2, 3, 4]
    test_cases = {
        "reordered_all_dims": np.array([4, 2, 3], dtype=np.int64),
        "reordered_last_dims": np.array([2, 4, 3], dtype=np.int64),
        "reduced_dims": np.array([2, 12], dtype=np.int64),
        "extended_dims": np.array([2, 3, 2, 2], dtype=np.int64),
        "one_dim": np.array([24], dtype=np.int64),
    }
    data = np.random.random_sample(original_shape).astype(np.float32)

    for test_name, shape in test_cases.items():
      node = onnx.helper.make_node(
          "Reshape",
          inputs=["data", "shape"],
          outputs=["reshaped"],
      )

      reshaped = reshape_reference_implementation(data, shape)
      inputs=[data, shape]
      outputs=[reshaped]

      backend_test_case_node.expect(node, inputs, outputs, name="test_reshape_" + test_name)
      # pylint: disable=protected-access
      testcase = backend_test_case_node._NodeTestCases.pop()
      # pylint: enable=protected-access
      model = testcase.model
      
      with jax.disable_jit():
        outputs_jax = run_model(model, inputs, static_args= {"shape": shape})
        atol = testcase.atol
        rtol = testcase.rtol
        self.assertCountEqual(list(outputs[0].shape), list(outputs_jax[0].shape))
        self.assertAllClose(outputs, outputs_jax, rtol=rtol, atol=atol)


if __name__ == "__main__":
  absltest.main()
