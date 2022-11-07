"""Tests for backend."""
from absl.testing import absltest
from jax import numpy as jnp
from jonnx.backend import JaxBackend
from jonnx.core import node
from jonnx.utils import registry
import numpy as np
import onnx


@registry.register_op('_Relu')
class Relu(node.Node):

  def __call__(self, x):
    return [jnp.maximum(x, 0)]


class BackendTest(absltest.TestCase):

  def _create_dummy_model(self):
    """Create a dummy model for test."""
    model_input_name = 'X'
    model_input_0 = onnx.helper.make_tensor_value_info(model_input_name,
                                                       onnx.TensorProto.FLOAT,
                                                       [None, 3, 32, 32])
    model_output_name = 'Y'
    model_output_0 = onnx.helper.make_tensor_value_info(model_output_name,
                                                        onnx.TensorProto.FLOAT,
                                                        [None, 3, 32, 32])

    relu1_output_node_name = 'Y'

    relu1_node = onnx.helper.make_node(
        name='ReLU1',  # Name is optional.
        op_type='Relu',
        inputs=[model_input_name],
        outputs=[relu1_output_node_name],
    )

    # Create the graph (GraphProto)
    graph_def = onnx.helper.make_graph(
        nodes=[relu1_node],
        name='ReluSingleOp',
        inputs=[model_input_0],  # Graph input
        outputs=[model_output_0],  # Graph output
        initializer=[],
    )

    # Create the model (ModelProto)
    model_def = onnx.helper.make_model(graph_def)
    model_def.opset_import[0].version = 13
    model_def = onnx.shape_inference.infer_shapes(model_def)
    onnx.checker.check_model(model_def)
    return model_def

  def test_single_relu_model(self):
    model = self._create_dummy_model()
    inputs = np.random.randint(-10, 10, [10, 3, 32, 32])
    outputs = JaxBackend.run_model(model, inputs)
    outputs_2 = JaxBackend.run_model(model, inputs, mode='interpreter')
    self.assertTrue((outputs[0] == outputs_2[0]).all())


if __name__ == '__main__':
  absltest.main()
