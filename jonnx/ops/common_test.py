"""Test common ONNX ops."""
from absl.testing import absltest
from jax._src import test_util as jtu
from jonnx.utils.test_utils import expect
import numpy as np
import onnx


class CommonOpTest(jtu.JaxTestCase):

  def test_abs(self):
    node = onnx.helper.make_node(
        "Abs",
        inputs=["x"],
        outputs=["y"],
    )
    x = np.random.randn(3, 4, 5).astype(np.float32)
    y = abs(x)

    expect(node, inputs=[x], outputs=[y], name="test_abs")

  def test_acos(self):
    node = onnx.helper.make_node(
        "Acos",
        inputs=["x"],
        outputs=["y"],
    )

    x = np.array([-0.5, 0, 0.5]).astype(np.float32)
    y = np.arccos(x)
    expect(node, inputs=[x], outputs=[y], name="test_acos_example")

    x = np.random.rand(3, 4, 5).astype(np.float32)
    y = np.arccos(x)
    expect(node, inputs=[x], outputs=[y], name="test_acos")

  def test_acosh(self):
    node = onnx.helper.make_node(
        "Acosh",
        inputs=["x"],
        outputs=["y"],
    )

    x = np.array([10, np.e, 1]).astype(np.float32)
    y = np.arccosh(x)  # expected output [2.99322295,  1.65745449,  0.]
    expect(node, inputs=[x], outputs=[y], name="test_acosh_example")

    x = np.random.uniform(1.0, 10.0, (3, 4, 5)).astype(np.float32)
    y = np.arccosh(x)
    expect(node, inputs=[x], outputs=[y], name="test_acosh")

  def test_constant(self):
    values = np.random.randn(5, 5).astype(np.float32)
    node = onnx.helper.make_node(
        "Constant",
        inputs=[],
        outputs=["value"],
        value=onnx.helper.make_tensor(
            name="const_tensor",
            data_type=onnx.TensorProto.FLOAT,
            dims=values.shape,
            vals=values.flatten().astype(float),
        ),
    )

    expect(node, inputs=[], outputs=[values], name="test_constant")

  def test_matmul(self):
    node = onnx.helper.make_node(
        "MatMul",
        inputs=["a", "b"],
        outputs=["c"],
    )

    # 2d
    a = np.random.randn(3, 4).astype(np.float32)
    b = np.random.randn(4, 3).astype(np.float32)
    c = np.matmul(a, b)
    expect(node, inputs=[a, b], outputs=[c], name="test_matmul_2d")

    # 3d
    a = np.random.randn(2, 3, 4).astype(np.float32)
    b = np.random.randn(2, 4, 3).astype(np.float32)
    c = np.matmul(a, b)
    expect(node, inputs=[a, b], outputs=[c], name="test_matmul_3d")

    # 4d
    a = np.random.randn(1, 2, 3, 4).astype(np.float32)
    b = np.random.randn(1, 2, 4, 3).astype(np.float32)
    c = np.matmul(a, b)
    expect(node, inputs=[a, b], outputs=[c], name="test_matmul_4d")


if __name__ == "__main__":
  absltest.main()
