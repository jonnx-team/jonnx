"""Those simple ONNX ops."""
import jax
from jax import lax
from jax import numpy as jnp
import numpy as np
from jonnx.core import node
from jonnx.utils import registry
from numpy import broadcast
from onnx import numpy_helper
from onnx import NodeProto


@registry.register_op('Abs')
class Abs(node.Node):

  def __call__(self, x):
    return [lax.abs(x)]


@registry.register_op('Acos')
class Acos(node.Node):

  def __call__(self, x):
    return [lax.acos(x)]


@registry.register_op('Acosh')
class Acosh(node.Node):

  def __call__(self, x):
    return [lax.acosh(x)]


@registry.register_op('Constant')
class Constant(node.Node):

  def __call__(self):
    output_name = self.output[0]
    # TODO: only support 'value' currently
    # https://github.com/onnx/onnx/blob/main/docs/Operators.md#constant
    key = 'value'
    assert key in self.attribute, f'node attribute keys include {keys(self.attribute)}'
    tensor_proto = self.attribute[key]
    return [numpy_helper.to_array(tensor_proto)]


@registry.register_op('MatMul')
class MatMul(node.Node):

  def __call__(self, x, y):
    return [jnp.matmul(x, y)]


@registry.register_op('Add')
class Add(node.Node):

  def __call__(self, a, b):
    """Numpy-backed implementation of ONNX Add op."""
    axis = self.attribute.get('axis', None)
    broadcast = self.attribute.get('broadcast', None)
    if broadcast:
      axis = (a.dim - b.ndim) if axis is None else axis % a.ndim
      assert a.shape[axis:][:b.ndim] == b.shape
      b_shape = np.ones(a.ndim, dtype='int64')
      b_shape[axis:axis + b.ndim] = b.shape
      b = jnp.reshape(b, b_shape)
    return [a + b]


@registry.register_op('Relu')
class Relu(node.Node):

  def __call__(self, x):
    return [jnp.maximum(x, 0)]
