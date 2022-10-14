"""Those simple ONNX ops."""
from jax import lax
from jax import numpy as jnp
from jonnx.core import node
from jonnx.utils import registry
from onnx import numpy_helper


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
    tensor_proto = self.attribute[output_name]
    return [numpy_helper.to_array(tensor_proto)]


@registry.register_op('MatMul')
class MatMul(node.Node):

  def __call__(self, x, y):
    return [jnp.matmul(x, y)]
