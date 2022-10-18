"""ONNX ReShape Op."""
from functools import partial

from jax import jit
import jax.numpy as jnp
import jax.numpy as np
from jonnx.core import node
from jonnx.core import tensor
from jonnx.utils import registry


@registry.register_op('Reshape')
class Reshape(node.Node):

  def __call__(self, x, shape = None):
    if not shape:
      shape = self.attribute.get('shape', None)
    assert shape, "shape is None."
    return [jnp.reshape(x, tuple(shape))]
