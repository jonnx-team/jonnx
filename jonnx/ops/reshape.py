"""ONNX ReShape Op."""
from functools import partial

import jax.numpy as jnp
import jax.numpy as np
from jonnx.core import node
from jonnx.core import tensor
from jonnx.utils import registry


@registry.register_op('Reshape')
class Reshape(node.Node):

  def __call__(self, x, shape=None):
    if shape is None:
      shape = self.attribute.get('shape', None)
    return [jnp.reshape(x, tuple(shape))]
