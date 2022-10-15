"""ONNX ReShape Op."""
from functools import partial

from jax import jit
import jax.numpy as jnp
import jax.numpy as np
from jonnx.core import node
from jonnx.core import tensor
from jonnx.utils import registry


# TODO(https://github.com/jonnx-team/jonnx/issues/2):  static_argnums shape compplain miss hash function.
# Only jit.disable_jit work for this op.
# @registry.register_op('Reshape')
class Reshape(node.Node):

  @partial(jit, static_argnums=(2))
  def __call__(self, x, shape):
    return [jnp.reshape(x, shape)]
