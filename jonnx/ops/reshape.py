"""ONNX ReShape Op."""
import jax.numpy as jnp
from jonnx.utils import registry


@registry.register_op('ReShape')
def reshape(x, shape, **kwargs):
  return [jnp.reshape(x, shape)],
