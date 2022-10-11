"""ONNX Pool ops."""
from jax import lax
from jax import numpy as jnp
from jonnx.utils import registry


@registry.register_op('MaxPool')
def onnx_maxpool(x, kernel_shape, pads=None, strides=None, **kwargs):
  """Numpy-backed implementation of ONNX MaxPool op."""
  prefix = (1,) * (x.ndim - len(kernel_shape))
  dims = prefix + tuple(kernel_shape)
  pads = tuple(pads) if pads else [0] * len(kernel_shape)
  strides = (prefix + tuple(strides)) if strides else [1] * len(kernel_shape)
  return [lax.reduce_window(x, -jnp.inf, lax.max, dims, strides, 'VALID')]
