"""ONNX op definition."""
from functools import partial
from typing import Sequence, Tuple

import jax
from jax import numpy as jnp
from jax import lax
from jonnx.core import module
from jonnx.core import node
from jonnx.utils import registry


@registry.register_op('BatchNormalization')
class BatchNormalization(node.Node):
  
  
  @partial(jax.jit, static_argnames={'self'})
  def __call__(self, x, s, bias, mean, var):
    epsilon = self.attribute.get('epsilon', 1e-5)
    dims_x = len(x.shape)
    dim_ones = (1,) * (dims_x - 2)
    s = s.reshape(-1, *dim_ones)
    bias = bias.reshape(-1, *dim_ones)
    mean = mean.reshape(-1, *dim_ones)
    var = var.reshape(-1, *dim_ones)
    ot = s * (x - mean) / lax.sqrt(var + epsilon) + bias
    return [ot]

